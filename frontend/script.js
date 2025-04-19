// ===============================================================
// Configuration & Constants
// ===============================================================
// const API_BASE = "http://localhost:8000"; // For Local Hosting
const API_BASE = "https://option-strategy-website.onrender.com"; // For Production
const REFRESH_INTERVAL_MS = 3000; // Auto-refresh interval (3 seconds )
const HIGHLIGHT_DURATION_MS = 1500; // How long highlights last

const SELECTORS = {
    assetDropdown: "#asset",
    expiryDropdown: "#expiry",
    spotPriceDisplay: "#spotPriceDisplay", // Matches HTML
    optionChainTableBody: "#optionChainTable tbody",
    strategyTableBody: "#strategyTable tbody",
    updateChartButton: "#updateChartBtn",
    clearPositionsButton: "#clearStrategyBtn", // Corrected ID
    payoffChartContainer: "#payoffChartContainer",
    analysisResultContainer: "#analysisResult",
    // Selectors for metrics list items (targeting the value span)
    metricsList: '.metrics-list', // Container for metrics
    maxProfitDisplay: "#maxProfit .metric-value",
    maxLossDisplay: "#maxLoss .metric-value",
    breakevenDisplay: "#breakeven .metric-value",
    rewardToRiskDisplay: "#rewardToRisk .metric-value",
    netPremiumDisplay: "#netPremium .metric-value",
    costBreakdownContainer: "#costBreakdownContainer", // The <details> element
    costBreakdownList: "#costBreakdownList",          // The <ul> inside <details>
    newsResultContainer: "#newsResult",
    taxInfoContainer: "#taxInfo", // Container for tax details/table
    greeksTable: "#greeksTable", // Selector for the entire table
    greeksTableBody: "#greeksTable tbody", // Body for rows
    greeksTableContainer: '#greeksSection', // The whole section containing the table
    greeksAnalysisSection: '#greeksAnalysisSection', // Section for LLM analysis
    greeksAnalysisResultContainer: '#greeksAnalysisResult', // Container for LLM analysis text
    globalErrorDisplay: "#globalError", // For fetch/network errors
    statusMessageContainer: '#statusMessage', // General status messages (optional - ADD HTML if used)
    warningContainer: '#warningMessage', // For calculation/data warnings
};

// Basic Logger
const logger = {
    debug: (...args) => console.debug('[DEBUG]', ...args),
    info: (...args) => console.log('[INFO]', ...args),
    warn: (...args) => console.warn('[WARN]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
};

// Assign logger to window for easy access in helpers if needed
window.logger = logger;

// ===============================================================
// Global State
// ===============================================================
let currentSpotPrice = 0;
let strategyPositions = []; // Holds objects like: { strike_price: number, expiry_date: string, option_type: 'CE'|'PE', lots: number, tr_type: 'b'|'s', last_price: number, iv: number|null, days_to_expiry: number|null, lot_size: number|null }
let activeAsset = null;
let autoRefreshIntervalId = null; // Timer ID for auto-refresh
let previousOptionChainData = {}; // Store previous chain data for highlighting
let previousSpotPrice = 0; // Store previous spot price for highlighting


// ===============================================================
// UTILITY FUNCTIONS (Define First)
// ===============================================================

/** Safely formats a number or returns a fallback string, handling backend specials */
function formatNumber(value, decimals = 2, fallback = "N/A") {
    if (value === null || typeof value === 'undefined') { return fallback; }
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
        if (["∞", "INFINITY"].includes(upperVal)) return "∞";
        if (["-∞", "-INFINITY"].includes(upperVal)) return "-∞";
        if (["N/A", "UNDEFINED", "LOSS", "0 / 0", "∞ / ∞", "LOSS / ∞"].includes(upperVal)) return value;
    }
    const num = Number(value);
    if (!isNaN(num) && isFinite(num)) {
        return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }
    if (num === Infinity) return "∞";
    if (num === -Infinity) return "-∞";
    if (typeof value === 'string') return value;
    return fallback;
}

/** Safely formats currency, handling backend specials */
function formatCurrency(value, decimals = 2, fallback = "N/A", prefix = "₹") {
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
         if (["∞", "INFINITY", "-∞", "-INFINITY", "N/A", "UNDEFINED", "LOSS", "0 / 0", "∞ / ∞", "LOSS / ∞"].includes(upperVal)) {
             return value; // Don't prefix currency symbol to these specific strings
         }
    }
    const formattedNumberResult = formatNumber(value, decimals, null);
    if (formattedNumberResult !== null && !["∞", "-∞"].includes(formattedNumberResult)) {
        return `${prefix}${formattedNumberResult}`;
    }
    return formattedNumberResult === null ? fallback : formattedNumberResult;
}

/** Helper to display formatted metric/value in a UI element */
function displayMetric(value, targetElementSelector, prefix = '', suffix = '', decimals = 2, isCurrency = false, fallback = "N/A") {
     const element = document.querySelector(targetElementSelector);
     if (!element) { logger.warn(`displayMetric: Element not found: "${targetElementSelector}"`); return; }
     const formatFunc = isCurrency ? formatCurrency : formatNumber;
     const formattedValue = formatFunc(value, decimals, fallback, isCurrency ? "₹" : "");
     element.textContent = `${prefix}${formattedValue}${suffix}`;
}

/** Sets the loading/error/content/hidden state for an element using classes */
function setElementState(selectorOrElement, state, message = 'Loading...') {
    const element = (typeof selectorOrElement === 'string') ? document.querySelector(selectorOrElement) : selectorOrElement;
    if (!element) { logger.warn(`setElementState: Element not found: "${selectorOrElement}"`); return; }

    const isSelect = element.tagName === 'SELECT';
    const isButton = element.tagName === 'BUTTON';
    const isTbody = element.tagName === 'TBODY';
    const isTable = element.tagName === 'TABLE';
    const isContainer = element.tagName === 'DIV' || element.tagName === 'SECTION' || element.classList.contains('chart-container') || element.tagName === 'UL' || element.tagName === 'DETAILS';
    const isSpan = element.tagName === 'SPAN';
    const isGlobalError = element.id === SELECTORS.globalErrorDisplay.substring(1);

    element.classList.remove('loading', 'error', 'loaded', 'hidden');
    if (isSelect || isButton) element.disabled = false;
    element.style.display = '';
    if (isGlobalError) element.style.display = 'none';

    let defaultColspan = 7;
    if (element.closest(SELECTORS.greeksTable)) defaultColspan = 9;
    if (element.closest('.charges-table')) defaultColspan = 12;

    // Clear content appropriately, avoiding metrics list spans
    if (state !== 'error' && state !== 'loading') {
        if (isTbody) element.innerHTML = '';
        else if (isContainer && !element.closest(SELECTORS.metricsList) && !element.matches(SELECTORS.metricsList)) {
           // element.innerHTML = ''; // Be careful clearing containers, might remove structure
        }
    }

    switch (state) {
        case 'loading':
            element.classList.add('loading');
            if (isSelect) { element.innerHTML = `<option>${message}</option>`; element.disabled = true; }
            else if (isTbody) { element.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`; }
            else if (isTable) {
                const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`;
                const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
            }
            else if (isContainer) { element.innerHTML = `<div class="loading-text" style="padding: 20px; text-align: center;">${message}</div>`; }
            else if (isSpan) { element.textContent = '...'; }
            else if (!isButton && !isGlobalError) { element.textContent = message; }
            if (isButton) element.disabled = true;
            if (isGlobalError) { element.textContent = message; element.style.display = 'block'; }
            break;
        case 'error':
            element.classList.add('error');
            const displayMessage = `${message}`;
            if (isSelect) { element.innerHTML = `<option>${displayMessage}</option>`; element.disabled = true; }
            else if (isTbody) { element.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`; }
            else if (isTable) {
                 const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`;
                 const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
            }
            else if (isContainer) { element.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`; }
            else if (isSpan) { element.textContent = 'Error'; element.classList.add('error-message'); }
            else { element.textContent = displayMessage; element.classList.add('error-message'); }
            if (isGlobalError) { element.style.display = 'block'; element.textContent = displayMessage; }
            break;
        case 'content':
            element.classList.add('loaded');
            if (isGlobalError) element.style.display = 'none';
            break;
        case 'hidden':
            element.classList.add('hidden');
            element.style.display = 'none';
            break;
    }
}

/** Populates a dropdown select element. */
function populateDropdown(selector, items, placeholder = "-- Select --", defaultSelection = null) {
    const selectElement = document.querySelector(selector);
    if (!selectElement) return;
    const currentValue = selectElement.value;
    selectElement.innerHTML = "";

    if (!items || items.length === 0) {
        selectElement.innerHTML = `<option value="">-- No options available --</option>`;
        selectElement.disabled = true;
        return;
    }

    if (placeholder) {
        const placeholderOption = document.createElement("option");
        placeholderOption.value = ""; placeholderOption.textContent = placeholder;
        placeholderOption.disabled = true; selectElement.appendChild(placeholderOption);
    }

    items.forEach(item => {
        const option = document.createElement("option");
        option.value = item; option.textContent = item;
        selectElement.appendChild(option);
    });

    let valueSet = false;
    if (items.includes(currentValue)) { selectElement.value = currentValue; valueSet = true; }
    else if (defaultSelection !== null && items.includes(String(defaultSelection))) { selectElement.value = String(defaultSelection); valueSet = true; }
    if (!valueSet && placeholder) { selectElement.value = ""; }
    selectElement.disabled = false;
}

/** Fetches data from the API with enhanced error handling. */
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    options.headers = { ...defaultHeaders, ...options.headers };
    const method = options.method || 'GET';
    const requestBody = options.body ? JSON.parse(options.body) : '';
    logger.debug(`fetchAPI Request: ${method} ${url}`, requestBody || '(No Body)');

    try {
        const response = await fetch(url, options);
        let responseData = null;
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
             try { responseData = await response.json(); }
             catch (jsonError) {
                  const bodyText = await response.text().catch(() => '[Could not read body]');
                  logger.error(`API Error (${method} ${url} - ${response.status}): Failed to parse JSON. Body: ${bodyText}`, jsonError);
                  throw new Error(`Invalid JSON response (Status: ${response.status})`);
             }
        } else if (response.status !== 204) {
             const textResponse = await response.text().catch(() => '[Could not read body]');
             logger.warn(`Non-JSON response from ${method} ${url} (Status: ${response.status}). Body: ${textResponse.substring(0,100)}...`);
             if (!response.ok) throw new Error(textResponse || `HTTP error ${response.status}`);
             responseData = null; // Assume null for non-JSON success
        }

        logger.debug(`fetchAPI Response Status: ${response.status} for ${method} ${url}`);

        if (!response.ok) {
            const errorMessage = responseData?.detail || responseData?.message || responseData?.error || response.statusText || `HTTP error ${response.status}`;
            logger.error(`API Error (${method} ${url} - ${response.status}): ${errorMessage}`, responseData);
            throw new Error(errorMessage);
        }

        setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear global error on success
        logger.debug(`fetchAPI Response Data:`, responseData);
        return responseData;

    } catch (error) {
        logger.error(`Fetch/Network Error or API Error (${method} ${url}):`, error);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Network/API Error: ${error.message || 'Could not connect or invalid response'}`);
        throw error;
    }
}

/** Applies a temporary highlight effect to an element */
function highlightElement(element) {
    if (!element) return;
    element.classList.remove('value-changed');
    void element.offsetWidth; // Trigger reflow
    element.classList.add('value-changed');
    setTimeout(() => { element.classList.remove('value-changed'); }, HIGHLIGHT_DURATION_MS);
}

/** Helper to calculate days to expiry from YYYY-MM-DD string */
function calculateDaysToExpiry(expiryDateStr) {
    try {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(expiryDateStr)) throw new Error("Invalid date format.");
        const expiryDate = new Date(expiryDateStr + 'T00:00:00Z');
        const today = new Date();
        const todayUTC = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));
        if (isNaN(expiryDate.getTime()) || isNaN(todayUTC.getTime())) throw new Error("Could not parse dates.");
        const diffTime = expiryDate.getTime() - todayUTC.getTime() + 1000;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        return Math.max(0, diffDays);
    } catch (e) {
        logger.error("Error calculating DTE for", expiryDateStr, e); return null;
    }
}

/** Helper to round numbers safely */
function roundToPrecision(num, precision) {
    if (typeof num !== 'number' || !isFinite(num)) return null;
    const factor = Math.pow(10, precision);
    return Math.round(num * factor) / factor;
}

/** Finds the nearest strike key (string) to the spot price */
function findATMStrikeAsStringKey(strikeStringKeys = [], spotPrice) {
    if (!Array.isArray(strikeStringKeys) || strikeStringKeys.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         logger.warn("Cannot find ATM strike key: Invalid input.", { numKeys: strikeStringKeys?.length, spotPrice }); return null;
    }
    let closestKey = null; let minDiff = Infinity;
    for (const key of strikeStringKeys) {
        const numericStrike = Number(key);
        if (!isNaN(numericStrike)) {
            const diff = Math.abs(numericStrike - spotPrice);
            if (diff < minDiff) { minDiff = diff; closestKey = key; }
        } else { logger.warn(`Skipping non-numeric strike key '${key}' during ATM calc.`); }
    }
    logger.debug(`Calculated ATM strike key: ${closestKey} (Min diff: ${minDiff.toFixed(4)}) for spot: ${spotPrice.toFixed(4)}`);
    return closestKey;
}

// ===============================================================
// RENDERING FUNCTIONS (Define before use)
// ===============================================================

/** Renders the news items */
function renderNews(containerElement, newsData) {
    containerElement.innerHTML = "";
    if (!newsData || newsData.length === 0) { containerElement.innerHTML = '<p class="placeholder-text">No recent news found.</p>'; return; }
    const firstItemHeadline = newsData[0]?.headline?.toLowerCase() || "";
    if (newsData.length === 1 && (firstItemHeadline.includes("error") || firstItemHeadline.includes("no recent news") || firstItemHeadline.includes("timeout") || firstItemHeadline.includes("no news data"))) {
        const messageClass = firstItemHeadline.includes("error") || firstItemHeadline.includes("timeout") ? "error-message" : "placeholder-text";
        containerElement.innerHTML = `<p class="${messageClass}">${newsData[0].headline}</p>`; return;
    }
    const ul = document.createElement("ul"); ul.className = "news-list";
    newsData.forEach(item => {
        const li = document.createElement("li"); li.className = "news-item";
        const headline = document.createElement("div"); headline.className = "news-headline";
        const link = document.createElement("a"); link.href = item.link || "#"; link.textContent = item.headline || "No Title";
        link.target = "_blank"; link.rel = "noopener noreferrer"; headline.appendChild(link);
        const summary = document.createElement("p"); summary.className = "news-summary";
        summary.textContent = item.summary || "No summary available.";
        li.appendChild(headline); li.appendChild(summary); ul.appendChild(li);
    });
    containerElement.appendChild(ul);
}

/** Renders the Plotly chart */
async function renderPayoffChart(containerElement, figureJsonString) {
    logger.debug("Attempting to render Plotly chart...");
    if (!containerElement) { logger.error("renderPayoffChart: Target container element not found."); return; }
    if (typeof Plotly === 'undefined') { logger.error("renderPayoffChart: Plotly.js not loaded."); setElementState(containerElement, 'error', 'Charting library failed.'); return; }
    if (!figureJsonString || typeof figureJsonString !== 'string') { logger.error("renderPayoffChart: Invalid figure JSON."); setElementState(containerElement, 'error', 'Invalid chart data.'); return; }
    try {
        const figure = JSON.parse(figureJsonString);
        figure.layout = figure.layout || {};
        figure.layout.height = 450; figure.layout.autosize = true; figure.layout.margin = { l: 60, r: 30, t: 30, b: 50 };
        figure.layout.template = 'plotly_white'; figure.layout.showlegend = false; figure.layout.hovermode = 'x unified';
        figure.layout.font = { family: 'Arial, sans-serif', size: 12 };
        figure.layout.yaxis = figure.layout.yaxis || {}; figure.layout.yaxis.title = { text: 'Profit / Loss (₹)', standoff: 10 };
        figure.layout.yaxis.automargin = true; figure.layout.yaxis.gridcolor = 'rgba(220, 220, 220, 0.7)';
        figure.layout.yaxis.zeroline = true; figure.layout.yaxis.zerolinecolor = 'rgba(0, 0, 0, 0.5)'; figure.layout.yaxis.zerolinewidth = 1;
        figure.layout.yaxis.tickprefix = "₹"; figure.layout.yaxis.tickformat = ',.0f';
        figure.layout.xaxis = figure.layout.xaxis || {}; figure.layout.xaxis.title = { text: 'Underlying Spot Price', standoff: 10 };
        figure.layout.xaxis.automargin = true; figure.layout.xaxis.gridcolor = 'rgba(220, 220, 220, 0.7)';
        figure.layout.xaxis.zeroline = false; figure.layout.xaxis.tickformat = ',.0f';
        const plotConfig = { responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d', 'toImage'] };
        containerElement.style.display = ''; containerElement.innerHTML = '';
        await Plotly.react(containerElement.id, figure.data, figure.layout, plotConfig);
        setElementState(containerElement, 'content'); logger.info("Successfully rendered Plotly chart.");
    } catch (renderError) {
        logger.error("Error during Plotly chart processing:", renderError);
        setElementState(containerElement, 'error', `Chart Display Error: ${renderError.message}`);
    }
}

/** Renders the tax table */
function renderTaxTable(containerElement, taxData) {
    const logger = window.logger || console;
    if (!taxData || !taxData.charges_summary || !taxData.breakdown_per_leg || !Array.isArray(taxData.breakdown_per_leg)) {
        containerElement.innerHTML = '<p class="error-message">Charge calculation data unavailable.</p>';
        logger.warn("renderTaxTable called with invalid taxData:", taxData);
        setElementState(containerElement, 'content'); return;
    }
    containerElement.innerHTML = "";
    const details = document.createElement('details'); details.className = "results-details tax-details"; details.open = false;
    const summary = document.createElement('summary');
    summary.innerHTML = `<strong>Estimated Charges Breakdown (Total: ${formatCurrency(taxData.total_estimated_cost, 2)})</strong>`;
    details.appendChild(summary);
    const tableWrapper = document.createElement('div'); tableWrapper.className = 'table-wrapper thin-scrollbar'; details.appendChild(tableWrapper);
    const table = document.createElement("table"); table.className = "results-table charges-table data-table";
    const charges = taxData.charges_summary || {}; const breakdown = taxData.breakdown_per_leg;
    const tableBody = breakdown.map(t => {
        const actionDisplay = (t.transaction_type || '').toUpperCase() === 'B' ? 'BUY' : (t.transaction_type || '').toUpperCase() === 'S' ? 'SELL' : '?';
        const typeDisplay = (t.option_type || '').toUpperCase();
        return `
        <tr><td>${actionDisplay}</td><td>${typeDisplay}</td><td>${formatNumber(t.strike, 2, '-')}</td><td>${formatNumber(t.lots, 0, '-')}</td>
            <td>${formatNumber(t.premium_per_share, 2, '-')}</td><td>${formatNumber(t.stt ?? 0, 2)}</td><td>${formatNumber(t.stamp_duty ?? 0, 2)}</td>
            <td>${formatNumber(t.sebi_fee ?? 0, 4)}</td><td>${formatNumber(t.txn_charge ?? 0, 4)}</td><td>${formatNumber(t.brokerage ?? 0, 2)}</td>
            <td>${formatNumber(t.gst ?? 0, 2)}</td><td class="note" title="${t.stt_note || ''}">${((t.stt_note || '').substring(0, 15))}${ (t.stt_note || '').length > 15 ? '...' : ''}</td></tr>`;
    }).join('');
    const total_stt = charges.stt ?? 0; const total_stamp = charges.stamp_duty ?? 0; const total_sebi = charges.sebi_fee ?? 0;
    const total_txn = charges.txn_charges ?? 0; const total_brokerage = charges.brokerage ?? 0; const total_gst = charges.gst ?? 0;
    const overall_total = taxData.total_estimated_cost ?? 0;
    table.innerHTML = `
        <thead><tr><th>Act</th><th>Type</th><th>Strike</th><th>Lots</th><th>Premium</th><th>STT</th><th>Stamp</th><th>SEBI</th><th>Txn</th><th>Broker</th><th>GST</th><th title="STT Note">STT Note</th></tr></thead>
        <tbody>${tableBody}</tbody>
        <tfoot><tr class="totals-row"><td colspan="5">Total Estimated Charges</td><td>${formatCurrency(total_stt, 2)}</td><td>${formatCurrency(total_stamp, 2)}</td><td>${formatCurrency(total_sebi, 4)}</td>
            <td>${formatCurrency(total_txn, 4)}</td><td>${formatCurrency(total_brokerage, 2)}</td><td>${formatCurrency(total_gst, 2)}</td><td style="font-weight: bold;">${formatCurrency(overall_total, 2)}</td></tr></tfoot>`;
    tableWrapper.appendChild(table); containerElement.appendChild(details);
}

/** Renders the Greeks table and calculates/returns portfolio totals */
function renderGreeksTable(tableElement, greeksList) {
    const logger = window.logger || console; tableElement.innerHTML = '';
    if (!tableElement || !(tableElement instanceof HTMLTableElement)) { logger.error("renderGreeksTable: Invalid tableElement."); return null; }
    const caption = tableElement.createCaption(); caption.className = "table-caption";
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    let hasCalculatedGreeks = false; let skippedLegsCount = 0; let processedLegsCount = 0;
    if (!Array.isArray(greeksList)) { caption.textContent = "Error: Invalid Greeks data."; setElementState(tableElement, 'error'); return null; }
    const totalLegsInput = greeksList.length;
    if (totalLegsInput === 0) {
        caption.textContent = "Portfolio Option Greeks (No legs)"; const tbody = tableElement.createTBody();
        tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">No option legs.</td></tr>`; setElementState(tableElement, 'content'); return totals;
    }
    const thead = tableElement.createTHead();
    thead.innerHTML = `<tr><th>Action</th><th>Lots</th><th>Type</th><th>Strike</th><th title="Delta/Share">Δ Delta</th><th title="Gamma/Share">Γ Gamma</th><th title="Theta/Share(Day)">Θ Theta</th><th title="Vega/Share(1% IV)">Vega</th><th title="Rho/Share(1% Rate)">Ρ Rho</th></tr>`;
    const tbody = tableElement.createTBody();
    greeksList.forEach((g, index) => {
        const row = tbody.insertRow(); const inputData = g?.input_data; const gv_per_share = g?.calculated_greeks_per_share; const gv_per_lot = g?.calculated_greeks_per_lot;
        if (!inputData || !gv_per_share) { logger.warn(`renderGreeksTable: Malformed data leg ${index + 1}.`); skippedLegsCount++; row.innerHTML = `<td colspan="9" class="skipped-leg">Leg ${index + 1}: Invalid data</td>`; return; }
        const actionDisplay = (inputData.tr_type === 'b') ? 'BUY' : (inputData.tr_type === 's' ? 'SELL' : '?'); const typeDisplay = (inputData.op_type === 'c') ? 'CE' : (inputData.op_type === 'p' ? 'PE' : '?');
        const lots = parseInt(inputData.lots || '0', 10); const lotSize = parseInt(inputData.lot_size || '0', 10); const strike = inputData.strike; const lotsDisplay = (lots !== 0) ? `${lots}` : 'N/A';
        row.insertCell().textContent = actionDisplay; row.insertCell().textContent = lotsDisplay; row.insertCell().textContent = typeDisplay; row.insertCell().textContent = formatNumber(strike, 2);
        row.insertCell().textContent = formatNumber(gv_per_share.delta, 4, '-'); row.insertCell().textContent = formatNumber(gv_per_share.gamma, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.theta, 4, '-'); row.insertCell().textContent = formatNumber(gv_per_share.vega, 4, '-'); row.insertCell().textContent = formatNumber(gv_per_share.rho, 4, '-');
        let legDelta = 0, legGamma = 0, legTheta = 0, legVega = 0, legRho = 0; let isValidForTotal = false;
        if (gv_per_lot && lots !== 0) {
            if (isFinite(gv_per_lot.delta)) { legDelta = gv_per_lot.delta * lots; isValidForTotal = true; } if (isFinite(gv_per_lot.gamma)) { legGamma = gv_per_lot.gamma * lots; } if (isFinite(gv_per_lot.theta)) { legTheta = gv_per_lot.theta * lots; } if (isFinite(gv_per_lot.vega)) { legVega = gv_per_lot.vega * lots; } if (isFinite(gv_per_lot.rho)) { legRho = gv_per_lot.rho * lots; }
        } else if (gv_per_share && lots !== 0 && lotSize > 0) {
            const quantity = lots * lotSize;
            if (isFinite(gv_per_share.delta)) { legDelta = gv_per_share.delta * quantity; isValidForTotal = true; } if (isFinite(gv_per_share.gamma)) { legGamma = gv_per_share.gamma * quantity; } if (isFinite(gv_per_share.theta)) { legTheta = gv_per_share.theta * quantity; } if (isFinite(gv_per_share.vega)) { legVega = gv_per_share.vega * quantity; } if (isFinite(gv_per_share.rho)) { legRho = gv_per_share.rho * quantity; }
        }
        if (isValidForTotal) {
            totals.delta += legDelta; totals.gamma += legGamma; totals.theta += legTheta; totals.vega += legVega; totals.rho += legRho;
            hasCalculatedGreeks = true; processedLegsCount++; row.classList.add('greeks-calculated');
        } else { logger.warn(`renderGreeksTable: Skipping leg ${index + 1} from total calculation.`); skippedLegsCount++; row.classList.add('greeks-skipped'); row.style.opacity = '0.6'; row.style.fontStyle = 'italic'; }
    });
    caption.textContent = `Portfolio Option Greeks (${processedLegsCount} Processed, ${skippedLegsCount} Skipped)`;
    const tfoot = tableElement.createTFoot(); const footerRow = tfoot.insertRow(); footerRow.className = 'totals-row';
    if (hasCalculatedGreeks) {
        const headerCell = footerRow.insertCell(); headerCell.colSpan = 4; headerCell.textContent = 'Total Portfolio Exposure'; headerCell.style.textAlign = 'right'; headerCell.style.fontWeight = 'bold';
        footerRow.insertCell().textContent = formatNumber(totals.delta, 4); footerRow.insertCell().textContent = formatNumber(totals.gamma, 4); footerRow.insertCell().textContent = formatNumber(totals.theta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.vega, 4); footerRow.insertCell().textContent = formatNumber(totals.rho, 4); setElementState(tableElement, 'content');
    } else if (totalLegsInput > 0) { const cell = footerRow.insertCell(); cell.colSpan = 9; cell.textContent = 'Could not calculate totals.'; cell.style.textAlign = 'center'; cell.style.fontStyle = 'italic'; setElementState(tableElement, 'content'); }
    const finalTotals = { delta: roundToPrecision(totals.delta, 4), gamma: roundToPrecision(totals.gamma, 4), theta: roundToPrecision(totals.theta, 4), vega: roundToPrecision(totals.vega, 4), rho: roundToPrecision(totals.rho, 4) };
    logger.info(`renderGreeksTable: Rendered ${processedLegsCount}/${totalLegsInput}. Totals: ${JSON.stringify(finalTotals)}`);
    return finalTotals;
}

/** Renders the cost breakdown list */
function renderCostBreakdown(listElement, costBreakdownData) {
    const logger = window.logger || console;
    const localFormatCurrency = window.formatCurrency || ((val) => `₹${Number(val).toFixed(2)}`);
    if (!listElement) { logger.error("renderCostBreakdown: Target list element is null."); return; }
    if (!Array.isArray(costBreakdownData)) { logger.error("renderCostBreakdown: costBreakdownData is not an array."); listElement.innerHTML = '<li>Error displaying breakdown.</li>'; return; }
    listElement.innerHTML = ""; // Clear previous
    if (costBreakdownData.length === 0) { listElement.innerHTML = '<li>No premium breakdown details.</li>'; return; }
    costBreakdownData.forEach((item, index) => {
        try {
            const li = document.createElement("li");
            const action = item.action || 'N/A'; const lots = item.lots ?? '?'; const quantity = item.quantity ?? '?';
            const type = item.type || 'N/A'; const strike = item.strike ?? '?'; const premiumPerShare = item.premium_per_share !== undefined ? localFormatCurrency(item.premium_per_share) : 'N/A';
            const totalPremium = item.total_premium !== undefined ? Math.abs(item.total_premium) : null; const effect = item.effect || 'N/A';
            let premiumEffectText = 'N/A'; if (totalPremium !== null) { premiumEffectText = effect === 'Paid' ? `Paid ${localFormatCurrency(totalPremium)}` : `Received ${localFormatCurrency(totalPremium)}`; }
            li.textContent = `${action} ${lots} Lot(s) [${quantity} Qty] ${type} ${strike} @ ${premiumPerShare} (${premiumEffectText} Total)`;
            listElement.appendChild(li);
        } catch (e) { logger.error(`Error rendering breakdown item ${index}:`, e, item); const errorLi = document.createElement("li"); errorLi.textContent = `Error leg ${index + 1}.`; errorLi.style.color = 'red'; listElement.appendChild(errorLi); }
    });
}


// ===============================================================
// CORE LOGIC & EVENT HANDLERS (Define before Initialization uses them)
// ===============================================================

/** Resets ONLY the calculation output areas */
function resetCalculationOutputsUI() {
    logger.debug("Resetting calculation output UI elements...");
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer); if (chartContainer) { try { if (typeof Plotly !== 'undefined') Plotly.purge(chartContainer.id); } catch (e) { /* ignore */ } chartContainer.innerHTML = '<div class="placeholder-text">Add legs and click "Update Strategy"</div>'; setElementState(chartContainer, 'content'); }
    const taxContainer = document.querySelector(SELECTORS.taxInfoContainer); if (taxContainer) { taxContainer.innerHTML = '<p class="placeholder-text">Update strategy to calculate charges.</p>'; setElementState(taxContainer, 'content'); }
    const greeksTable = document.querySelector(SELECTORS.greeksTable); const greeksSection = document.querySelector(SELECTORS.greeksTableContainer);
    if (greeksTable) { const caption = greeksTable.querySelector('caption'); if (caption) caption.textContent = 'Portfolio Option Greeks'; const tbody = greeksTable.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">Update strategy to calculate Greeks.</td></tr>`; const tfoot = greeksTable.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = ""; setElementState(greeksTable, 'content'); }
    if (greeksSection) { setElementState(greeksSection, 'content'); }
    const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
    if (greeksAnalysisSection) { setElementState(greeksAnalysisSection, 'hidden'); } if (greeksAnalysisContainer) { greeksAnalysisContainer.innerHTML = ''; setElementState(greeksAnalysisContainer, 'content'); }
    displayMetric('N/A', SELECTORS.maxProfitDisplay); displayMetric('N/A', SELECTORS.maxLossDisplay); displayMetric('N/A', SELECTORS.breakevenDisplay); displayMetric('N/A', SELECTORS.rewardToRiskDisplay); displayMetric('N/A', SELECTORS.netPremiumDisplay, '', '', 2, true); setElementState(SELECTORS.metricsList, 'content');
    const breakdownList = document.querySelector(SELECTORS.costBreakdownList); const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
    if (breakdownList) { breakdownList.innerHTML = ""; setElementState(breakdownList, 'content'); } if (breakdownContainer) { breakdownContainer.open = false; setElementState(breakdownContainer, 'hidden'); }
    const warningContainer = document.querySelector(SELECTORS.warningContainer); if (warningContainer) { warningContainer.textContent = ''; setElementState(warningContainer, 'hidden'); }
    logger.debug("Calculation output UI reset complete.");
}

/** Updates the strategy table in the UI from the global `strategyPositions` array */
/** Updates the strategy table in the UI from the global `strategyPositions` array */
function updateStrategyTable() {
    const logger = window.logger || console;
    logger.debug("--- updateStrategyTable START ---"); // Log entry
    const tableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (!tableBody) {
        logger.error("updateStrategyTable failed: Strategy table body not found.");
        return;
    }

    // Log the array content *before* rendering
    logger.debug("Rendering strategy table from strategyPositions:", JSON.parse(JSON.stringify(strategyPositions)));

    tableBody.innerHTML = ""; // Clear previous rows

    if (strategyPositions.length === 0) {
        logger.debug("Strategy positions array is empty. Showing placeholder.");
        tableBody.innerHTML = '<tr><td colspan="7" class="placeholder-text">No positions added. Click prices in the chain.</td></tr>';
        return;
    }

    strategyPositions.forEach((pos, index) => {
        logger.debug(`Rendering row for index ${index}:`, pos); // Log each position being rendered
        const isLong = pos.lots >= 0; pos.tr_type = isLong ? 'b' : 's';
        const positionType = isLong ? "BUY" : "SELL"; const positionClass = isLong ? "long-position" : "short-position"; const buttonClass = isLong ? "button-buy" : "button-sell";
        const row = document.createElement("tr"); row.className = positionClass; row.dataset.index = index;
        // Check strike_price before formatting
        const formattedStrike = (pos.strike_price !== undefined && pos.strike_price !== null)
                                ? formatNumber(pos.strike_price, pos.strike_price % 1 === 0 ? 0 : 2, 'Err')
                                : 'Err';
        row.innerHTML = `<td>${pos.option_type || 'N/A'}</td><td>${formattedStrike}</td><td>${pos.expiry_date || 'N/A'}</td><td><input type="number" value="${pos.lots}" data-index="${index}" min="-100" max="100" step="1" class="lots-input number-input-small" aria-label="Lots for position ${index+1}"></td><td><button class="toggle-buy-sell ${buttonClass}" data-index="${index}" title="Toggle Buy/Sell">${positionType}</button></td><td>${formatCurrency(pos.last_price, 2)}</td><td><button class="remove-btn" data-index="${index}" aria-label="Remove position ${index+1}" title="Remove leg">×</button></td>`;
        try {
            tableBody.appendChild(row);
        } catch(e) {
            logger.error(`Error appending row ${index} to strategy table body:`, e);
        }
    });
    logger.debug("--- updateStrategyTable END ---");
}

/** Resets the entire results area AND the strategy input table */
function resetResultsUI() {
    logger.info("Resetting results UI & clearing strategy input...");
    strategyPositions = []; // Clear the data array FIRST
    if (typeof updateStrategyTable === 'function') { updateStrategyTable(); } // THEN update the table UI
    else { logger.error("updateStrategyTable function not defined during resetResultsUI call!"); }
    if (typeof resetCalculationOutputsUI === 'function') { resetCalculationOutputsUI(); } // Reset chart, metrics etc.
    else { logger.error("resetCalculationOutputsUI function not defined during resetResultsUI call!"); }
    const messageContainer = document.querySelector(SELECTORS.statusMessageContainer); if (messageContainer) { messageContainer.textContent = ''; setElementState(messageContainer, 'hidden'); }
    logger.info("Full UI reset complete.");
}

/** Gathers VALID strategy leg data from the global `strategyPositions` array */
function gatherStrategyLegsFromTable() { // Keep name for compatibility
    logger.debug("--- [gatherStrategyLegs] START ---");
    if (!Array.isArray(strategyPositions)) { logger.error("[gatherStrategyLegs] strategyPositions is not an array."); return []; }
    if (strategyPositions.length === 0) { logger.warn("[gatherStrategyLegs] strategyPositions is empty."); return []; }
    logger.debug("[gatherStrategyLegs] Source:", JSON.parse(JSON.stringify(strategyPositions)));
    const validLegs = []; let invalidLegCount = 0;
    strategyPositions.forEach((pos, index) => {
        let legIsValid = true; let validationError = null;
        if (!pos || typeof pos !== 'object') { validationError = "Not object."; legIsValid = false; }
        else {
            if (!pos.option_type || (pos.option_type !== 'CE' && pos.option_type !== 'PE')) { validationError = `Type:${pos.option_type}`; legIsValid = false; }
            else if (pos.strike_price===undefined || isNaN(parseFloat(pos.strike_price)) || parseFloat(pos.strike_price) <= 0) { validationError = `Strike:${pos.strike_price}`; legIsValid = false; }
            else if (!pos.expiry_date || !/^\d{4}-\d{2}-\d{2}$/.test(pos.expiry_date)) { validationError = `Expiry:${pos.expiry_date}`; legIsValid = false; }
            else if (pos.lots===undefined || !Number.isInteger(pos.lots) || pos.lots === 0) { validationError = `Lots:${pos.lots}`; legIsValid = false; }
            else if (pos.last_price===undefined || isNaN(parseFloat(pos.last_price)) || parseFloat(pos.last_price) < 0) { validationError = `Price:${pos.last_price}`; legIsValid = false; }
            else if (pos.days_to_expiry===undefined || !Number.isInteger(pos.days_to_expiry) || pos.days_to_expiry < 0) { validationError = `DTE:${pos.days_to_expiry}`; legIsValid = false; }
            else if (pos.iv !== null && pos.iv !== undefined && (isNaN(parseFloat(pos.iv)) || parseFloat(pos.iv) < 0)) { validationError = `IV:${pos.iv}`; legIsValid = false; }
            else if (pos.lot_size !== null && pos.lot_size !== undefined && (!Number.isInteger(pos.lot_size) || pos.lot_size <= 0)) { validationError = `LotSize:${pos.lot_size}`; legIsValid = false; }
        }
        if (legIsValid) {
            const lotsInt = pos.lots; // Already checked it's integer
            const formattedLeg = {
                 op_type: pos.option_type === 'CE' ? 'c' : 'p', strike: String(pos.strike_price), tr_type: lotsInt >= 0 ? 'b' : 's',
                 op_pr: String(pos.last_price), lot: String(Math.abs(lotsInt)),
                 lot_size: (pos.lot_size && Number.isInteger(pos.lot_size) && pos.lot_size > 0) ? String(pos.lot_size) : null,
                 iv: (pos.iv !== null && pos.iv !== undefined && !isNaN(parseFloat(pos.iv))) ? parseFloat(pos.iv) : null,
                 days_to_expiry: pos.days_to_expiry, expiry_date: pos.expiry_date,
            };
            validLegs.push(formattedLeg);
        } else { logger.error(`[gather] Skipping invalid leg ${index}. ${validationError}. Data:`, JSON.parse(JSON.stringify(pos))); invalidLegCount++; }
    });
    if (invalidLegCount > 0 && validLegs.length === 0) { alert(`Error: ${invalidLegCount} invalid leg(s) found, NO valid legs remaining. Cannot calculate. Check console.`); }
    else if (invalidLegCount > 0) { alert(`Warning: ${invalidLegCount} invalid leg(s) ignored. Calculation based on ${validLegs.length} valid leg(s). Check console.`); }
    logger.debug(`[gather] Returning ${validLegs.length} valid formatted legs (ignored ${invalidLegCount}).`);
    logger.debug("--- [gatherStrategyLegs] END ---");
    return validLegs;
}

/** Adds a position to the global `strategyPositions` array */
/** Adds a position to the global `strategyPositions` array */
function addPosition(strike, type, price, iv) {
    const logger = window.logger || console;
    logger.debug("--- addPosition START ---", { strike, type, price, iv }); // Log entry

    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) {
        logger.error("addPosition failed: No expiry selected.");
        alert("Please select an expiry date first.");
        return;
    }
    logger.debug("Expiry found:", expiry);

    const lastPrice = (typeof price === 'number' && !isNaN(price) && price >= 0) ? price : 0;
    const impliedVol = (typeof iv === 'number' && !isNaN(iv) && iv > 0) ? iv : null;
    const dte = calculateDaysToExpiry(expiry);

    if (impliedVol === null) { logger.warn(`Adding position ${type} ${strike} @ ${expiry} without valid IV.`); }
    if (dte === null) { logger.error(`Could not calculate DTE for ${expiry}.`); alert(`Error: Invalid expiry date.`); return; }

    const lotSize = null; // Placeholder

    const newPosition = {
        strike_price: strike, expiry_date: expiry, option_type: type,
        lots: 1, tr_type: 'b', last_price: lastPrice,
        iv: impliedVol, days_to_expiry: dte, lot_size: lotSize
    };
    logger.debug("Constructed newPosition object:", JSON.parse(JSON.stringify(newPosition)));

    // --- CRITICAL DEBUG AREA ---
    logger.debug("strategyPositions array BEFORE push:", JSON.parse(JSON.stringify(strategyPositions)));
    try {
        strategyPositions.push(newPosition); // Add to the global array
        logger.debug("strategyPositions array AFTER push:", JSON.parse(JSON.stringify(strategyPositions))); // Check if it grew
        logger.info(`Successfully pushed position. Total legs: ${strategyPositions.length}`);
    } catch (e) {
        logger.error("!!! ERROR DURING strategyPositions.push !!!", e);
        alert("A critical error occurred while trying to store the position.");
        return; // Stop if push fails
    }
    // --- END CRITICAL DEBUG AREA ---


    // Ensure updateStrategyTable exists before calling
    if(typeof updateStrategyTable === 'function') {
        logger.debug("Calling updateStrategyTable...");
        updateStrategyTable(); // Update UI table
    } else {
        logger.error("updateStrategyTable function is not defined! Cannot update strategy UI.");
        alert("Internal Error: Cannot display updated strategy table.");
    }
    logger.debug("--- addPosition END ---");
}

/** Updates the number of lots for a position */
function updateLots(index, rawValue) {
    if (index < 0 || index >= strategyPositions.length) return;
    const trimmedValue = String(rawValue).trim();
    const inputElement = document.querySelector(`${SELECTORS.strategyTableBody} input.lots-input[data-index="${index}"]`);
    if (!/^-?\d+$/.test(trimmedValue) || trimmedValue === '-') { logger.warn(`Invalid lots input idx ${index}: "${rawValue}"`); if (inputElement) inputElement.value = strategyPositions[index].lots; return; }
    const newLots = parseInt(trimmedValue, 10);
    const minLots = -100; const maxLots = 100;
    if (newLots < minLots || newLots > maxLots) { logger.warn(`Lots idx ${index} (${newLots}) out of range.`); alert(`Lots must be between ${minLots}-${maxLots}.`); if (inputElement) inputElement.value = strategyPositions[index].lots; return; }
    if (newLots === 0) { logger.info(`Lots 0 for idx ${index}, removing.`); removePosition(index); }
    else {
        const previousLots = strategyPositions[index].lots; strategyPositions[index].lots = newLots; strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's';
        const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`); const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);
        if (row && toggleButton) { const isNowLong = newLots >= 0; const positionType = isNowLong ? "BUY" : "SELL"; const buttonClass = isNowLong ? "button-buy" : "button-sell"; row.className = isNowLong ? "long-position" : "short-position"; toggleButton.textContent = positionType; toggleButton.className = `toggle-buy-sell ${buttonClass}`; }
        else { logger.warn(`UI elements not found idx ${index}, refreshing table.`); updateStrategyTable(); }
        logger.info(`Updated lots idx ${index}: ${previousLots} -> ${newLots}`);
    }
}

/** Toggles a position between Buy and Sell */
function toggleBuySell(index) {
    if (index < 0 || index >= strategyPositions.length) return;
    const previousLots = strategyPositions[index].lots; let newLots = -previousLots;
    if (newLots === 0) { newLots = 1; } // Default to Buy 1 if toggling from 0
    strategyPositions[index].lots = newLots; strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's';
    logger.info(`Toggled Buy/Sell idx ${index}. Prev: ${previousLots}, New: ${newLots}`);
    const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
    const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`); const lotsInput = row?.querySelector(`input.lots-input[data-index="${index}"]`);
    if (row && toggleButton && lotsInput) {
        const isLong = newLots >= 0; const positionType = isLong ? "BUY" : "SELL"; const buttonClass = isLong ? "button-buy" : "button-sell";
        row.className = isLong ? "long-position" : "short-position"; toggleButton.textContent = positionType; toggleButton.className = `toggle-buy-sell ${buttonClass}`; lotsInput.value = newLots;
    } else { logger.warn(`UI elements not found idx ${index} during toggle, refreshing table.`); updateStrategyTable(); }
}

/** Removes a position from the strategy */
function removePosition(index) {
    if (index < 0 || index >= strategyPositions.length) return;
    const removedPos = strategyPositions.splice(index, 1); logger.info("Removed position idx", index, removedPos[0]);
    updateStrategyTable(); // Re-render table with updated indices
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer); const hasChartContent = chartContainer?.querySelector('.plotly') !== null;
    if (strategyPositions.length > 0 && hasChartContent) { logger.info("Remaining positions, updating calculations..."); fetchPayoffChart(); } // Recalculate if needed
    else if (strategyPositions.length === 0) { logger.info("Strategy empty, resetting outputs."); resetCalculationOutputsUI(); }
}

/** Clears all positions and resets calculation outputs */
function clearAllPositions() {
    if (strategyPositions.length === 0) return;
    if (confirm("Clear all strategy legs?")) { logger.info("Clearing all positions..."); strategyPositions = []; updateStrategyTable(); resetCalculationOutputsUI(); logger.info("Strategy cleared."); }
}

/** Helper function to trigger the ATM scroll logic */
function triggerATMScroll(tbodyElement, atmKeyToUse) {
    const logger = window.logger || console;
    setTimeout(() => {
        try {
            const numericATMStrike = parseFloat(atmKeyToUse);
            logger.debug(`Scroll Timeout: Finding ATM row data-strike="${numericATMStrike}". Tbody valid:`, tbodyElement instanceof HTMLElement);
            if (isNaN(numericATMStrike) || !(tbodyElement instanceof HTMLElement)) { logger.warn(`Invalid ATM key or tbody for scroll.`); return; }
            let atmRow = tbodyElement.querySelector(`tr[data-strike="${numericATMStrike}"]`);
            if (!atmRow) { logger.debug(`Numeric match failed, trying key: ${atmKeyToUse}`); atmRow = tbodyElement.querySelector(`tr[data-strike-key="${atmKeyToUse}"]`); }
            if (!atmRow) {
                // Optional: Find closest if exact match fails (use with caution)
                logger.debug(`Exact match failed, checking closest.`); const allRows = tbodyElement.querySelectorAll('tr[data-strike]'); let closestRow = null; let closestDiff = Infinity;
                allRows.forEach(row => { const rowStrike = parseFloat(row.dataset.strike); if (!isNaN(rowStrike)) { const diff = Math.abs(rowStrike - numericATMStrike); if (diff < closestDiff) { closestDiff = diff; closestRow = row; } } });
                if (closestRow && closestDiff < 0.01) { logger.debug(`Using closest match (diff ${closestDiff})`); atmRow = closestRow; }
            }
            if (atmRow) {
                atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" }); logger.debug(`Scrolled to ATM: ${atmRow.dataset.strike}`);
                atmRow.classList.add("highlight-atm"); setTimeout(() => { if (atmRow?.parentNode) atmRow.classList.remove("highlight-atm"); }, 2000);
            } else { logger.warn(`ATM row (${atmKeyToUse} / ${numericATMStrike}) not found for scroll.`); }
        } catch (e) { logger.error("Error inside scroll timeout:", e); }
    }, 250);
}

/** Fetches and displays the spot price */
async function fetchNiftyPrice(asset, isRefresh = false) {
    if (!asset) return; const priceElement = document.querySelector(SELECTORS.spotPriceDisplay);
    if (!isRefresh) setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot: ...');
    try {
        const data = await fetchAPI(`/get_spot_price?asset=${encodeURIComponent(asset)}`);
        const newSpotPrice = data?.spot_price;
        if (newSpotPrice === null || isNaN(parseFloat(newSpotPrice))) throw new Error("Spot price unavailable/invalid.");
        const validSpotPrice = parseFloat(newSpotPrice);
        const previousValue = currentSpotPrice; currentSpotPrice = validSpotPrice;
        if (priceElement) { priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}`; setElementState(SELECTORS.spotPriceDisplay, 'content'); }
        if (isRefresh && Math.abs(currentSpotPrice - previousValue) > 0.001 && previousValue !== 0) { logger.debug(`Spot change: ${previousValue.toFixed(2)} -> ${currentSpotPrice.toFixed(2)}`); highlightElement(priceElement); }
    } catch (error) {
         logger.error(`Error fetching spot price for ${asset}:`, error.message); currentSpotPrice = 0;
         if (!isRefresh) { setElementState(SELECTORS.spotPriceDisplay, 'error', `Spot: Error`); throw error; }
         else { logger.warn(`Spot Price refresh Error (${asset}):`, error.message); if (priceElement) priceElement.classList.add('error-message'); }
    }
}

/** Fetches and displays the option chain */
async function fetchOptionChain(scrollToATM = false, isRefresh = false) {
    const asset = activeAsset; const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    const optionChainTable = document.getElementById('optionChainTable');
    if (!optionChainTable) { logger.error("#optionChainTable not found."); return; }
    let targetTbody = optionChainTable.querySelector('tbody'); // Find current/target tbody
    if (!asset || !expiry) { if (targetTbody) { targetTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">Select Asset & Expiry</td></tr>`; if (!isRefresh) setElementState(targetTbody, 'content'); } previousOptionChainData = {}; return; }
    if (!isRefresh && targetTbody) { setElementState(targetTbody, 'loading', 'Loading Chain...'); }
    else if (!isRefresh && !targetTbody) { targetTbody = document.createElement('tbody'); optionChainTable.appendChild(targetTbody); setElementState(targetTbody, 'loading', 'Loading Chain...'); logger.warn("Initial tbody created."); }
    try {
        if (currentSpotPrice <= 0 && scrollToATM && !isRefresh) { logger.info("Fetching spot for ATM scroll..."); try { await fetchNiftyPrice(asset); } catch (spotError) { /* Ignore */ } if (currentSpotPrice <= 0) { logger.warn("Spot unavailable, cannot calc ATM."); scrollToATM = false; } }
        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`); const currentChainData = data?.option_chain;
        if (!currentChainData || typeof currentChainData !== 'object' || Object.keys(currentChainData).length === 0) {
            logger.warn(`No chain data for ${asset}/${expiry}.`); if (targetTbody) { targetTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">No chain data found</td></tr>`; if (!isRefresh) setElementState(targetTbody, 'content'); } previousOptionChainData = {}; return;
        }
        const strikeStringKeys = Object.keys(currentChainData).sort((a, b) => Number(a) - Number(b)); const atmStrikeObjectKey = currentSpotPrice > 0 ? findATMStrikeAsStringKey(strikeStringKeys, currentSpotPrice) : null;
        const newTbody = document.createElement('tbody');
        strikeStringKeys.forEach((strikeStringKey) => {
            const optionData = currentChainData[strikeStringKey] || {}; const call = optionData.call || {}; const put = optionData.put || {}; const strikeNumericValue = parseFloat(strikeStringKey); if (isNaN(strikeNumericValue)) return;
            const prevOptionData = previousOptionChainData[strikeStringKey] || {}; const prevCall = prevOptionData.call || {}; const prevPut = prevOptionData.put || {};
            const tr = newTbody.insertRow(); tr.dataset.strike = strikeNumericValue; tr.dataset.strikeKey = strikeStringKey; if (atmStrikeObjectKey === strikeStringKey) tr.classList.add("atm-strike");
            const columns = [
                { class: 'call clickable price', type: 'CE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') }, { class: 'call oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') }, { class: 'call iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'strike', isStrike: true, format: val => formatNumber(val, val % 1 === 0 ? 0 : 2) },
                { class: 'put iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` }, { class: 'put oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') }, { class: 'put clickable price', type: 'PE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') },
            ];
            columns.forEach(col => {
                try {
                    const td = tr.insertCell(); td.className = col.class; let currentValue; let sourceObj = null; let prevDataObject = null;
                    if (col.isStrike) { currentValue = strikeNumericValue; } else { sourceObj = col.class.includes('call') ? call : put; prevDataObject = col.class.includes('call') ? prevCall : prevPut; currentValue = sourceObj?.[col.dataKey]; }
                    td.textContent = col.format(currentValue);
                    if (col.type && sourceObj) { td.dataset.type = col.type; td.dataset.price = String(sourceObj['last_price'] ?? '0'); const ivValue = sourceObj['implied_volatility']; td.dataset.iv = (ivValue != null) ? String(ivValue) : ''; }
                    if (isRefresh && !col.isStrike && prevDataObject) {
                        let previousValue = prevDataObject[col.dataKey]; let changed = false; const currentExists = currentValue != null; const previousExists = previousValue != null;
                        if (currentExists && previousExists) { changed = (typeof currentValue === 'number' && typeof previousValue === 'number') ? Math.abs(currentValue - previousValue) > 0.001 : String(col.format(currentValue)) !== String(col.format(previousValue)); }
                        else if (currentExists !== previousExists) { changed = true; } if (changed) { highlightElement(td); }
                    } else if (isRefresh && !col.isStrike && currentValue != null && !prevDataObject) { highlightElement(td); }
                } catch (cellError) { logger.error(`Error cell ${strikeStringKey}/${col.dataKey}:`, cellError); const errorTd = tr.insertCell(); errorTd.textContent = 'ERR'; errorTd.className = col.class + ' error-message'; }
            });
        });
        const oldTbody = optionChainTable.querySelector('tbody'); if (oldTbody) { optionChainTable.replaceChild(newTbody, oldTbody); } else { logger.warn("Old tbody not found, appending new."); optionChainTable.appendChild(newTbody); }
        if (!isRefresh) { setElementState(newTbody, 'content'); } previousOptionChainData = currentChainData;
        if (scrollToATM && atmStrikeObjectKey !== null && !isRefresh) { triggerATMScroll(newTbody, atmStrikeObjectKey); }
    } catch (error) {
        logger.error(`Error fetchOptionChain ${asset}/${expiry}:`, error); const errorTbody = optionChainTable.querySelector('tbody');
        if (errorTbody) { errorTbody.innerHTML = `<tr><td colspan="7" class="error-message">Chain Error: ${error.message}</td></tr>`; if (!isRefresh) setElementState(errorTbody, 'error'); }
        if (isRefresh) { logger.warn(`Chain refresh failed: ${error.message}`); } previousOptionChainData = {};
    }
}

/** Fetches and populates expiry dates */
async function fetchExpiries(asset) {
    if (!asset) return; setElementState(SELECTORS.expiryDropdown, 'loading');
    try {
        const data = await fetchAPI(`/expiry_dates?asset=${encodeURIComponent(asset)}`); const expiries = data?.expiry_dates || [];
        populateDropdown(SELECTORS.expiryDropdown, expiries, "-- Select Expiry --", expiries[0]); setElementState(SELECTORS.expiryDropdown, 'content');
        const selectedExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
        if (selectedExpiry) { if (expiries.length > 0 && selectedExpiry === expiries[0]) { logger.info(`Default expiry ${selectedExpiry} selected. Fetching chain...`); await fetchOptionChain(true); } }
        else { logger.warn(`No expiry dates found for ${asset}.`); setElementState(SELECTORS.expiryDropdown, 'error', 'No Expiries'); setElementState(SELECTORS.optionChainTableBody, 'content'); document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">No expiries found</td></tr>`; throw new Error("No expiry dates found."); }
    } catch (error) {
        logger.error(`Error fetching expiries for ${asset}: ${error.message}`); populateDropdown(SELECTORS.expiryDropdown, [], "-- Error Loading --"); setElementState(SELECTORS.expiryDropdown, 'error'); setElementState(SELECTORS.optionChainTableBody, 'error', `Failed loading expiries.`); throw error;
    }
}

/** Fetches stock analysis */
async function fetchAnalysis(asset) {
    const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer); if (!analysisContainer) return; if (!asset) { analysisContainer.innerHTML = '<p class="placeholder-text">Select asset.</p>'; setElementState(analysisContainer, 'content'); return; }
    setElementState(analysisContainer, 'loading', 'Fetching analysis...'); logger.debug(`Fetching analysis for ${asset}...`);
    try {
        if (typeof marked === 'undefined') { logger.warn("Waiting for marked.js..."); await new Promise(resolve => setTimeout(resolve, 500)); if (typeof marked === 'undefined') throw new Error("Markdown parser failed."); }
        const data = await fetchAPI("/get_stock_analysis", { method: "POST", body: JSON.stringify({ asset }) }); logger.debug(`Received analysis for ${asset}`);
        const rawAnalysis = data?.analysis || "*Analysis generation failed.*"; const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, ''); analysisContainer.innerHTML = marked.parse(potentiallySanitized);
        setElementState(analysisContainer, 'content'); logger.info(`Rendered analysis for ${asset}`);
    } catch (error) {
        logger.error(`Error fetching/rendering analysis for ${asset}:`, error); let displayMessage = `Analysis Error: ${error.message}`;
        if (error.message?.includes("Essential stock data not found")) displayMessage = `Analysis unavailable: ${error.message}`;
        else if (error.message?.includes("Analysis blocked")) displayMessage = `Analysis blocked due to content restrictions.`;
        else if (error.message?.includes("Analysis generation failed") || error.message?.includes("Analysis feature not configured")) displayMessage = `Analysis Error: ${error.message}`;
        setElementState(analysisContainer, 'error', displayMessage);
    }
}

/** Fetches news */
async function fetchNews(asset) {
    if (!asset) return; const newsContainer = document.querySelector(SELECTORS.newsResultContainer); if (!newsContainer) return;
    setElementState(newsContainer, 'loading', 'Fetching news...');
    try {
        const data = await fetchAPI(`/get_news?asset=${encodeURIComponent(asset)}`); const newsItems = data?.news;
        if (Array.isArray(newsItems)) { renderNews(newsContainer, newsItems); setElementState(newsContainer, 'content'); }
        else { throw new Error("Invalid news data format."); }
    } catch (error) { logger.error(`Error fetching/rendering news for ${asset}:`, error); setElementState(newsContainer, 'error', `News Error: ${error.message}`); }
}

/** Fetches Greeks analysis */
async function fetchAndDisplayGreeksAnalysis(asset, portfolioGreeksData) {
    const container = document.querySelector(SELECTORS.greeksAnalysisResultContainer); const section = document.querySelector(SELECTORS.greeksAnalysisSection);
    if (!container || !section) { logger.error("Greeks analysis container/section not found."); return; }
    if (!asset || !portfolioGreeksData || typeof portfolioGreeksData !== 'object') { logger.warn("Greeks analysis skipped: Invalid input."); setElementState(section, 'hidden'); return; }
    const allZeroOrNull = Object.values(portfolioGreeksData).every(v => v === null || Math.abs(v) < 1e-9);
    if (allZeroOrNull) { logger.info("Greeks analysis skipped: All Greeks zero."); container.innerHTML = '<p class="placeholder-text">No net option exposure.</p>'; setElementState(section, 'content'); setElementState(container, 'content'); return; }
    logger.info(`Fetching Greeks analysis for ${asset}...`); setElementState(section, 'content'); setElementState(container, 'loading', 'Fetching Greeks analysis...');
    try {
        if (typeof marked === 'undefined') { logger.warn("Waiting for marked.js..."); await new Promise(resolve => setTimeout(resolve, 200)); if (typeof marked === 'undefined') throw new Error("Markdown parser failed."); }
        const requestBody = { asset_symbol: asset, portfolio_greeks: portfolioGreeksData };
        const data = await fetchAPI("/get_greeks_analysis", { method: "POST", body: JSON.stringify(requestBody) });
        const rawAnalysis = data?.greeks_analysis || "*Greeks analysis failed.*"; const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
        container.innerHTML = marked.parse(potentiallySanitized); setElementState(container, 'content'); logger.info(`Rendered Greeks analysis for ${asset}`);
    } catch (error) { logger.error(`Error fetching/rendering Greeks analysis for ${asset}:`, error); setElementState(container, 'error', `Greeks Analysis Error: ${error.message}`); setElementState(section, 'content'); }
}

/** Fetches payoff chart and all calculation results */
async function fetchPayoffChart() {
    logger.info("--- [fetchPayoffChart] START ---"); const updateButton = document.querySelector(SELECTORS.updateChartButton);
    const asset = activeAsset; if (!asset) { alert("Select asset first."); logger.error("[fetchPayoff] No active asset."); return; }
    logger.debug("[fetchPayoff] Gathering legs...");
    const currentStrategyLegs = gatherStrategyLegsFromTable();
    if (!currentStrategyLegs || currentStrategyLegs.length === 0) { logger.warn("[fetchPayoff] No valid legs."); resetCalculationOutputsUI(); alert("Add strategy legs first."); return; }
    logger.debug(`[fetchPayoff] Found ${currentStrategyLegs.length} valid legs.`);
    logger.debug("[fetchPayoff] Resetting output UI..."); resetCalculationOutputsUI();
    logger.debug("[fetchPayoff] Setting loading states...");
    setElementState(SELECTORS.payoffChartContainer, 'loading', 'Calculating...'); setElementState(SELECTORS.taxInfoContainer, 'loading');
    setElementState(SELECTORS.greeksTableContainer, 'loading'); setElementState(SELECTORS.greeksTable, 'loading'); setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
    setElementState(SELECTORS.metricsList, 'loading'); displayMetric('...', SELECTORS.maxProfitDisplay); displayMetric('...', SELECTORS.maxLossDisplay);
    displayMetric('...', SELECTORS.breakevenDisplay); displayMetric('...', SELECTORS.rewardToRiskDisplay); displayMetric('...', SELECTORS.netPremiumDisplay);
    setElementState(SELECTORS.costBreakdownContainer, 'hidden'); setElementState(SELECTORS.warningContainer, 'hidden');
    if (updateButton) updateButton.disabled = true;
    const requestData = { asset: asset, strategy: currentStrategyLegs }; logger.debug("[fetchPayoff] Request Data:", JSON.parse(JSON.stringify(requestData)));
    try {
        const data = await fetchAPI('/get_payoff_chart', { method: 'POST', body: JSON.stringify(requestData) }); logger.debug("[fetchPayoff] Response:", JSON.parse(JSON.stringify(data)));
        if (!data || typeof data !== 'object') throw new Error("Invalid response format.");
        if (!data.success) { const errMsg = data.message || "Calculation failed server-side."; logger.error(`[fetchPayoff] Backend failure: ${errMsg}`, data); throw new Error(errMsg); }
        logger.debug("[fetchPayoff] Rendering results...");
        const metricsContainerData = data.metrics; const metrics = metricsContainerData?.metrics;
        if (metrics) {
            displayMetric(metrics.max_profit, SELECTORS.maxProfitDisplay); displayMetric(metrics.max_loss, SELECTORS.maxLossDisplay);
            const bePoints = Array.isArray(metrics.breakeven_points) ? metrics.breakeven_points.join(' / ') : metrics.breakeven_points; displayMetric(bePoints || 'N/A', SELECTORS.breakevenDisplay);
            displayMetric(metrics.reward_to_risk_ratio, SELECTORS.rewardToRiskDisplay); displayMetric(metrics.net_premium, SELECTORS.netPremiumDisplay, '', '', 2, true); setElementState(SELECTORS.metricsList, 'content');
            const warnings = metrics.warnings; const warningElement = document.querySelector(SELECTORS.warningContainer);
            if (warningElement && Array.isArray(warnings) && warnings.length > 0) { warningElement.innerHTML = `<strong>Warnings:</strong><ul>${warnings.map(w => `<li>${w}</li>`).join('')}</ul>`; setElementState(warningElement, 'content'); warningElement.style.display = ''; }
            else if (warningElement) { setElementState(warningElement, 'hidden'); }
        } else { logger.error("[fetchPayoff] Metrics data missing."); setElementState(SELECTORS.metricsList, 'error'); /* Set spans to Error */ }
        const costBreakdownData = metricsContainerData?.cost_breakdown_per_leg; const breakdownList = document.querySelector(SELECTORS.costBreakdownList); const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
        if (breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) { renderCostBreakdown(breakdownList, costBreakdownData); setElementState(breakdownContainer, 'content'); breakdownContainer.style.display = ''; breakdownContainer.open = false; }
        else if (breakdownContainer) { setElementState(breakdownContainer, 'hidden'); }
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer); if (taxContainer) { if (data.charges) { renderTaxTable(taxContainer, data.charges); setElementState(taxContainer, 'content'); } else { taxContainer.innerHTML = "<p class='placeholder-text'>Charge data unavailable.</p>"; setElementState(taxContainer, 'content'); } }
        const chartContainer = document.querySelector(SELECTORS.payoffChartContainer); const chartDataKey = "chart_figure_json"; if (chartContainer) { if (data[chartDataKey]) { renderPayoffChart(chartContainer, data[chartDataKey]); } else { logger.error("[fetchPayoff] Chart JSON missing."); setElementState(chartContainer, 'error', 'Chart unavailable.'); } }
        const greeksTableElement = document.querySelector(SELECTORS.greeksTable); const greeksSectionElement = document.querySelector(SELECTORS.greeksTableContainer);
        if (greeksTableElement && greeksSectionElement) {
            if (data.greeks && Array.isArray(data.greeks)) {
                const calculatedTotals = renderGreeksTable(greeksTableElement, data.greeks); setElementState(greeksSectionElement, 'content');
                if (calculatedTotals) { const hasMeaningfulGreeks = Object.values(calculatedTotals).some(v => v !== null && Math.abs(v) > 1e-9); if (hasMeaningfulGreeks) { fetchAndDisplayGreeksAnalysis(asset, calculatedTotals); } else { logger.info("Greeks analysis skipped: zero exposure."); const analysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); const analysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer); if (analysisSection && analysisContainer) { analysisContainer.innerHTML = '<p class="placeholder-text">No net option exposure.</p>'; setElementState(analysisSection, 'content'); setElementState(analysisContainer, 'content'); } } }
                else { logger.error("[fetchPayoff] Greeks totals calculation failed."); const greeksAS = document.querySelector(SELECTORS.greeksAnalysisSection); if (greeksAS) setElementState(greeksAS, 'hidden'); }
            } else { logger.warn("[fetchPayoff] Greeks data missing."); greeksSectionElement.innerHTML = '<h3 class="section-subheader">Options Greeks</h3><p class="placeholder-text">Greeks data unavailable.</p>'; setElementState(greeksSectionElement, 'content'); const greeksAS = document.querySelector(SELECTORS.greeksAnalysisSection); if (greeksAS) setElementState(greeksAS, 'hidden'); }
        }
        logger.info("[fetchPayoff] Successfully processed results.");
    } catch (error) {
        logger.error(`[fetchPayoff] Error: ${error.message}`, error); resetCalculationOutputsUI(); let errorMsg = `Calculation Error: ${error.message || 'Failed.'}`;
        setElementState(SELECTORS.payoffChartContainer, 'error', errorMsg); setElementState(SELECTORS.taxInfoContainer, 'error', 'Error'); setElementState(SELECTORS.greeksTableContainer, 'error', 'Error'); setElementState(SELECTORS.greeksTable, 'error'); setElementState(SELECTORS.metricsList, 'error');
        displayMetric('Error', SELECTORS.maxProfitDisplay); displayMetric('Error', SELECTORS.maxLossDisplay); displayMetric('Error', SELECTORS.breakevenDisplay); displayMetric('Error', SELECTORS.rewardToRiskDisplay); displayMetric('Error', SELECTORS.netPremiumDisplay);
        setElementState(SELECTORS.greeksAnalysisSection, 'hidden'); setElementState(SELECTORS.costBreakdownContainer, 'hidden'); setElementState(SELECTORS.warningContainer, 'hidden'); setElementState(SELECTORS.globalErrorDisplay, 'error', errorMsg);
    } finally { if (updateButton) updateButton.disabled = false; logger.info("--- [fetchPayoffChart] END ---"); }
}

/** Helper function to set loading states during asset change */
function setLoadingStateForAssetChange() {
    logger.debug("Setting loading states for asset change..."); // Add log
    // Use setElementState for consistent handling
    setElementState(SELECTORS.expiryDropdown, 'loading', 'Loading Expiries...');
    setElementState(SELECTORS.optionChainTableBody, 'loading', 'Loading Chain...');
    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Loading Analysis...');
    setElementState(SELECTORS.newsResultContainer, 'loading', 'Loading News...');
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot: ...'); // Concise loading
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear previous global error

    // Reset calculation outputs as new asset means previous calcs are invalid
    // Ensure resetCalculationOutputsUI is defined before this point
    if (typeof resetCalculationOutputsUI === 'function') {
        resetCalculationOutputsUI();
    } else {
        logger.error("resetCalculationOutputsUI function not defined when trying to set loading state!");
    }
}

async function sendDebugAssetSelection(asset) {
    // Ensure fetchAPI is defined before calling
    if (typeof fetchAPI !== 'function') {
        logger.error("fetchAPI is not defined. Cannot send debug asset selection.");
        return; // Exit if fetchAPI is missing
    }
    try {
        // Use fetchAPI to handle the request and potential errors
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST',
             // Body is already stringified by fetchAPI if needed, just pass the object
             body: JSON.stringify({ asset: asset })
        });
        // Log success as a warning as it's a debug feature
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        // Log error, but don't block UI flow as it's non-critical
        logger.error("Failed to send debug asset selection:", debugErr.message);
    }
}


/** Handles asset dropdown change */
async function handleAssetChange() {
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown); const selectedAsset = assetDropdown?.value;
    if (!selectedAsset) { logger.info("Asset selection cleared."); activeAsset = null; stopAutoRefresh(); resetPageToInitialState(); return; }
    if (selectedAsset === activeAsset) { logger.debug(`Asset unchanged (${selectedAsset}).`); return; }
    logger.info(`Asset changed to: ${selectedAsset}. Fetching data...`); activeAsset = selectedAsset; stopAutoRefresh();
    previousOptionChainData = {}; previousSpotPrice = 0; currentSpotPrice = 0;
    resetResultsUI(); setLoadingStateForAssetChange(); sendDebugAssetSelection(activeAsset);
    try {
        const [spotResult, expiryResult, analysisResult, newsResult] = await Promise.allSettled([
            fetchNiftyPrice(activeAsset), fetchExpiries(activeAsset), fetchAnalysis(activeAsset), fetchNews(activeAsset)
        ]);
        let hasCriticalError = spotResult.status === 'rejected' || expiryResult.status === 'rejected';
        if (analysisResult.status === 'rejected') logger.error(`Analysis fetch failed: ${analysisResult.reason?.message}`);
        if (newsResult.status === 'rejected') logger.error(`News fetch failed: ${newsResult.reason?.message}`);
        if (!hasCriticalError) { logger.info(`Essential data loaded for ${activeAsset}. Starting auto-refresh.`); startAutoRefresh(); }
        else { logger.error(`Failed loading essential data for ${activeAsset}. Refresh NOT started.`); setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed loading market data for ${activeAsset}.`); }
    } catch (err) {
        logger.error(`Unexpected error in handleAssetChange for ${activeAsset}:`, err); setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed loading data: ${err.message}`); stopAutoRefresh();
        setElementState(SELECTORS.expiryDropdown, 'error'); setElementState(SELECTORS.optionChainTableBody, 'error'); setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error'); setElementState(SELECTORS.analysisResultContainer, 'error'); setElementState(SELECTORS.newsResultContainer, 'error');
    }
}

/** Handles expiry dropdown change */
async function handleExpiryChange() {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    previousOptionChainData = {}; // Clear previous chain data
    if (!expiry) { setElementState(SELECTORS.optionChainTableBody, 'content'); document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select Expiry</td></tr>`; return; }
    logger.info(`Expiry changed to: ${expiry}. Fetching chain...`);
    await fetchOptionChain(true); // Fetch new chain & scroll
}

/** Handles clicks within the option chain table body */
/** Handles clicks within the option chain table body */
function handleOptionChainClick(event) {
    logger.debug("--- handleOptionChainClick START ---"); // Log start
    const targetCell = event.target.closest('td.clickable');
    if (!targetCell) {
        logger.debug("Click was not on a clickable TD cell. Exiting.");
        return; // Ignore clicks elsewhere
    }
    logger.debug("Clicked Cell:", targetCell);

    const row = targetCell.closest('tr');
    logger.debug("Found Row:", row);

    // Check existence of required data attributes
    if (!row || row.dataset.strike === undefined || targetCell.dataset.type === undefined || targetCell.dataset.price === undefined) {
        logger.error("CRITICAL: Missing required data attributes on clicked cell/row.", {
            rowDataset: row?.dataset,
            cellDataset: targetCell?.dataset
        });
        alert("Could not retrieve necessary option details (strike, type, price). Check console.");
        return;
    }
    logger.debug("All required data attributes found.");

    // Parse values carefully
    const rawStrike = row.dataset.strike;
    const rawType = targetCell.dataset.type;
    const rawPrice = targetCell.dataset.price;
    const rawIV = targetCell.dataset.iv;

    logger.debug("Raw Data Attributes:", { rawStrike, rawType, rawPrice, rawIV });

    const strike = parseFloat(rawStrike);
    const type = rawType; // Type is already string 'CE' or 'PE'
    const price = parseFloat(rawPrice);
    const iv = (rawIV !== null && rawIV !== '' && !isNaN(parseFloat(rawIV))) ? parseFloat(rawIV) : null;

    logger.debug("Parsed Values:", { strike, type, price, iv });

    // Final validation of parsed numeric values
    if (!isNaN(strike) && type && !isNaN(price)) {
         logger.debug("Parsed values seem valid. Calling addPosition...");
         addPosition(strike, type, price, iv); // Pass potentially null IV
    } else {
        // This should ideally not happen if the attributes are set correctly
        logger.error('Final data parsing failed unexpectedly in handleOptionChainClick', {
            strike, type, price, iv,
            raw: { strike: rawStrike, type: rawType, price: rawPrice, iv: rawIV }
        });
        alert('An error occurred parsing valid option details after retrieval.');
    }
    logger.debug("--- handleOptionChainClick END ---");
}

/** Handles clicks within the strategy table body */
function handleStrategyTableClick(event) {
     const removeButton = event.target.closest('button.remove-btn'); if (removeButton?.dataset.index) { const index = parseInt(removeButton.dataset.index, 10); if (!isNaN(index)) removePosition(index); return; }
     const toggleButton = event.target.closest('button.toggle-buy-sell'); if (toggleButton?.dataset.index) { const index = parseInt(toggleButton.dataset.index, 10); if (!isNaN(index)) toggleBuySell(index); return; }
}

/** Handles input changes within the strategy table body */
function handleStrategyTableChange(event) {
    if (event.target.matches('input[type="number"].lots-input') && event.target.dataset.index) { const index = parseInt(event.target.dataset.index, 10); if (!isNaN(index)) updateLots(index, event.target.value); }
}

// ===============================================================
// INITIALIZATION (Define AFTER all functions needed by it)
// ===============================================================

/** Loads assets, populates dropdown, sets default */
async function loadAssets() {
    logger.info("Loading assets..."); setElementState(SELECTORS.assetDropdown, 'loading'); let defaultAsset = null;
    try {
        const data = await fetchAPI("/get_assets"); const assets = data?.assets || []; const potentialDefault = assets.includes("NIFTY") ? "NIFTY" : (assets[0] || null);
        populateDropdown(SELECTORS.assetDropdown, assets, "-- Select Asset --", potentialDefault); setElementState(SELECTORS.assetDropdown, 'content');
        const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
        if (assetDropdown && assetDropdown.value) { defaultAsset = assetDropdown.value; logger.info(`Assets loaded. Default: ${defaultAsset}`); }
        else if (assets.length === 0) { logger.warn("No assets found."); setElementState(SELECTORS.assetDropdown, 'error', 'No assets'); }
        else { logger.warn("No default asset determined."); }
    } catch (error) { logger.error("Failed to load assets:", error); populateDropdown(SELECTORS.assetDropdown, [], "-- Error --"); setElementState(SELECTORS.assetDropdown, 'error'); throw error; }
    return defaultAsset;
}

/** Sets up initial event listeners */
function setupEventListeners() {
    logger.info("Setting up event listeners...");
    // Check if handlers exist before adding listeners
    if (typeof handleAssetChange !== 'function') logger.error("handleAssetChange not defined for listener!");
    else document.querySelector(SELECTORS.assetDropdown)?.addEventListener("change", handleAssetChange);
    if (typeof handleExpiryChange !== 'function') logger.error("handleExpiryChange not defined for listener!");
    else document.querySelector(SELECTORS.expiryDropdown)?.addEventListener("change", handleExpiryChange);
    if (typeof fetchPayoffChart !== 'function') logger.error("fetchPayoffChart not defined for listener!");
    else document.querySelector(SELECTORS.updateChartButton)?.addEventListener("click", fetchPayoffChart);
    if (typeof clearAllPositions !== 'function') logger.error("clearAllPositions not defined for listener!");
    else document.querySelector(SELECTORS.clearPositionsButton)?.addEventListener("click", clearAllPositions);
    const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (strategyTableBody) {
        if (typeof handleStrategyTableChange === 'function') strategyTableBody.addEventListener('input', handleStrategyTableChange); else logger.error("handleStrategyTableChange not defined!");
        if (typeof handleStrategyTableClick === 'function') strategyTableBody.addEventListener('click', handleStrategyTableClick); else logger.error("handleStrategyTableClick not defined!");
    } else { logger.warn("Strategy table body not found for listeners."); }
    const optionChainTableBody = document.querySelector(SELECTORS.optionChainTableBody);
    if (optionChainTableBody) {
        if (typeof handleOptionChainClick === 'function') optionChainTableBody.addEventListener('click', handleOptionChainClick); else logger.error("handleOptionChainClick not defined!");
    } else { logger.warn("Option chain table body not found for listeners."); }
}

/** Initializes the page state and fetches initial data */
async function initializePage() {
    logger.info("Initializing page...");
    if (typeof resetResultsUI !== 'function') { logger.error("initializePage: resetResultsUI missing!"); return; } resetResultsUI();
    setElementState(SELECTORS.expiryDropdown, 'content'); document.querySelector(SELECTORS.expiryDropdown).innerHTML = '<option value="">-- Select Asset --</option>';
    setElementState(SELECTORS.optionChainTableBody, 'content'); document.querySelector(SELECTORS.optionChainTableBody).innerHTML = '<tr><td colspan="7">Select Asset & Expiry</td></tr>';
    setElementState(SELECTORS.spotPriceDisplay, 'content'); document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
    setElementState(SELECTORS.analysisResultContainer, 'content'); document.querySelector(SELECTORS.analysisResultContainer).innerHTML = '<p class="placeholder-text">Select asset for analysis.</p>';
    setElementState(SELECTORS.newsResultContainer, 'content'); document.querySelector(SELECTORS.newsResultContainer).innerHTML = '<p class="placeholder-text">Select asset for news.</p>';
    setElementState(SELECTORS.globalErrorDisplay, 'hidden');
    try {
        if (typeof loadAssets !== 'function') throw new Error("loadAssets missing!"); const defaultAsset = await loadAssets();
        if (defaultAsset) { logger.info(`Default asset: ${defaultAsset}. Fetching initial data...`); if (typeof handleAssetChange !== 'function') throw new Error("handleAssetChange missing!"); await handleAssetChange(); }
        else { logger.warn("No default asset. Waiting for selection."); }
    } catch (error) { logger.error("Page Initialization failed:", error); setElementState(SELECTORS.globalErrorDisplay, 'error', `Init Error: ${error.message}`); setElementState(SELECTORS.assetDropdown, 'error'); setElementState(SELECTORS.expiryDropdown, 'error'); setElementState(SELECTORS.spotPriceDisplay, 'error'); }
    logger.info("Initialization sequence complete.");
}

/** Loads the markdown parser script */
function loadMarkdownParser() {
    if (typeof marked !== 'undefined') { logger.info("Markdown parser already loaded."); return; }
    try {
        const script = document.createElement("script"); script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js"; script.async = true;
        script.onload = () => logger.info("Markdown parser loaded dynamically."); script.onerror = () => logger.error("Failed to load Markdown parser."); document.head.appendChild(script);
    } catch (e) { logger.error("Error creating script tag for marked.js", e); }
}

// ===============================================================
// AUTO-REFRESH LOGIC
// ===============================================================

async function refreshLiveData() {
    if (!activeAsset) { logger.warn("Auto-refresh: No active asset."); stopAutoRefresh(); return; }
    const currentExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value; if (!currentExpiry) { logger.debug("Auto-refresh skipped: No expiry."); return; }
    logger.debug(`Auto-refreshing for ${activeAsset}...`);
    const results = await Promise.allSettled([ fetchNiftyPrice(activeAsset, true), fetchOptionChain(false, true) ]);
    results.forEach((result, index) => { if (result.status === 'rejected') { const source = index === 0 ? 'Spot' : 'Chain'; logger.warn(`Auto-refresh: ${source} fetch failed: ${result.reason?.message}`); if (index === 0) setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Err'); } });
    logger.debug(`Auto-refresh cycle finished for ${activeAsset}.`);
}

function startAutoRefresh() {
    stopAutoRefresh(); if (!activeAsset) { logger.info("No active asset, refresh not started."); return; }
    logger.info(`Starting auto-refresh every ${REFRESH_INTERVAL_MS / 1000}s for ${activeAsset}`);
    previousSpotPrice = currentSpotPrice; autoRefreshIntervalId = setInterval(refreshLiveData, REFRESH_INTERVAL_MS);
}

function stopAutoRefresh() {
    if (autoRefreshIntervalId) { clearInterval(autoRefreshIntervalId); autoRefreshIntervalId = null; logger.info("Auto-refresh stopped."); }
}

// ===============================================================
// START EXECUTION (After all functions are defined)
// ===============================================================
document.addEventListener("DOMContentLoaded", () => {
    logger.info("DOM Ready. Initializing...");
    // Check essential functions again before running
    if (typeof loadMarkdownParser === 'function' && typeof initializePage === 'function' && typeof setupEventListeners === 'function') {
        loadMarkdownParser();
        initializePage();
        setupEventListeners();
    } else {
        console.error("CRITICAL ERROR: Core init functions missing. Check script order/definitions.");
        alert("Critical script error. Please refresh.");
    }
});