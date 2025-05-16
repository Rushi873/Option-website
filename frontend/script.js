// ===============================================================
// Configuration & Constants
// ===============================================================
const API_BASE = "http://localhost:8000"; // For Local Hosting
//const API_BASE = "http://localhost:8080";
//const API_BASE = "https://option-strategy-website.onrender.com"; // For Production
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
let scrollAttemptCounter = 0;
let currentLotSize = null;


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
     if (!element) {
         // Log warning only if the element is expected to exist (e.g., not during initial reset perhaps)
         // Consider adding context if possible: logger.warn(`displayMetric: Element not found: "${targetElementSelector}" [Context: Loading/Updating/Error]`);
         logger.warn(`displayMetric: Element not found: "${targetElementSelector}"`);
         return;
     }

     let formattedValue = fallback; // Default to fallback

     // --- Logic to handle different value types ---
     if (value === null || typeof value === 'undefined') {
          formattedValue = fallback;
     } else if (typeof value === 'number' && isFinite(value)) {
          // It's a valid finite number, format it
          const formatFunc = isCurrency ? formatCurrency : formatNumber;
          formattedValue = formatFunc(value, decimals, fallback, isCurrency ? "₹" : "");
     } else if (typeof value === 'string') {
          // Handle specific known strings from backend/formatting first
          const upperVal = value.toUpperCase();
          const knownNonNumericStrings = ["∞", "INFINITY", "-∞", "-INFINITY", "N/A", "UNDEFINED", "LOSS", "...", "Error", "0 / 0", "∞ / ∞", "LOSS / ∞"]; // Add known strings
          if (knownNonNumericStrings.includes(upperVal) || knownNonNumericStrings.includes(value) ) {
               formattedValue = value; // Display known strings as is
          } else {
               // Check if it looks like a pre-formatted string (e.g., breakeven points "23500 / 24000")
               // If it contains numbers and potentially ' / ' but isn't just a number itself
               const seemsPreformatted = /[0-9]/.test(value) && /[\s/]+/.test(value) && isNaN(Number(value.replace(/,/g, ''))); // Basic check
               if (seemsPreformatted) {
                    formattedValue = value; // Assume it's pre-formatted (like joined breakeven points)
                    logger.debug(`displayMetric: Displaying potentially pre-formatted string directly: "${value}" for ${targetElementSelector}`);
               } else {
                    // Treat as a single number/string that might need formatting
                    const formatFunc = isCurrency ? formatCurrency : formatNumber;
                    formattedValue = formatFunc(value, decimals, fallback, isCurrency ? "₹" : "");
               }
          }
     } else {
          // Handle other types if necessary, otherwise use fallback
          formattedValue = String(value); // Attempt string conversion as last resort
     }
     // --- End Logic ---

     element.textContent = `${prefix}${formattedValue}${suffix}`;
     // Ensure error class is removed if we successfully set content
     element.classList.remove('error-message');
     logger.debug(`displayMetric: Set "${targetElementSelector}" to "${prefix}${formattedValue}${suffix}"`);
}
/** Sets the loading/error/content/hidden state for an element using classes */
function setElementState(selectorOrElement, state, message = 'Loading...') {
    const logger = window.logger || console; // Ensure logger is available
    const element = (typeof selectorOrElement === 'string') ? document.querySelector(selectorOrElement) : selectorOrElement;
    if (!element) { logger.warn(`setElementState: Element not found: "${selectorOrElement}"`); return; }

    // --- Identify Element Type ---
    const isMetricsList = element.matches(SELECTORS.metricsList);
    const isSelect = element.tagName === 'SELECT';
    const isButton = element.tagName === 'BUTTON';
    const isTbody = element.tagName === 'TBODY';
    const isTable = element.tagName === 'TABLE';
    // **Refined Container Check** (Avoid matching table/tbody directly)
    const isContainer = !isMetricsList && !isTable && !isTbody && (element.tagName === 'DIV' || element.tagName === 'SECTION' || element.classList.contains('chart-container') || element.tagName === 'UL' || element.tagName === 'DETAILS');
    const isSpan = element.tagName === 'SPAN';
    const isGlobalError = element.id === SELECTORS.globalErrorDisplay?.substring(1); // Safer access

    // --- Reset Previous States ---
    element.classList.remove('is-loading', 'has-error', 'is-loaded', 'is-hidden'); // Use consistent state classes
    if (isSelect || isButton) element.disabled = false;
    element.style.display = ''; // Reset display potentially set by 'hidden'
    if (isGlobalError) element.style.display = 'none'; // Keep hiding global error by default

    // Get default colspan for table/tbody operations
    let defaultColspan = 7;
    const closestGreeksTable = element.closest(SELECTORS.greeksTable);
    const closestChargesTable = element.closest('.charges-table'); // Example selector
    if (closestGreeksTable) defaultColspan = 9;
    else if (closestChargesTable) defaultColspan = 12;
    // Ensure colspan is at least 1
    defaultColspan = Math.max(1, defaultColspan);


    // --- Handle State Application ---
    switch (state) {
        case 'loading':
            if (isContainer) {
                element.classList.add('is-loading');
                // Add a data attribute for the message, CSS can display it if needed
                element.dataset.loadingMessage = message;
                // **IMPORTANT: Do NOT set innerHTML here for containers**
            }
            else if (isTable) {
                // Still okay to clear tbody/tfoot for a Table element when loading
                element.classList.add('is-loading'); // Add class to table too if needed
                const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`;
                const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
                const caption = element.querySelector('caption'); if (caption) caption.textContent = "Loading..."; // Update caption
            }
            else if (isTbody) {
                // Set loading message within the tbody
                element.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`;
            }
            else if (isMetricsList) {
                element.classList.add('is-loading');
                 // Logic to set individual spans to '...' is handled elsewhere (in fetchPayoffChart)
                 logger.debug("Setting metrics list state to loading (class added)");
            }
            else if (isSelect) { element.innerHTML = `<option>${message}</option>`; element.disabled = true; element.classList.add('is-loading'); }
            else if (isButton) { element.disabled = true; element.classList.add('is-loading'); }
            else if (isSpan) { element.textContent = '...'; element.classList.add('is-loading'); }
            else if (isGlobalError) { element.textContent = message; element.style.display = 'block'; element.classList.add('is-loading'); }
            else { element.textContent = message; element.classList.add('is-loading'); } // Fallback
            break;

        case 'error':
            const displayMessage = message || 'Error'; // Use message or default
            if (isContainer) {
                element.classList.add('has-error');
                element.dataset.errorMessage = displayMessage;
                 // **IMPORTANT: Do NOT set innerHTML here for containers**
                 // Optionally, clear loading message attribute
                 delete element.dataset.loadingMessage;
            }
            else if (isTable) {
                element.classList.add('has-error');
                const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`;
                const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
                const caption = element.querySelector('caption'); if (caption) caption.textContent = "Error";
            }
            else if (isTbody) {
                 element.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`;
            }
            else if (isMetricsList) {
                 element.classList.add('has-error');
                 logger.debug("Setting metrics list state to error");
                 const metricSpans = element.querySelectorAll('.metric-value');
                 metricSpans.forEach(span => { span.textContent = 'Error'; span.classList.add('error-message'); });
            }
            else if (isSelect) { element.innerHTML = `<option>${displayMessage}</option>`; element.disabled = true; element.classList.add('has-error'); }
            else if (isButton) { element.classList.add('has-error'); /* Button remains enabled on error? Maybe disable? element.disabled = true; */ }
            else if (isSpan) { element.textContent = 'Error'; element.classList.add('has-error', 'error-message'); }
            else if (isGlobalError) { element.textContent = displayMessage; element.style.display = 'block'; element.classList.add('has-error'); }
            else { element.textContent = displayMessage; element.classList.add('has-error', 'error-message');} // Fallback
            break;

        case 'content':
            element.classList.add('is-loaded');
            // Clear potential messages set via data attributes if they exist
            delete element.dataset.loadingMessage;
            delete element.dataset.errorMessage;
            if (isGlobalError) element.style.display = 'none';
            if (isMetricsList) {
                 // Remove error class from spans if present
                 const metricSpans = element.querySelectorAll('.metric-value.error-message');
                 metricSpans.forEach(span => { span.classList.remove('error-message'); });
            }
             // For tables, the content should be rendered by specific functions (like renderGreeksTable)
             // before setElementState(table, 'content') is called.
             // This case just ensures the correct class is present.
            break;

        case 'hidden':
            element.classList.add('is-hidden');
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
    // (Keep your existing findATMStrikeAsStringKey function as is)
    // ... (your existing code) ...
    const logger = window.logger || console; // Ensure logger is available here too
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
    // Consider adding a check here: if (!closestKey) logger.warn(...)
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
    const funcName = "renderPayoffChart";
    logger.info(`[${funcName}] Attempting to render Plotly chart...`);

    if (!containerElement) {
        logger.error(`[${funcName}] Target container element not found.`);
        return; // Cannot proceed
    }
    // Check if Plotly library is loaded
    if (typeof Plotly === 'undefined' || !Plotly) {
        logger.error(`[${funcName}] Plotly.js library not loaded or failed to initialize.`);
        setElementState(containerElement, 'error', 'Charting library failed to load.');
        return;
    }
     // Check if figureJsonString is a non-empty string
    if (!figureJsonString || typeof figureJsonString !== 'string' || figureJsonString.trim() === '{}' || figureJsonString.trim() === '') {
        logger.error(`[${funcName}] Invalid or empty figure JSON string received. JSON: `, figureJsonString);
        setElementState(containerElement, 'error', 'Invalid or empty chart data received from server.');
        return;
    }

    logger.debug(`[${funcName}] Received Figure JSON string (first 500 chars):`, figureJsonString.substring(0, 500) + (figureJsonString.length > 500 ? '...' : ''));

    let figure;
    try {
        // --- Explicitly Parse JSON ---
        figure = JSON.parse(figureJsonString);
        logger.debug(`[${funcName}] Successfully parsed Figure JSON.`);

        // --- Validate Parsed Figure (Basic Checks) ---
        if (!figure || typeof figure !== 'object') {
            throw new Error("Parsed figure is not a valid object.");
        }
        if (!Array.isArray(figure.data) || figure.data.length === 0) {
             // Allow empty data if layout provides context, but log warning
             logger.warn(`[${funcName}] Parsed figure contains no data traces. Chart might appear empty.`);
             // Example: Ensure data array exists even if empty for Plotly.react
             figure.data = figure.data || [];
        }
        if (!figure.layout || typeof figure.layout !== 'object') {
             logger.warn(`[${func_name}] Parsed figure is missing layout object. Applying defaults.`);
             figure.layout = {}; // Ensure layout exists
        }

        // --- Apply Layout Defaults/Overrides (as before) ---
        figure.layout.height = figure.layout.height || 450; // Keep backend height if provided
        figure.layout.autosize = true;
        figure.layout.margin = figure.layout.margin || { l: 60, r: 30, t: 30, b: 50 };
        figure.layout.template = figure.layout.template || 'plotly_white';
        figure.layout.showlegend = figure.layout.showlegend !== undefined ? figure.layout.showlegend : false; // Keep backend legend setting if present
        figure.layout.hovermode = figure.layout.hovermode || 'x unified';
        figure.layout.font = figure.layout.font || { family: 'Arial, sans-serif', size: 12 };
        // Y-Axis specific
        figure.layout.yaxis = figure.layout.yaxis || {};
        figure.layout.yaxis.title = figure.layout.yaxis.title || { text: 'Profit / Loss (₹)', standoff: 10 };
        figure.layout.yaxis.automargin = true;
        figure.layout.yaxis.gridcolor = figure.layout.yaxis.gridcolor || 'rgba(220, 220, 220, 0.7)';
        figure.layout.yaxis.zeroline = figure.layout.yaxis.zeroline !== undefined ? figure.layout.yaxis.zeroline : true; // Keep backend setting
        figure.layout.yaxis.zerolinecolor = figure.layout.yaxis.zerolinecolor || 'rgba(0, 0, 0, 0.5)';
        figure.layout.yaxis.zerolinewidth = figure.layout.yaxis.zerolinewidth || 1;
        figure.layout.yaxis.tickprefix = figure.layout.yaxis.tickprefix || "₹";
        figure.layout.yaxis.tickformat = figure.layout.yaxis.tickformat || ',.0f';
        // X-Axis specific
        figure.layout.xaxis = figure.layout.xaxis || {};
        figure.layout.xaxis.title = figure.layout.xaxis.title || { text: 'Underlying Spot Price', standoff: 10 };
        figure.layout.xaxis.automargin = true;
        figure.layout.xaxis.gridcolor = figure.layout.xaxis.gridcolor || 'rgba(220, 220, 220, 0.7)';
        figure.layout.xaxis.zeroline = figure.layout.xaxis.zeroline !== undefined ? figure.layout.xaxis.zeroline : false;
        figure.layout.xaxis.tickformat = figure.layout.xaxis.tickformat || ',.0f';

        // Plotly config options
        const plotConfig = {
             responsive: true,
             displayModeBar: true,
             displaylogo: false,
             modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d', 'toImage'] // Customize buttons
        };

        // Ensure container is ready
        containerElement.style.display = ''; // Make sure it's visible
        containerElement.innerHTML = ''; // Clear any previous content/placeholders

        // --- Render the Chart ---
        // Use Plotly.react for efficient updates if plot already exists
        logger.info(`[${funcName}] Calling Plotly.react with ID: ${containerElement.id}`);
        await Plotly.react(containerElement.id, figure.data, figure.layout, plotConfig);

        setElementState(containerElement, 'content'); // Mark as loaded
        logger.info(`[${funcName}] Plotly.react call completed successfully.`);

    } catch (renderError) {
        logger.error(`[${funcName}] Error during Plotly chart processing or rendering:`, renderError);
        // Provide specific error message
        const errorDetail = (renderError instanceof SyntaxError) ? 'Invalid JSON data format.' : renderError.message;
        setElementState(containerElement, 'error', `Chart Display Error: ${errorDetail}`);
    }
}

/** Renders the tax table */
function renderTaxTable(containerElement, taxData) {
    const logger = window.logger || console;
    // Use utility functions if available, otherwise fallback
    const localFormatCurrency = window.formatCurrency || ((val, dec = 2, fallback = 'N/A', prefix = '₹') => {
         // Basic fallback currency formatter
         if (val === null || typeof val === 'undefined') { return fallback; }
         const num = Number(val);
         if (!isNaN(num) && isFinite(num)) {
             return `${prefix}${num.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec })}`;
         }
         return String(val); // Return string representation if not a finite number
    });
    const localFormatNumber = window.formatNumber || ((val, dec = 0, fallback = '-') => {
        // Basic fallback number formatter
        const num = Number(val);
        return (val === null || val === undefined || isNaN(num) || !isFinite(num))
               ? fallback
               : num.toLocaleString(undefined, {minimumFractionDigits: dec, maximumFractionDigits: dec});
    });


    if (!taxData || !taxData.charges_summary || !taxData.breakdown_per_leg || !Array.isArray(taxData.breakdown_per_leg)) {
        containerElement.innerHTML = '<p class="error-message">Charge calculation data unavailable.</p>';
        logger.warn("renderTaxTable called with invalid taxData:", taxData);
        setElementState(containerElement, 'content'); return;
    }

    containerElement.innerHTML = ""; // Clear previous content
    const details = document.createElement('details');
    details.className = "results-details tax-details"; // Add classes as needed
    details.open = false; // Start closed

    const summary = document.createElement('summary');
    summary.innerHTML = `<strong>Estimated Charges Breakdown (Total: ${localFormatCurrency(taxData.total_estimated_cost, 2)})</strong>`;
    details.appendChild(summary);

    // Wrapper div for scrolling
    const tableWrapper = document.createElement('div');
    // Add specific class for potentially unique scrollbar styling if needed
    tableWrapper.className = 'table-wrapper charges-table-wrapper thin-scrollbar';
    details.appendChild(tableWrapper);

    const table = document.createElement("table");
    table.className = "results-table charges-table data-table"; // Ensure all classes are present

    const charges = taxData.charges_summary || {};
    const breakdown = taxData.breakdown_per_leg;

    // Generate table body rows
    const tableBody = breakdown.map(t => {
        const actionDisplay = (t.transaction_type || '').toUpperCase() === 'B' ? 'BUY' : (t.transaction_type || '').toUpperCase() === 'S' ? 'SELL' : '?';
        const typeDisplay = (t.option_type || '').toUpperCase(); // e.g., 'P', 'C'
        const sttNoteText = t.stt_note || ''; // Get the full note

        // Use safe formatting utilities for all numeric cells
        return `
        <tr>
            <td>${actionDisplay}</td>
            <td>${typeDisplay}</td>
            <td>${localFormatNumber(t.strike, 2, '-')}</td>
            <td>${localFormatNumber(t.lots, 0, '-')}</td>
            <td>${localFormatNumber(t.premium_per_share, 2, '-')}</td>
            <td>${localFormatNumber(t.stt ?? 0, 2)}</td>
            <td>${localFormatNumber(t.stamp_duty ?? 0, 2)}</td>
            <td>${localFormatNumber(t.sebi_fee ?? 0, 4)}</td>
            <td>${localFormatNumber(t.txn_charge ?? 0, 4)}</td>
            <td>${localFormatNumber(t.brokerage ?? 0, 2)}</td>
            <td>${localFormatNumber(t.gst ?? 0, 2)}</td>
            <td class="note" title="${sttNoteText}">${sttNoteText}</td>
        </tr>`;
        // --- End of Correction ---
    }).join('');

    // Extract totals safely
    const total_stt = charges.stt ?? 0;
    const total_stamp = charges.stamp_duty ?? 0;
    const total_sebi = charges.sebi_fee ?? 0;
    const total_txn = charges.txn_charges ?? 0;
    const total_brokerage = charges.brokerage ?? 0;
    const total_gst = charges.gst ?? 0;
    const overall_total = taxData.total_estimated_cost ?? 0;

    // Set table innerHTML
    table.innerHTML = `
        <thead>
            <tr>
                <th>Act</th><th>Type</th><th>Strike</th><th>Lots</th><th>Premium</th>
                <th>STT</th><th>Stamp</th><th>SEBI</th><th>Txn</th><th>Broker</th><th>GST</th>
                <th title="STT Note">STT Note</th>
            </tr>
        </thead>
        <tbody>
            ${tableBody}
        </tbody>
        <tfoot>
            <tr class="totals-row">
                <td colspan="5">Total Estimated Charges</td>
                <td>${localFormatCurrency(total_stt, 2)}</td>
                <td>${localFormatCurrency(total_stamp, 2)}</td>
                <td>${localFormatCurrency(total_sebi, 4)}</td>
                <td>${localFormatCurrency(total_txn, 4)}</td>
                <td>${localFormatCurrency(total_brokerage, 2)}</td>
                <td>${localFormatCurrency(total_gst, 2)}</td>
                <td style="font-weight: bold;">${localFormatCurrency(overall_total, 2)}</td>
            </tr>
        </tfoot>`;

    tableWrapper.appendChild(table);
    containerElement.appendChild(details);

    setElementState(containerElement, 'content');
    logger.info("Tax table rendered.");
}

/** Renders the Greeks table and calculates/returns portfolio totals */
/** Renders the Greeks table and calculates/returns portfolio totals (v3 - Consistent Totals) */
function renderGreeksTable(tableElement, greeksList) {
    const logger = window.logger || console;
    // Clear previous content safely
    tableElement.innerHTML = ''; // Clear header, body, footer, caption

    if (!tableElement || !(tableElement instanceof HTMLTableElement)) {
        logger.error("renderGreeksTable: Invalid tableElement provided.");
        return null; // Cannot proceed
    }

    // Recreate basic table structure
    const caption = tableElement.createCaption();
    caption.className = "table-caption";
    const thead = tableElement.createTHead();
    const tbody = tableElement.createTBody();
    const tfoot = tableElement.createTFoot();

    // --- Initialize Totals ---
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    let hasCalculatedAnyTotal = false;
    let processedLegsCount = 0;
    let skippedLegsCount = 0;

    // --- Handle Invalid Input List ---
    if (!Array.isArray(greeksList)) {
        caption.textContent = "Error: Invalid Greeks data received.";
        tbody.innerHTML = `<tr><td colspan="9" class="error-message">Invalid Greeks data format from server.</td></tr>`;
        setElementState(tableElement, 'error');
        return null;
    }

    const totalLegsInput = greeksList.length;

    // --- Set Table Header (Using Per Lot Titles) ---
    thead.innerHTML = `<tr><th>Action</th><th>Lots</th><th>Type</th><th>Strike</th><th title="Delta/Lot">Δ Delta / Lot</th><th title="Gamma/Lot">Γ Gamma / Lot</th><th title="Theta/Lot(Day)">Θ Theta / Lot</th><th title="Vega/Lot(1% IV)">Vega / Lot</th><th title="Rho/Lot(1% Rate)">Ρ Rho / Lot</th></tr>`;


    // --- Handle Empty Input List ---
    if (totalLegsInput === 0) {
        caption.textContent = "Portfolio Option Greeks (No Legs)";
        tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">No option legs in the strategy.</td></tr>`;
        setElementState(tableElement, 'content');
        return totals; // Return zero totals
    }


    // --- Process Each Leg ---
    greeksList.forEach((g, index) => {
        const row = tbody.insertRow();
        let legIsValidForTotal = false;
        let legSkippedReason = "Unknown";
        let lotsValueForCalc = NaN; // Initialize for calculation check

        try {
            // Validate essential data structures for this leg
            const inputData = g?.input_data;
            const gv_per_lot = g?.calculated_greeks_per_lot; // Use per-lot data

            if (!inputData || typeof inputData !== 'object') {
                 throw new Error("Missing or invalid input_data field");
            }
             if (!gv_per_lot || typeof gv_per_lot !== 'object') {
                 throw new Error("Missing or invalid calculated_greeks_per_lot field");
            }

            // --- Extract and Validate Input Data for Display and Calculation ---
            // *** CORRECTED LOTS HANDLING: Use SINGULAR 'lot' key ***
            const lots = inputData.lot; // Access the key 'lot' from the backend response
            // *******************************************************

            // Validate if it's a valid, non-zero integer
            if (typeof lots !== 'number' || !Number.isInteger(lots) || lots === 0) {
                 // Use the original value from inputData['lot'] in the error message
                 throw new Error(`Invalid or missing 'lot' value in input_data (${inputData.lot})`);
            }
            lotsValueForCalc = lots; // Store the valid integer for calculation

            const strike = inputData.strike; // Expect number
            const tr_type = inputData.tr_type?.toLowerCase(); // 'b' or 's'
            const op_type = inputData.op_type?.toLowerCase(); // 'c' or 'p'

            // Validate other inputs needed for display (should be okay if backend worked)
            if (typeof strike !== 'number' || strike <= 0) throw new Error(`Invalid strike (${strike})`);
            if (tr_type !== 'b' && tr_type !== 's') throw new Error(`Invalid tr_type (${tr_type})`);
            if (op_type !== 'c' && op_type !== 'p') throw new Error(`Invalid op_type (${op_type})`);


            // --- Display Leg Info in Table Row ---
            const actionDisplay = (tr_type === 'b') ? 'BUY' : 'SELL';
            const typeDisplay = (op_type === 'c') ? 'CE' : 'PE';
            const lotsDisplay = `${lotsValueForCalc}`; // Display the validated integer

            row.insertCell().textContent = actionDisplay;
            row.insertCell().textContent = lotsDisplay;
            row.insertCell().textContent = typeDisplay;
            row.insertCell().textContent = formatNumber(strike, 2); // Assuming strike is number

            // *** Display PER LOT Greeks ***
            // Add checks for existence and type before formatting
            if (typeof gv_per_lot.delta !== 'number') throw new Error("Missing/invalid gv_per_lot.delta");
            row.insertCell().textContent = formatNumber(gv_per_lot.delta, 4, '-');

            if (typeof gv_per_lot.gamma !== 'number') throw new Error("Missing/invalid gv_per_lot.gamma");
            row.insertCell().textContent = formatNumber(gv_per_lot.gamma, 4, '-');

             if (typeof gv_per_lot.theta !== 'number') throw new Error("Missing/invalid gv_per_lot.theta");
            row.insertCell().textContent = formatNumber(gv_per_lot.theta, 4, '-');

             if (typeof gv_per_lot.vega !== 'number') throw new Error("Missing/invalid gv_per_lot.vega");
            row.insertCell().textContent = formatNumber(gv_per_lot.vega, 4, '-');

             if (typeof gv_per_lot.rho !== 'number') throw new Error("Missing/invalid gv_per_lot.rho");
            row.insertCell().textContent = formatNumber(gv_per_lot.rho, 4, '-');


            // --- Calculate Leg Contribution to Portfolio Total ---
            // Ensure all per-lot greeks are finite numbers before calculating total
            const legDelta = Number.isFinite(gv_per_lot.delta) ? gv_per_lot.delta * lotsValueForCalc : NaN;
            const legGamma = Number.isFinite(gv_per_lot.gamma) ? gv_per_lot.gamma * lotsValueForCalc : NaN;
            const legTheta = Number.isFinite(gv_per_lot.theta) ? gv_per_lot.theta * lotsValueForCalc : NaN;
            const legVega = Number.isFinite(gv_per_lot.vega) ? gv_per_lot.vega * lotsValueForCalc : NaN;
            const legRho = Number.isFinite(gv_per_lot.rho) ? gv_per_lot.rho * lotsValueForCalc : NaN;

            // Check if ALL calculated leg contributions are valid numbers
            if ( [legDelta, legGamma, legTheta, legVega, legRho].every(v => Number.isFinite(v)) )
            {
                // Add to totals
                totals.delta += legDelta;
                totals.gamma += legGamma;
                totals.theta += legTheta;
                totals.vega += legVega;
                totals.rho += legRho;
                legIsValidForTotal = true;
                hasCalculatedAnyTotal = true; // Mark that at least one leg contributed
                row.classList.add('greeks-calculated');
            } else {
                 // Identify which specific Greek calculation failed for better logging
                 let nonFiniteDetails = [];
                 if (!Number.isFinite(legDelta)) nonFiniteDetails.push(`Delta (${gv_per_lot.delta} * ${lotsValueForCalc})`);
                 if (!Number.isFinite(legGamma)) nonFiniteDetails.push(`Gamma (${gv_per_lot.gamma} * ${lotsValueForCalc})`);
                 if (!Number.isFinite(legTheta)) nonFiniteDetails.push(`Theta (${gv_per_lot.theta} * ${lotsValueForCalc})`);
                 if (!Number.isFinite(legVega)) nonFiniteDetails.push(`Vega (${gv_per_lot.vega} * ${lotsValueForCalc})`);
                 if (!Number.isFinite(legRho)) nonFiniteDetails.push(`Rho (${gv_per_lot.rho} * ${lotsValueForCalc})`);
                 throw new Error(`Non-finite total Greek value(s): ${nonFiniteDetails.join(', ')}`);
            }
            processedLegsCount++;

        } catch (error) { // Catch errors during validation or calculation for this leg
            legSkippedReason = error.message || "Processing error";
            logger.warn(`renderGreeksTable: Skipping leg ${index + 1} from total calculation. Reason: ${legSkippedReason}. Data Received:`, g);
            skippedLegsCount++;
            // Ensure row is cleared and error message is shown
            while (row.cells.length > 0) { row.deleteCell(0); } // Clear potentially half-added cells
            const errorCell = row.insertCell();
            errorCell.colSpan = 9; // Span all columns
            errorCell.textContent = `Leg ${index + 1}: Error (${legSkippedReason})`; // Display specific error
            errorCell.className = 'greeks-skipped skipped-leg error-message'; // Style as skipped/error
            row.style.opacity = '0.6';
            row.style.fontStyle = 'italic';
        }
    }); // End forEach loop

    // --- Update Caption ---
    caption.textContent = `Portfolio Option Greeks (${processedLegsCount} Processed, ${skippedLegsCount} Skipped)`;

    // --- Render Footer Row with Totals ---
    // Clear previous footer content if any
    tfoot.innerHTML = '';
    const footerRow = tfoot.insertRow();
    footerRow.className = 'totals-row';

    if (hasCalculatedAnyTotal) { // Only show totals if at least one leg was calculated
        const headerCell = footerRow.insertCell();
        headerCell.colSpan = 4; // Action, Lots, Type, Strike
        headerCell.textContent = 'Total Portfolio Exposure';
        headerCell.style.textAlign = 'right';
        headerCell.style.fontWeight = 'bold';
        // Display formatted totals
        footerRow.insertCell().textContent = formatNumber(totals.delta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.gamma, 4);
        footerRow.insertCell().textContent = formatNumber(totals.theta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.vega, 4);
        footerRow.insertCell().textContent = formatNumber(totals.rho, 4);
        setElementState(tableElement, 'content');
    } else { // Handle cases where no legs could be calculated
        const cell = footerRow.insertCell(); cell.colSpan = 9;
        cell.textContent = (totalLegsInput > 0 && skippedLegsCount === totalLegsInput)
            ? 'Could not calculate totals for any leg.'
            : (totalLegsInput === 0 ? 'No legs to calculate.' : 'Totals unavailable.'); // More specific message
        cell.style.textAlign = 'center'; cell.style.fontStyle = 'italic';
        setElementState(tableElement, 'content'); // Still content, just no totals data
    }

    // --- Return Final Calculated Totals ---
    // Return the raw totals object for potential further use (like analysis)
    const finalTotals = { delta: totals.delta, gamma: totals.gamma, theta: totals.theta, vega: totals.vega, rho: totals.rho };
    logger.info(`renderGreeksTable (Per Lot): Rendered ${processedLegsCount}/${totalLegsInput}. Final Calculated Totals: ${JSON.stringify(finalTotals)}`);
    return finalTotals; // Return object, even if all values are 0
}


// ===============================================================
// CORE LOGIC & EVENT HANDLERS (Define before Initialization uses them)
// ===============================================================

/** Resets ONLY the calculation output areas */
/** Resets ONLY the calculation output areas (More Robust) */
function resetCalculationOutputsUI() {
    logger.debug("Resetting calculation output UI elements...");

    // --- Payoff Chart ---
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    if (chartContainer) {
        try {
            if (typeof Plotly !== 'undefined' && chartContainer._fullLayout) {
                 Plotly.purge(chartContainer.id);
                 logger.debug("Plotly chart purged.");
            }
        } catch (e) { logger.warn("Error purging Plotly chart:", e); }
        chartContainer.innerHTML = '<div class="placeholder-text">Add legs and click "Update Strategy"</div>';
        setElementState(chartContainer, 'content');
    } else { logger.warn("Payoff chart container not found for reset."); }

    // --- Tax Info ---
    const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
    if (taxContainer) {
        taxContainer.innerHTML = '<p class="placeholder-text">Update strategy to calculate charges.</p>';
        setElementState(taxContainer, 'content');
    } else { logger.warn("Tax info container not found for reset."); }

    // --- Greeks Table ---
    const greeksTable = document.querySelector(SELECTORS.greeksTable);
    const greeksSection = document.querySelector(SELECTORS.greeksTableContainer);
    if (greeksTable) {
        const caption = greeksTable.querySelector('caption'); if (caption) caption.textContent = 'Portfolio Option Greeks';
        const tbody = greeksTable.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">Update strategy to calculate Greeks.</td></tr>`;
        const tfoot = greeksTable.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = "";
        setElementState(greeksTable, 'content');
    } else { logger.warn("Greeks TABLE element not found for reset."); }
    if (greeksSection) setElementState(greeksSection, 'content');
    else { logger.warn("Greeks SECTION container not found for reset.")}

    // --- Greeks Analysis ---
    // const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection);
    // const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
    // if (greeksAnalysisSection) setElementState(greeksAnalysisSection, 'hidden');
    // else { logger.warn("Greeks analysis SECTION not found for reset.")}
    // if (greeksAnalysisContainer) { greeksAnalysisContainer.innerHTML = ''; setElementState(greeksAnalysisContainer, 'content'); }
    // else { logger.warn("Greeks analysis RESULT container not found for reset.")}

    // --- Metrics ---
    const metricsList = document.querySelector(SELECTORS.metricsList);
    if (metricsList) {
        // Reset individual metrics
        if (document.querySelector(SELECTORS.maxProfitDisplay)) displayMetric('N/A', SELECTORS.maxProfitDisplay);
        if (document.querySelector(SELECTORS.maxLossDisplay)) displayMetric('N/A', SELECTORS.maxLossDisplay);
        if (document.querySelector(SELECTORS.breakevenDisplay)) displayMetric('N/A', SELECTORS.breakevenDisplay); // Reset BE
        if (document.querySelector(SELECTORS.rewardToRiskDisplay)) displayMetric('N/A', SELECTORS.rewardToRiskDisplay); // Reset RR
        if (document.querySelector(SELECTORS.netPremiumDisplay)) displayMetric('N/A', SELECTORS.netPremiumDisplay, '', '', 2, true); // Reset NP
        setElementState(metricsList, 'content');
    } else { logger.warn("Metrics LIST container not found for reset."); }

    // --- Cost Breakdown (REMOVED) ---
    // const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
    // const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
    // if (breakdownContainer) {
    //     if (breakdownList) {
    //         breakdownList.innerHTML = "";
    //         setElementState(breakdownList, 'content');
    //     } else { logger.warn("Cost breakdown LIST element not found for reset."); }
    //     breakdownContainer.open = false;
    //     setElementState(breakdownContainer, 'hidden');
    // } else { logger.warn("Cost breakdown CONTAINER element not found for reset."); }
    logger.debug("Cost Breakdown reset logic skipped."); // Optional log

    // --- Warning Message ---
    const warningContainer = document.querySelector(SELECTORS.warningContainer);
    if (warningContainer) {
        warningContainer.textContent = '';
        setElementState(warningContainer, 'hidden');
    } else { logger.warn("Warning container element not found for reset."); }

    logger.debug("Calculation output UI reset complete.");
}

/** Updates the strategy table in the UI from the global `strategyPositions` array */
/** Updates the strategy table in the UI from the global `strategyPositions` array */
function updateStrategyTable() {
    // ***** LOG: Function Start *****
    console.log("==============================================");
    console.log("[updateStrategyTable] Function executing.");
    // Log a DEEP COPY of the array to prevent modification issues during logging
    console.log("[updateStrategyTable] Rendering from strategyPositions:", JSON.parse(JSON.stringify(strategyPositions)));
    console.log("[updateStrategyTable] strategyPositions array length:", strategyPositions?.length ?? 'undefined');
    console.log("==============================================");
    // ********************************

    const tableBody = document.querySelector(SELECTORS.strategyTableBody);

    // --- 1. Validate Target Element ---
    if (!tableBody) {
        console.error(`[updateStrategyTable] CRITICAL FAILURE: Target table body element "${SELECTORS.strategyTableBody}" NOT FOUND in the DOM. Cannot update table.`);
        // Display error to user
        setElementState(SELECTORS.globalErrorDisplay, 'error', 'Internal Error: Strategy table display component missing.');
        return; // Cannot proceed
    }
    console.log("[updateStrategyTable] Found target table body element:", tableBody);

    // --- 2. Clear Previous Content ---
    console.log("[updateStrategyTable] Clearing tableBody innerHTML...");
    tableBody.innerHTML = ""; // Clear previous rows safely
    console.log("[updateStrategyTable] tableBody innerHTML cleared.");

    // --- 3. Handle Empty Strategy ---
    if (!Array.isArray(strategyPositions) || strategyPositions.length === 0) {
        console.log("[updateStrategyTable] No positions in the array, rendering placeholder.");
        // Use a class for consistent placeholder styling
        tableBody.innerHTML = `<tr><td colspan="7" class="placeholder-text">Click option prices to add legs...</td></tr>`;
        // Ensure parent section is visible if needed (optional, depends on initial state)
        // setElementState(tableBody.closest('section'), 'content');
        return; // Nothing more to do
    }

    // --- 4. Iterate and Render Rows ---
    console.log(`[updateStrategyTable] Starting loop for ${strategyPositions.length} positions.`);
    let rowsAppendedCount = 0;

    strategyPositions.forEach((pos, index) => {
        // --- 4a. Validate Position Object ---
        if (!pos || typeof pos !== 'object') {
            console.error(`[updateStrategyTable] !!! ERROR: Skipping invalid item at index ${index}. Item:`, pos);
            // Optionally render a placeholder row for the invalid item
            const errorRow = document.createElement('tr');
            errorRow.innerHTML = `<td colspan="7" class="error-message">Error loading leg ${index + 1}</td>`;
            try { tableBody.appendChild(errorRow); } catch(e) { console.error("Failed to append error row", e); }
            return; // Skip to the next iteration
        }

        console.log(`[updateStrategyTable] Loop iteration ${index}. Processing valid position:`, JSON.parse(JSON.stringify(pos))); // Deep copy for log

        try {
            const row = document.createElement('tr');
            row.dataset.index = index; // Essential for later updates/removals
            // Use integer lots for logic, format absolute value for display
            const lotsInt = parseInt(pos.lots || 0, 10); // Ensure lots is an integer
            const isLong = lotsInt >= 0; // Treat 0 as long for class? Or neutral? Let's stick to >=0 for BUY
            row.className = isLong ? "long-position" : "short-position";

            // --- 4b. Create Cells (Order matches recommended HTML header) ---

            // Cell 1: Action (Buy/Sell Toggle Button)
            const actionCell = row.insertCell();
            actionCell.style.textAlign = 'center'; // Ensure button is centered
            const toggleButton = document.createElement('button');
            const positionType = isLong ? "BUY" : "SELL";
            const buttonClass = isLong ? "button-buy" : "button-sell";
            toggleButton.textContent = positionType;
            toggleButton.className = `toggle-buy-sell ${buttonClass}`;
            toggleButton.dataset.index = index;
            toggleButton.title = "Click to toggle Buy/Sell";
            actionCell.appendChild(toggleButton);

            // Cell 2: Lots Input
            const lotsCell = row.insertCell();
            lotsCell.style.textAlign = 'center'; // Ensure input is centered relative to cell
            const lotsInput = document.createElement('input');
            lotsInput.type = 'number';
            lotsInput.className = 'lots-input number-input-small'; // Use specific class for styling if needed
            lotsInput.value = Math.abs(lotsInt); // Display absolute value
            lotsInput.dataset.index = index;
            lotsInput.min = 1; // User inputs positive lots
            lotsInput.max = 100; // Example max
            lotsInput.step = 1;
            lotsInput.required = true;
            lotsInput.title = "Enter number of lots (positive)";
            lotsCell.appendChild(lotsInput);

            // Cell 3: Option Type (CE/PE)
            const typeCell = row.insertCell();
            typeCell.textContent = pos.option_type || 'N/A'; // Use || 'N/A' as fallback
            typeCell.className = pos.option_type === 'CE' ? 'call' : (pos.option_type === 'PE' ? 'put' : '');
            typeCell.style.fontWeight = 'bold';

            // Cell 4: Strike Price
            const strikeCell = row.insertCell();
            // Check if strike exists and is number before formatting
            const strikePriceNum = parseFloat(pos.strike_price);
            strikeCell.textContent = !isNaN(strikePriceNum) ? formatNumber(strikePriceNum, strikePriceNum % 1 === 0 ? 0 : 2) : 'N/A';
            strikeCell.className = 'strike'; // Add class if needed for styling
            strikeCell.style.textAlign = 'right'; // Typically right-aligned

            // Cell 5: Last Price (Premium)
            const priceCell = row.insertCell();
            // Check if price exists before formatting
            priceCell.textContent = formatCurrency(pos.last_price, 2, 'N/A'); // formatCurrency handles non-numbers
            priceCell.className = 'price';
            priceCell.style.textAlign = 'right'; // Typically right-aligned

            // Cell 6: IV (Implied Volatility)
            const ivCell = row.insertCell();
            // IV is stored as decimal (e.g., 0.15), display as percentage
            const ivValue = pos.iv;
            ivCell.textContent = (ivValue !== null && ivValue !== undefined && !isNaN(ivValue))
                ? `${formatNumber(ivValue * 100, 1)}%` // Display as XX.X%
                : '-'; // Use dash for missing IV
            ivCell.className = 'iv';
            ivCell.style.textAlign = 'right';
            ivCell.title = "Implied Volatility";

            // Cell 7: Remove Button
            const removeCell = row.insertCell();
            removeCell.style.textAlign = 'center';
            const removeButton = document.createElement('button');
            removeButton.textContent = '✖'; // Use a clear symbol
            removeButton.className = 'remove-btn button-danger button-small'; // Style as needed
            removeButton.dataset.index = index;
            removeButton.title = "Remove this leg";
            removeCell.appendChild(removeButton);

            // --- 4c. Append Row to Table Body ---
            // Check connection right before appending (Paranoid check)
            if (!tableBody.isConnected) {
                 console.error(`[updateStrategyTable] !!! CRITICAL ERROR: tableBody disconnected from DOM before appending row ${index} !!!`);
                 // Potentially stop or try to recover, though this indicates a larger issue elsewhere
                 setElementState(SELECTORS.globalErrorDisplay, 'error', 'Internal Error: Strategy table became unavailable during update.');
                 return; // Stop the loop if table vanished
            }
            tableBody.appendChild(row);
            rowsAppendedCount++; // Increment counter

            console.log(`[updateStrategyTable] Row ${index}: Successfully created and appended.`);

        } catch (loopError) {
            // Catch errors during row/cell creation or append
            console.error(`[updateStrategyTable] !!! ERROR processing/appending row ${index} !!! Data:`, JSON.parse(JSON.stringify(pos)), "Error:", loopError);
            // Optionally render an error row here too
             const errorRow = document.createElement('tr');
             errorRow.innerHTML = `<td colspan="7" class="error-message">Error rendering leg ${index + 1}: ${loopError.message}</td>`;
             try { tableBody.appendChild(errorRow); } catch(e) { console.error("Failed to append error row after loop error", e); }
        }
    });

    // --- 5. Final Log ---
    console.log(`[updateStrategyTable] Finished loop. ${rowsAppendedCount} of ${strategyPositions.length} positions successfully rendered and appended.`);
    // Log final state only if debugging deep DOM issues:
    // console.log("[updateStrategyTable] Final tbody innerHTML:", tableBody.innerHTML);
}

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
// ===============================================================

/** Gathers VALID strategy leg data from the global `strategyPositions` array */
/** Gathers VALID strategy leg data from the global `strategyPositions` array
 *  Formats data for the backend API (/get_payoff_chart), aligning with
 *  the expectations of the Python `prepare_strategy_data` function.
 */
function gatherStrategyLegsFromTable() {
    logger.debug("--- [gatherStrategyLegs] START ---");
    if (!Array.isArray(strategyPositions)) {
        logger.error("[gatherStrategyLegs] strategyPositions is not an array.");
        return []; // Return empty if state is invalid
    }
    if (strategyPositions.length === 0) {
        logger.warn("[gatherStrategyLegs] strategyPositions is empty.");
        return [];
    }
    logger.debug("[gatherStrategyLegs] Raw strategyPositions Source:", JSON.parse(JSON.stringify(strategyPositions)));

    const formattedLegsForAPI = [];
    let invalidLegCount = 0;

    strategyPositions.forEach((pos, index) => {
        let legIsValid = true;
        let validationErrors = []; // Collect specific errors

        // --- Validate each field from the frontend state (pos object) ---
        if (!pos || typeof pos !== 'object') {
            validationErrors.push("Position data is not a valid object.");
            legIsValid = false;
        } else {
            // op_type ('CE'/'PE' -> 'c'/'p')
            if (!pos.option_type || (pos.option_type !== 'CE' && pos.option_type !== 'PE')) {
                validationErrors.push(`Invalid option_type: ${pos.option_type}`); legIsValid = false;
            }
            // strike (must be positive number)
            const strikeNum = parseFloat(pos.strike_price);
            if (isNaN(strikeNum) || strikeNum <= 0) {
                validationErrors.push(`Invalid strike_price: ${pos.strike_price}`); legIsValid = false;
            }
            // lots (must be non-zero integer) -> tr_type ('b'/'s'), lot (positive integer)
            const lotsInt = parseInt(pos.lots);
            if (isNaN(lotsInt) || lotsInt === 0) {
                validationErrors.push(`Invalid lots: ${pos.lots}`); legIsValid = false;
            }
             // tr_type derived from lots sign is inherently valid ('b' or 's') if lotsInt is valid non-zero

            // op_pr (last_price, must be non-negative number)
            const priceNum = parseFloat(pos.last_price);
            if (isNaN(priceNum) || priceNum < 0) {
                 // Allow 0 price, log warning if negative
                 if (priceNum < 0) logger.warn(`[gatherStrategyLegs] Leg ${index+1}: Negative last_price (${pos.last_price}) detected, sending 0.`);
                 // validationErrors.push(`Invalid last_price: ${pos.last_price}`); legIsValid = false; // Don't fail for 0
            }
             // lot_size (must be positive integer)
            const lotSizeInt = parseInt(pos.lot_size);
            if (isNaN(lotSizeInt) || lotSizeInt <= 0) {
                 validationErrors.push(`Invalid lot_size: ${pos.lot_size}`); legIsValid = false;
            }
            // iv (can be null, but if number, must be non-negative)
            let ivNum = null;
            if (pos.iv !== null && pos.iv !== undefined) {
                 ivNum = parseFloat(pos.iv);
                 if (isNaN(ivNum) || ivNum < 0) {
                      validationErrors.push(`Invalid iv: ${pos.iv}`); legIsValid = false;
                 }
            }
            // days_to_expiry (must be non-negative integer)
            const dteInt = parseInt(pos.days_to_expiry);
            if (isNaN(dteInt) || dteInt < 0) {
                 validationErrors.push(`Invalid days_to_expiry: ${pos.days_to_expiry}`); legIsValid = false;
            }
            // expiry_date (must be YYYY-MM-DD string)
            if (!pos.expiry_date || !/^\d{4}-\d{2}-\d{2}$/.test(pos.expiry_date)) {
                 validationErrors.push(`Invalid expiry_date format: ${pos.expiry_date}`); legIsValid = false;
            }
        }

        // --- If Valid, Format for Backend ---
        if (legIsValid) {
            const formattedLeg = {
                 // Keys expected by backend's prepare_strategy_data
                 op_type: pos.option_type === 'CE' ? 'c' : 'p',   // 'c' or 'p'
                 strike: String(pos.strike_price),               // String
                 tr_type: parseInt(pos.lots) >= 0 ? 'b' : 's',    // 'b' or 's'
                 op_pr: String(Math.max(0, parseFloat(pos.last_price))), // String, ensure non-negative
                 lot: String(Math.abs(parseInt(pos.lots))),     // String, positive integer
                 lot_size: String(pos.lot_size),                 // String
                 // Send IV as is (null or number), backend handles validation/extraction
                 iv: (pos.iv !== null && pos.iv !== undefined) ? parseFloat(pos.iv) : null, // number or null
                 // Send DTE if valid, otherwise rely on backend calculation from expiry_date
                 days_to_expiry: parseInt(pos.days_to_expiry), // number (integer >= 0)
                 expiry_date: pos.expiry_date                 // String YYYY-MM-DD
            };
            formattedLegsForAPI.push(formattedLeg);
            logger.debug(`[gatherStrategyLegs] Leg ${index+1} formatted successfully:`, formattedLeg);
        } else {
            logger.error(`[gatherStrategyLegs] Skipping invalid leg ${index+1}. Errors: ${validationErrors.join(', ')}. Raw Data:`, JSON.parse(JSON.stringify(pos)));
            invalidLegCount++;
        }
    });

    // --- User Feedback on Skipped Legs ---
    if (invalidLegCount > 0 && formattedLegsForAPI.length === 0) {
        // Only show alert if ALL legs were invalid
        const errorMsg = `Error: ${invalidLegCount} invalid strategy leg(s) found, NO valid legs remaining. Calculation cannot proceed. Please check console for details on invalid legs.`;
        logger.error(errorMsg);
        setElementState(SELECTORS.globalErrorDisplay, 'error', errorMsg);
        // Don't auto-hide critical error
        // alert(errorMsg); // Less ideal than using globalErrorDisplay
    } else if (invalidLegCount > 0) {
        // Show warning if some legs were skipped but others remain
        const warnMsg = `Warning: ${invalidLegCount} invalid leg(s) were ignored. Calculation will proceed based on ${formattedLegsForAPI.length} valid leg(s). Check console for details.`;
        logger.warn(warnMsg);
        setElementState(SELECTORS.warningContainer, 'content', warnMsg); // Use warning container
        setElementState(SELECTORS.warningContainer, 'content'); // Make visible
        // Optional: Auto-hide warning
        // setTimeout(() => setElementState(SELECTORS.warningContainer, 'hidden'), 7000);
    }

    logger.debug(`[gatherStrategyLegs] Returning ${formattedLegsForAPI.length} formatted legs (ignored ${invalidLegCount}).`);
    logger.debug("--- [gatherStrategyLegs] END ---");
    return formattedLegsForAPI; // Return the list of formatted dicts
}


/** Adds a position to the global `strategyPositions` array (UPDATED TO READ currentLotSize) */
function addPosition(strike, type, price, iv) {
    console.log("--- addPosition START ---", { strike, type, price, iv });

    // --- Gather Prerequisites ---
    const expiryDropdown = document.querySelector(SELECTORS.expiryDropdown);
    const expiry = expiryDropdown?.value;
    const dte = expiry ? calculateDaysToExpiry(expiry) : null;
    // const lotSize = getLotSizeForAsset(activeAsset); // <--- REMOVE THIS LINE
    const lotSize = currentLotSize; // <--- READ FROM GLOBAL VARIABLE

    // --- Validate Prerequisites ---
    let errors = [];
    if (!activeAsset) { errors.push("No active asset selected."); }
    if (!expiry) { errors.push("No expiry date selected."); }
    if (dte === null || dte < 0) { errors.push(`Invalid Days to Expiry (${dte}) for expiry '${expiry}'.`); }
    // Check the global variable directly
    if (!lotSize || isNaN(lotSize) || lotSize <= 0) {
        errors.push(`Lot size unknown or invalid (${lotSize}) for asset '${activeAsset}'. Was it fetched correctly?`);
    }

    if (errors.length > 0) {
        const errorMsg = `Cannot add position: ${errors.join(' ')}`;
        logger.error("[addPosition] ABORTED:", errorMsg);
        setElementState(SELECTORS.globalErrorDisplay, 'error', errorMsg);
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
        return; // Stop execution
    }

    console.log('[addPosition] Prerequisites Valid:', { activeAsset, expiry, dte, lotSize }); // Log the fetched lotSize

    // --- Prepare Data ---
    const lastPrice = (price !== null && !isNaN(price) && price >= 0) ? parseFloat(price) : 0;
    const impliedVol = (iv !== null && !isNaN(iv) && iv >= 0) ? parseFloat(iv) : null;

    // --- Create Position Object ---
    const newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type,
        lots: 1,
        tr_type: 'b',
        last_price: lastPrice,
        iv: impliedVol,
        days_to_expiry: dte,
        lot_size: lotSize // Use the validated global variable
    };

    console.log("[addPosition] Constructed newPosition:", JSON.parse(JSON.stringify(newPosition)));
    console.log("[addPosition] strategyPositions BEFORE push:", JSON.parse(JSON.stringify(strategyPositions)));

    // --- Add to Array ---
    try {
        strategyPositions.push(newPosition);
        console.log("[addPosition] strategyPositions AFTER push:", JSON.parse(JSON.stringify(strategyPositions)));
        console.log(`[addPosition] Position added successfully to array. Current count: ${strategyPositions.length}`);
    } catch (e) {
        console.error("[addPosition] CRITICAL ERROR during strategyPositions.push!", e);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to store position: ${e.message}`);
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
        return;
    }

    // --- Update UI ---
    if (typeof updateStrategyTable === 'function') {
        console.log('[addPosition] >>> Calling updateStrategyTable NOW. strategyPositions length:', strategyPositions.length);
        try {
            updateStrategyTable();
            console.log("[addPosition] <<< updateStrategyTable call completed successfully.");
        } catch (updateError) {
            console.error("[addPosition] !!! ERROR occurred directly within updateStrategyTable call !!!", updateError);
            setElementState(SELECTORS.globalErrorDisplay, 'error', `Error updating strategy table UI: ${updateError.message}`);
            setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
        }
    } else {
        console.error("[addPosition] !!! CRITICAL: updateStrategyTable function is MISSING or NOT a function. UI cannot be updated. !!!");
        setElementState(SELECTORS.globalErrorDisplay, 'error', "Internal Error: Cannot update strategy table display.");
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
    }

    console.log("--- addPosition END ---");
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
    const callId = ++scrollAttemptCounter;
    const callTime = Date.now();

    logger.debug(`[Scroll #${callId} @ ${callTime}] triggerATMScroll called immediately. Key: '${atmKeyToUse}'. Tbody is HTMLElement: ${tbodyElement instanceof HTMLElement}`);

    if (!(tbodyElement instanceof HTMLElement)) {
        logger.error(`[Scroll #${callId}] Invalid tbody passed. Aborting.`);
        return;
    }
    if (!atmKeyToUse) {
        logger.warn(`[Scroll #${callId}] Null/empty key. Aborting.`);
        return;
    }
    // Check connection right away
    if (!tbodyElement.isConnected) {
        logger.warn(`[Scroll #${callId}] FAILED: tbodyElement is NOT CONNECTED to the DOM on call!`);
        return;
    }

    const numericATMStrike = parseFloat(atmKeyToUse);
    if (isNaN(numericATMStrike)) {
        logger.warn(`[Scroll #${callId}] Could not parse numeric strike. Aborting.`);
        return;
    }

    // --- Log DOM state BEFORE querying ---
    const currentHTML = tbodyElement.innerHTML;
    const firstFewChars = currentHTML.substring(0, 300);
    logger.debug(`[Scroll #${callId}] Tbody valid & connected. Current innerHTML (start): \n\`\`\`html\n${firstFewChars}${currentHTML.length > 300 ? '...' : ''}\n\`\`\``);
    // --- End Log DOM state ---

    try {
        const primarySelector = `tr[data-strike="${numericATMStrike}"]`;
        logger.debug(`[Scroll #${callId}] Attempting primary query: "${primarySelector}"`);
        let atmRow = tbodyElement.querySelector(primarySelector);
        logger.debug(`[Scroll #${callId}] Primary query result:`, atmRow);

        if (!atmRow) {
            const fallbackSelector = `tr[data-strike-key="${atmKeyToUse}"]`;
            logger.debug(`[Scroll #${callId}] Primary failed. Fallback: "${fallbackSelector}"`);
            atmRow = tbodyElement.querySelector(fallbackSelector);
            logger.debug(`[Scroll #${callId}] Fallback query result:`, atmRow);
        }

        if (atmRow) {
            logger.info(`[Scroll #${callId}] SUCCESS: ATM Row Found. Scrolling...`, atmRow);
            atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
            logger.debug(`[Scroll #${callId}] Scrolled to: ${atmRow.dataset.strike || atmRow.dataset.strikeKey}`);
            atmRow.classList.add("highlight-atm");
            // Keep highlight removal timeout
            setTimeout(() => {
                if (atmRow?.parentNode) {
                    atmRow.classList.remove("highlight-atm");
                    logger.debug(`[Scroll #${callId}] Removed highlight class.`);
                } else {
                    logger.debug(`[Scroll #${callId}] Highlight remove skipped, row gone.`);
                }
            }, 2000);
        } else {
            logger.warn(`[Scroll #${callId}] FAILED: ATM row ('${atmKeyToUse}' / ${numericATMStrike}) not found immediately.`);
            const rowsPresentByStrike = Array.from(tbodyElement.querySelectorAll('tr[data-strike]')).map(r => r.dataset.strike);
            const rowsPresentByKey = Array.from(tbodyElement.querySelectorAll('tr[data-strike-key]')).map(r => r.dataset.strikeKey);
            logger.warn(`[Scroll #${callId}] Rows now via [data-strike] (${rowsPresentByStrike.length}): [${rowsPresentByStrike.join(', ')}]`);
            logger.warn(`[Scroll #${callId}] Rows now via [data-strike-key] (${rowsPresentByKey.length}): [${rowsPresentByKey.join(', ')}]`);
        }
    } catch (e) {
        logger.error(`[Scroll #${callId}] Error inside scroll logic:`, e);
    }
}


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
    const asset = activeAsset;
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    const optionChainTable = document.getElementById('optionChainTable');

    if (!optionChainTable) { logger.error("Option chain table element not found."); return; }
    let targetTbody = optionChainTable.querySelector('tbody');

    if (!asset || !expiry) {
        if (targetTbody) { targetTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">Select Asset & Expiry</td></tr>`; if (!isRefresh) setElementState(targetTbody, 'content'); }
        previousOptionChainData = {}; return;
    }

    if (!isRefresh) {
        if (!targetTbody) { targetTbody = document.createElement('tbody'); optionChainTable.appendChild(targetTbody); logger.warn("Created tbody for loading state."); }
        setElementState(targetTbody, 'loading', 'Loading Chain...');
    }

    try {
        if (currentSpotPrice <= 0 && scrollToATM && !isRefresh) {
            logger.info("Spot price needed for ATM scroll, fetching...");
            try { await fetchNiftyPrice(asset); } catch (e) { /* Logged by fetchNiftyPrice */ }
            if (currentSpotPrice <= 0) { logger.warn("Spot unavailable, cannot calc ATM."); scrollToATM = false; }
        }

        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);
        const currentChainData = data?.option_chain;

        if (!currentChainData || typeof currentChainData !== 'object' || Object.keys(currentChainData).length === 0) {
            logger.warn(`No chain data for ${asset}/${expiry}.`);
            const errorTbody = optionChainTable.querySelector('tbody');
            if (errorTbody) { errorTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">No chain data found</td></tr>`; if (!isRefresh) setElementState(errorTbody, 'content'); }
            previousOptionChainData = {}; return;
        }

        const strikeStringKeys = Object.keys(currentChainData).sort((a, b) => Number(a) - Number(b));
        const atmStrikeObjectKey = currentSpotPrice > 0 ? findATMStrikeAsStringKey(strikeStringKeys, currentSpotPrice) : null;
        const newTbody = document.createElement('tbody');

        strikeStringKeys.forEach((strikeStringKey) => {
            const optionData = currentChainData[strikeStringKey] || {};
            const call = optionData.call || {}; const put = optionData.put || {};
            const strikeNumericValue = parseFloat(strikeStringKey);
            if (isNaN(strikeNumericValue)) { logger.warn(`Skipping non-numeric strike key: ${strikeStringKey}`); return; }

            const prevOptionData = previousOptionChainData[strikeStringKey] || {};
            const prevCall = prevOptionData.call || {}; const prevPut = prevOptionData.put || {};

            const tr = newTbody.insertRow();
            // *** Key Fix for ATM Scroll: Set data-strike reliably ***
            tr.dataset.strike = strikeNumericValue; // Use the precise numeric value
            tr.dataset.strikeKey = strikeStringKey; // Keep original key if needed

            if (atmStrikeObjectKey === strikeStringKey) tr.classList.add("atm-strike");

            const columns = [
                { class: 'call clickable price', type: 'CE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') }, { class: 'call oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') }, { class: 'call iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'strike', isStrike: true, format: val => formatNumber(val, val % 1 === 0 ? 0 : 2) },
                { class: 'put iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` }, { class: 'put oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') }, { class: 'put clickable price', type: 'PE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') },
            ];

            columns.forEach(col => {
                try {
                    const td = tr.insertCell(); td.className = col.class;
                    let currentValue; let sourceObj = null; let prevDataObject = null;
                    if (col.isStrike) { currentValue = strikeNumericValue; }
                    else { sourceObj = col.class.includes('call') ? call : put; prevDataObject = col.class.includes('call') ? prevCall : prevPut; currentValue = sourceObj?.[col.dataKey]; }
                    td.textContent = col.format(currentValue);

                    // *** Key Fix for Adding Positions: Set data-* attributes robustly ***
                    if (col.type && sourceObj) {
                        td.dataset.type = col.type;
                        // Ensure price is a string '0' if missing/null/undefined
                        td.dataset.price = String(sourceObj['last_price'] ?? '0');
                        // Ensure IV is an empty string '' if missing/null/undefined
                        const ivValue = sourceObj['implied_volatility'];
                        td.dataset.iv = (ivValue !== null && ivValue !== undefined) ? String(ivValue) : '';
                        // Log the attributes being set for debugging
                        // logger.debug(`Set data on TD: type=${td.dataset.type}, price=${td.dataset.price}, iv=${td.dataset.iv}`);
                    }

                    // Highlighting logic (Improved comparison)
                    if (isRefresh && !col.isStrike && prevDataObject) {
                        let previousValue = prevDataObject[col.dataKey]; let changed = false; const currentExists = currentValue != null; const previousExists = previousValue != null;
                        if (currentExists && previousExists) { changed = (typeof currentValue === 'number' && typeof previousValue === 'number') ? Math.abs(currentValue - previousValue) > 0.001 : String(col.format(currentValue)) !== String(col.format(previousValue)); }
                        else if (currentExists !== previousExists) { changed = true; } if (changed) { highlightElement(td); }
                    } else if (isRefresh && !col.isStrike && currentValue != null && !prevDataObject) { highlightElement(td); }
                } catch (cellError) { logger.error(`Error cell ${strikeStringKey}/${col.dataKey}:`, cellError); const errorTd = tr.insertCell(); errorTd.textContent = 'ERR'; errorTd.className = col.class + ' error-message'; }
            });
        });

        const oldTbody = optionChainTable.querySelector('tbody');
        if (oldTbody) { optionChainTable.replaceChild(newTbody, oldTbody); }
        else { logger.warn("Old tbody not found, appending new."); optionChainTable.appendChild(newTbody); }

        if (!isRefresh) { setElementState(newTbody, 'content'); }
        previousOptionChainData = currentChainData;

        if (scrollToATM && atmStrikeObjectKey !== null && !isRefresh) {
             // *** Pass the correct tbody (newTbody) to the scroll function ***
             triggerATMScroll(newTbody, atmStrikeObjectKey);
        }

    } catch (error) {
        logger.error(`Error fetchOptionChain ${asset}/${expiry}:`, error);
        const errorTbody = optionChainTable.querySelector('tbody');
        if (errorTbody) { errorTbody.innerHTML = `<tr><td colspan="7" class="error-message">Chain Error: ${error.message}</td></tr>`; if (!isRefresh) setElementState(errorTbody, 'error'); }
        if (isRefresh) { logger.warn(`Chain refresh failed: ${error.message}`); }
        previousOptionChainData = {};
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

// /** Fetches Greeks analysis */
// async function fetchAndDisplayGreeksAnalysis(asset, portfolioGreeksData) {
//     const container = document.querySelector(SELECTORS.greeksAnalysisResultContainer); const section = document.querySelector(SELECTORS.greeksAnalysisSection);
//     if (!container || !section) { logger.error("Greeks analysis container/section not found."); return; }
//     if (!asset || !portfolioGreeksData || typeof portfolioGreeksData !== 'object') { logger.warn("Greeks analysis skipped: Invalid input."); setElementState(section, 'hidden'); return; }
//     const allZeroOrNull = Object.values(portfolioGreeksData).every(v => v === null || Math.abs(v) < 1e-9);
//     if (allZeroOrNull) { logger.info("Greeks analysis skipped: All Greeks zero."); container.innerHTML = '<p class="placeholder-text">No net option exposure.</p>'; setElementState(section, 'content'); setElementState(container, 'content'); return; }
//     logger.info(`Fetching Greeks analysis for ${asset}...`); setElementState(section, 'content'); setElementState(container, 'loading', 'Fetching Greeks analysis...');
//     try {
//         if (typeof marked === 'undefined') { logger.warn("Waiting for marked.js..."); await new Promise(resolve => setTimeout(resolve, 200)); if (typeof marked === 'undefined') throw new Error("Markdown parser failed."); }
//         const requestBody = { asset_symbol: asset, portfolio_greeks: portfolioGreeksData };
//         const data = await fetchAPI("/get_greeks_analysis", { method: "POST", body: JSON.stringify(requestBody) });
//         const rawAnalysis = data?.greeks_analysis || "*Greeks analysis failed.*"; const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
//         container.innerHTML = marked.parse(potentiallySanitized); setElementState(container, 'content'); logger.info(`Rendered Greeks analysis for ${asset}`);
//     } catch (error) { logger.error(`Error fetching/rendering Greeks analysis for ${asset}:`, error); setElementState(container, 'error', `Greeks Analysis Error: ${error.message}`); setElementState(section, 'content'); }
// }


/** Fetches payoff chart and all calculation results */
/** Fetches payoff chart and all calculation results (Greeks Analysis Removed) */
async function fetchPayoffChart() {
    logger.info("--- [fetchPayoffChart] START ---");
    const updateButton = document.querySelector(SELECTORS.updateChartButton);
    const asset = activeAsset;

    // --- 1. Initial Checks & Data Gathering ---
    if (!asset) {
        alert("Select asset first.");
        logger.error("[fetchPayoff] No active asset.");
        return;
    }

    logger.debug("[fetchPayoff] Gathering legs...");
    const currentStrategyLegs = gatherStrategyLegsFromTable(); // Assumes validation inside

    if (!currentStrategyLegs || currentStrategyLegs.length === 0) {
        logger.warn("[fetchPayoff] No valid legs formatted for API call.");
        resetCalculationOutputsUI(); // Reset UI
        setElementState(SELECTORS.globalErrorDisplay, 'error', 'Please add valid strategy legs first.');
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 4000);
        return;
    }
    logger.info(`[fetchPayoff] Proceeding with ${currentStrategyLegs.length} valid legs.`);

    // Log data being sent
    console.log("[fetchPayoff] --- DATA SENT TO BACKEND (/get_payoff_chart) ---");
    try {
        console.log(JSON.stringify({ asset: asset, strategy: currentStrategyLegs }, null, 2));
    } catch (e) { console.error("Error stringifying request",e); }
    console.log("-------------------------------------------------------------");

    // --- 2. UI Reset & Loading States ---
    logger.debug("[fetchPayoff] Resetting output UI before fetch...");
    resetCalculationOutputsUI(); // Use the robust version which checks elements

    logger.debug("[fetchPayoff] Setting loading states...");
    // Set loading states robustly
    if (document.querySelector(SELECTORS.payoffChartContainer)) setElementState(SELECTORS.payoffChartContainer, 'loading', 'Calculating...'); else logger.warn("Payoff chart container missing for loading state.")
    if (document.querySelector(SELECTORS.taxInfoContainer)) setElementState(SELECTORS.taxInfoContainer, 'loading'); else logger.warn("Tax container missing for loading state.")
    if (document.querySelector(SELECTORS.greeksTableContainer)) setElementState(SELECTORS.greeksTableContainer, 'loading'); else logger.warn("Greeks container missing for loading state.")
    // Don't try to set loading state on greeksTable directly if container check is enough
    // if (document.querySelector(SELECTORS.greeksTable)) setElementState(SELECTORS.greeksTable, 'loading');
   // if (document.querySelector(SELECTORS.greeksAnalysisSection)) setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
    if (document.querySelector(SELECTORS.metricsList)) {
        setElementState(SELECTORS.metricsList, 'loading');
        // Set metrics text to '...' using textContent directly after checking element exists
        const updateMetricPlaceholder = (selector) => {
            const el = document.querySelector(selector);
            if (el) el.textContent = '...';
        };
        updateMetricPlaceholder(SELECTORS.maxProfitDisplay);
        updateMetricPlaceholder(SELECTORS.maxLossDisplay);
        updateMetricPlaceholder(SELECTORS.breakevenDisplay);
        updateMetricPlaceholder(SELECTORS.rewardToRiskDisplay);
        updateMetricPlaceholder(SELECTORS.netPremiumDisplay);
    } else { logger.warn("Metrics list missing for loading state.") }
    // Cost breakdown removed
    if (document.querySelector(SELECTORS.warningContainer)) setElementState(SELECTORS.warningContainer, 'hidden');

    if (updateButton) updateButton.disabled = true;

    // --- 3. API Call ---
    const requestData = { asset: asset, strategy: currentStrategyLegs };
    try {
        const data = await fetchAPI('/get_payoff_chart', { method: 'POST', body: JSON.stringify(requestData) });

        // --- 4. Process API Response ---
        console.log("--- [fetchPayoffChart] DEBUG START: Processing API Response ---");
        console.log("Raw API Response 'data':", data);

        // Validate response structure
        if (!data || typeof data !== 'object') {
             throw new Error("Invalid response format from server (not an object or null).");
        }
        if (data.success === false) { // Handle explicit backend failure flag
             throw new Error(data.message || "Calculation failed server-side (explicit failure).");
        }

        logger.info("[fetchPayoff] API call successful. Processing results...");

        // --- 4a. Render Metrics (Using Detailed Logic + Element Checks) ---
                // --- 4a. Render Metrics (Using Detailed Logic + Element Checks) ---
        const metricsContainerData = data.metrics;
        const metrics = metricsContainerData?.metrics;
        const metricsListElement = document.querySelector(SELECTORS.metricsList); // Find parent list

        console.log("Extracted 'data.metrics.metrics':", metrics); // Keep this log
        console.log(`Checking for Metrics List Container (${SELECTORS.metricsList}):`, metricsListElement); // Keep this log

        if (metrics && typeof metrics === 'object' && metricsListElement) {
            logger.debug("[fetchPayoff] Rendering metrics...");

            // Find elements FIRST (relative to list for robustness)
            const mpElement = metricsListElement.querySelector('#maxProfit .metric-value'); // Use ID directly here
            const mlElement = metricsListElement.querySelector('#maxLoss .metric-value');
            const beElement = metricsListElement.querySelector('#breakeven .metric-value');
            const rrElement = metricsListElement.querySelector('#rewardToRisk .metric-value');
            const npElement = metricsListElement.querySelector('#netPremium .metric-value');

            console.log(`  Metric Elements Found: MP=${!!mpElement}, ML=${!!mlElement}, BE=${!!beElement}, RR=${!!rrElement}, NP=${!!npElement}`); // Keep this log

            // --- Update elements Correctly ---

            // Max Profit
            if (mpElement) {
                // *** FIX: Pass metrics.max_profit ***
                console.log(`DEBUG: About to display Max Profit. Value: ${metrics.max_profit}`);
                displayMetric(metrics.max_profit, SELECTORS.maxProfitDisplay); // Use correct value
            } else { logger.warn("Max profit display element missing."); }

            // Max Loss
            if (mlElement) {
                // *** FIX: Pass metrics.max_loss ***
                 console.log(`DEBUG: About to display Max Loss. Value: ${metrics.max_loss}`);
                displayMetric(metrics.max_loss, SELECTORS.maxLossDisplay); // Use correct value
            } else { logger.warn("Max loss display element missing."); }

            // Breakeven Points
            if (beElement) {
                let beText = "N/A"; // Default
                const beValue = metrics.breakeven_points;
                logger.debug(`Processing Breakeven Value: ${JSON.stringify(beValue)} (Type: ${typeof beValue})`);

                // *** FIX: Ensure array values are formatted and handle single/multiple points ***
                if (Array.isArray(beValue) && beValue.length > 0) {
                    // Format each number in the array before joining
                    beText = beValue.map(point => {
                        // Try formatting, fallback to original string if formatting fails
                        return formatNumber(point, 2, String(point));
                    }).join(' / '); // Join multiple points with slash
                } else if (typeof beValue === 'string' && beValue.trim() !== '' && beValue.toUpperCase() !== 'N/A') {
                    // If it's a single string value from backend (but not 'N/A'), display it
                    beText = formatNumber(beValue, 2, beValue); // Try formatting as number first
                }
                // If beValue is null, undefined, empty array, or 'N/A' string, beText remains "N/A"

                logger.debug(`DEBUG: Final Breakeven text: '${beText}'`);
                beElement.textContent = beText; // Set the text content directly
            } else { logger.warn("Breakeven display element missing."); }

            // Reward:Risk Ratio
            if (rrElement) {
                // *** FIX: Pass metrics.reward_to_risk_ratio ***
                 console.log(`DEBUG: About to display R:R. Value: ${metrics.reward_to_risk_ratio}`);
                displayMetric(metrics.reward_to_risk_ratio, SELECTORS.rewardToRiskDisplay); // Use correct value
            } else { logger.warn("Reward:Risk display element missing."); }

            // Net Premium
            if (npElement) {
                // *** FIX: Pass metrics.net_premium ***
                const npValue = metrics.net_premium;
                 console.log(`DEBUG: About to display Net Premium. Value: ${npValue}`);
                // displayMetric handles currency formatting and number check
                displayMetric(npValue, SELECTORS.netPremiumDisplay, '', '', 2, true); // Use correct value, isCurrency=true
            } else { logger.warn("Net Premium display element missing."); }

            // --- End Updates ---

            setElementState(metricsListElement, 'content'); // Mark metrics section as loaded

            // Handle Warnings
            // ... (warning handling code remains the same) ...
             const warnings = metrics.warnings;
             const warningElement = document.querySelector(SELECTORS.warningContainer);
             if (warningElement) {
                  if (Array.isArray(warnings) && warnings.length > 0) {
                     console.log("Displaying Calculation Warnings:", warnings);
                     warningElement.innerHTML = `<strong>Calculation Warnings:</strong><ul>${warnings.map(w => `<li>${w}</li>`).join('')}</ul>`;
                     setElementState(warningElement, 'content');
                  } else { setElementState(warningElement, 'hidden'); }
             } else { logger.warn("[fetchPayoff] Warning container element not found."); }


        } else { // Handle missing metrics data or container
            logger.error("[fetchPayoff] Metrics data in response OR Metrics list container element not found. Cannot display metrics.");
            if (metricsListElement) setElementState(metricsListElement, 'error', 'Metrics Error');
        }


        // --- 4b. Render Other Sections ---
        console.log("--- [fetchPayoffChart] DEBUG: Processing Other Sections ---");

        // Cost Breakdown (REMOVED)
        logger.debug("Cost Breakdown rendering skipped.");

        // Tax Table
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
        console.log(`Checking Tax Container (${SELECTORS.taxInfoContainer}):`, taxContainer);
        if (taxContainer) {
            console.log("Tax Data Received:", data.charges);
            if (data.charges) { renderTaxTable(taxContainer, data.charges); setElementState(taxContainer, 'content'); }
            else { logger.warn("[fetchPayoff] Charges data missing."); taxContainer.innerHTML = "<p class='placeholder-text'>Charge data unavailable.</p>"; setElementState(taxContainer, 'content'); }
        } else { logger.warn("[fetchPayoff] Tax info container element not found."); }

        // Chart
        const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
        console.log(`Checking Chart Container (${SELECTORS.payoffChartContainer}):`, chartContainer);
        const chartDataKey = "chart_figure_json";
        if (chartContainer) {
            console.log("Chart JSON Key Exists:", data.hasOwnProperty(chartDataKey));
            if (data[chartDataKey]) { logger.debug("Rendering Payoff Chart..."); await renderPayoffChart(chartContainer, data[chartDataKey]); }
            else { logger.error("[fetchPayoff] Chart JSON missing."); setElementState(chartContainer, 'error', 'Chart unavailable.'); }
        } else { logger.warn("[fetchPayoff] Payoff chart container element not found."); }


        
        
        // Greeks
        console.log("--- [fetchPayoffChart] DEBUG: Processing Other Sections ---");

        // ... (Cost Breakdown, Tax Table, Chart rendering code remains the same) ...

        // --- Greeks Section ---
        const greeksSectionElement = document.querySelector(SELECTORS.greeksTableContainer); // Find the main Greeks container (e.g., #greeksSection)
        console.log(`Checking Greeks Section Container (${SELECTORS.greeksTableContainer}):`, greeksSectionElement);

        // Also find the elements needed for the analysis part
        const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); // The whole analysis section (e.g., #greeksAnalysisSection)
        const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer); // The inner div for results (e.g., #greeksAnalysisResult)
        console.log(`Checking Greeks Analysis Section (${SELECTORS.greeksAnalysisSection}):`, greeksAnalysisSection);
        console.log(`Checking Greeks Analysis Result Container (${SELECTORS.greeksAnalysisResultContainer}):`, greeksAnalysisContainer);


        if (greeksSectionElement) {
             // Attempt to find the Greeks table *inside* the container element
             const greeksTableElement = greeksSectionElement.querySelector('#greeksTable');
             console.log(`Checking Greeks Table (#greeksTable found within Section Container):`, greeksTableElement); // Log if the table element itself was found

             console.log("Greeks Data Received from API:", data.greeks); // Log the raw greeks data

             // Check if the greeks data from the API is valid (exists and is an array)
             if (data.greeks && Array.isArray(data.greeks)) {

                 // Check if the HTML table element was actually found in the DOM
                 if (greeksTableElement) {
                     logger.debug("[fetchPayoff] Found Greeks table element. Rendering Greeks Table...");

                     // Call the function to render the table and get calculated totals
                     const calculatedTotals = renderGreeksTable(greeksTableElement, data.greeks); // Function should handle its own internal errors/formatting

                     // Set the state of the main Greeks section to 'content' after rendering attempt
                     setElementState(greeksSectionElement, 'content');
                     logger.debug("[fetchPayoff] Greeks table rendering attempted. Section state set to 'content'.");


                     // --- Handle Greeks Analysis Display ---
                     // Check if we got valid totals AND if the analysis section elements exist
                     if (calculatedTotals && typeof calculatedTotals === 'object' && greeksAnalysisSection && greeksAnalysisContainer) {
                         logger.debug("[fetchPayoff] Checking calculated Greeks totals for analysis:", calculatedTotals);

                         // Check if any Greek total has a meaningful non-zero value (check for null, NaN, and very small numbers)
                         const hasMeaningfulGreeks = Object.values(calculatedTotals).some(v => v !== null && !isNaN(v) && Math.abs(v) > 1e-9); // Added !isNaN check

                         if (hasMeaningfulGreeks) {
                             logger.info("[fetchPayoff] Meaningful Greeks totals found. Fetching and displaying Greeks Analysis...");
                             // Call the function to fetch and display the analysis
                             // This function should handle its own loading/content/error states for greeksAnalysisSection and greeksAnalysisContainer
                             fetchAndDisplayGreeksAnalysis(asset, calculatedTotals);
                             // No need to set state for analysis section here, let fetchAndDisplayGreeksAnalysis manage it.

                         } else {
                             // Totals are zero or effectively zero, skip analysis and show placeholder
                             logger.info("[fetchPayoff] Greeks analysis skipped: Calculated totals are zero or negligible.");
                             greeksAnalysisContainer.innerHTML = '<p class="placeholder-text">No net option exposure, or Greeks are effectively zero. Analysis skipped.</p>';
                             setElementState(greeksAnalysisSection, 'content'); // Show the analysis section...
                             setElementState(greeksAnalysisContainer, 'content'); // ...with the placeholder message loaded.
                         }
                     } else {
                        // Hide analysis section if totals are null/invalid OR if the required HTML elements for analysis are missing
                        if (greeksAnalysisSection) {
                            setElementState(greeksAnalysisSection, 'hidden');
                        }
                        // Log why analysis is not proceeding
                        if (!calculatedTotals || typeof calculatedTotals !== 'object') {
                             logger.warn("[fetchPayoff] Greeks analysis skipped: Calculated totals were null or invalid from renderGreeksTable.");
                        }
                        if (!greeksAnalysisSection || !greeksAnalysisContainer) {
                             logger.warn("[fetchPayoff] Greeks analysis skipped: Analysis section HTML elements not found.");
                        }
                     }
                     // --- End Greeks Analysis Handling ---

                 } else {
                     // ERROR CASE: Greeks data received, but the #greeksTable element was NOT found within #greeksSection
                     logger.error(`[fetchPayoff] Greeks TABLE element ('#greeksTable') NOT FOUND within the section ('${SELECTORS.greeksTableContainer}'). Check HTML structure, IDs, and selectors.`);
                     setElementState(greeksSectionElement, 'error', 'Greeks table structure missing.'); // Set error state on the SECTION
                     // Ensure analysis section is hidden if the table structure fails
                     if (greeksAnalysisSection) {
                         setElementState(greeksAnalysisSection, 'hidden');
                     }
                 }
             } else {
                 // DATA UNAVAILABLE CASE: Greeks data missing or invalid format from API response
                 logger.warn("[fetchPayoff] Greeks data missing from API response or is not an array. Displaying 'unavailable' message.");
                 // Display a placeholder message directly within the section container
                 greeksSectionElement.innerHTML = '<h3 class="section-subheader">Options Greeks</h3><p class="placeholder-text">Greeks data unavailable.</p>';
                 setElementState(greeksSectionElement, 'content'); // Mark as content loaded (but showing the unavailable message)
                 // Ensure analysis section is hidden if greeks data is missing
                 if (greeksAnalysisSection) {
                     setElementState(greeksAnalysisSection, 'hidden');
                 }
             }
        } else {
             // CONTAINER NOT FOUND CASE: The main Greeks section container itself wasn't found
             logger.warn(`[fetchPayoff] Greeks SECTION container element ('${SELECTORS.greeksTableContainer}') not found in the DOM.`);
             // Hide the analysis section if its parent container is missing
             if (greeksAnalysisSection) {
                 setElementState(greeksAnalysisSection, 'hidden');
                 logger.debug("[fetchPayoff] Hiding analysis section because main Greeks container is missing.");
             }
        }

        console.log("--- [fetchPayoffChart] DEBUG END: Processing Complete ---");
        logger.info("[fetchPayoff] Successfully processed API results (or handled errors).");

    } catch (error) { // Main catch block for fetchAPI errors or errors thrown during processing
        const errorMsg = error.message || 'Calculation failed or server error.';
        logger.error(`[fetchPayoff] CRITICAL ERROR during fetch or processing: ${errorMsg}`, error);

        // --- Robust Reset UI to Error State ---
        // Helper to safely set error state
        const safeSetError = (selector, msg = 'Error', logWarnMsg = '') => {
            const el = document.querySelector(selector);
            if (el) {
                setElementState(selector, 'error', msg);
            } else {
                logger.warn(logWarnMsg || `[fetchPayoff] Element '${selector}' not found for setting error state.`);
            }
        };

        safeSetError(SELECTORS.payoffChartContainer, `Calculation Error: ${errorMsg}`);
        safeSetError(SELECTORS.taxInfoContainer);
        safeSetError(SELECTORS.greeksTableContainer); // Set error on the greeks container
        safeSetError(SELECTORS.metricsList, 'Metrics Error');

        // Set metric text to 'Error'
        const setErrorText = (selector) => { const el = document.querySelector(selector); if (el) el.textContent = 'Error'; };
        setErrorText(SELECTORS.maxProfitDisplay);
        setErrorText(SELECTORS.maxLossDisplay);
        setErrorText(SELECTORS.breakevenDisplay);
        setErrorText(SELECTORS.rewardToRiskDisplay);
        setErrorText(SELECTORS.netPremiumDisplay);

        // --- Hide dependent/optional sections on error ---
        const safeHide = (selector, logWarnMsg = '') => {
             const el = document.querySelector(selector);
             if (el) {
                 setElementState(selector, 'hidden');
             } else {
                  logger.warn(logWarnMsg || `[fetchPayoff] Element '${selector}' not found for hiding during error handling.`);
             }
        };
        safeHide(SELECTORS.greeksAnalysisSection); // Hide analysis section on any major error
        safeHide(SELECTORS.warningContainer);
        // Cost breakdown removed

        // Display global error message
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Calculation Error: ${errorMsg}`);

    } finally {
        // Ensure the update button is always re-enabled, regardless of success or failure
        const updateButton = document.querySelector(SELECTORS.updateChartButton); // Find button again just in case
        if (updateButton) {
            updateButton.disabled = false;
            logger.debug("[fetchPayoff] Update button re-enabled.");
        } else {
             logger.warn("[fetchPayoff] Could not find update button to re-enable in finally block.");
        }
        logger.info("--- [fetchPayoffChart] END ---");
    }
} // IMPORTANT: Mak





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
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
    const selectedAsset = assetDropdown?.value;

    if (!selectedAsset) {
        logger.info("Asset selection cleared.");
        activeAsset = null;
        currentLotSize = null; // Clear lot size
        stopAutoRefresh();
        // resetPageToInitialState(); // Consider what reset means here
        resetResultsUI(); // Reset calculation outputs and strategy table
        // Reset dependent dropdowns/displays
        setElementState(SELECTORS.expiryDropdown, 'content'); document.querySelector(SELECTORS.expiryDropdown).innerHTML = '<option value="">-- Select Asset --</option>';
        setElementState(SELECTORS.optionChainTableBody, 'content'); document.querySelector(SELECTORS.optionChainTableBody).innerHTML = '<tr><td colspan="7">Select Asset & Expiry</td></tr>';
        setElementState(SELECTORS.spotPriceDisplay, 'content'); document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
        setElementState(SELECTORS.analysisResultContainer, 'content'); document.querySelector(SELECTORS.analysisResultContainer).innerHTML = '<p class="placeholder-text">Select asset for analysis.</p>';
        setElementState(SELECTORS.newsResultContainer, 'content'); document.querySelector(SELECTORS.newsResultContainer).innerHTML = '<p class="placeholder-text">Select asset for news.</p>';
        return;
    }

    if (selectedAsset === activeAsset) {
        logger.debug(`Asset unchanged (${selectedAsset}).`);
        return;
    }

    logger.info(`Asset changed to: ${selectedAsset}. Fetching data...`);
    activeAsset = selectedAsset; // Set the new active asset
    currentLotSize = null; // Reset lot size before fetching
    stopAutoRefresh();
    previousOptionChainData = {};
    previousSpotPrice = 0;
    currentSpotPrice = 0;

    resetResultsUI(); // Clear strategy table & calculation results
    setLoadingStateForAssetChange(); // Set loading states for dependent UI parts
    // sendDebugAssetSelection(activeAsset); // Optional debug call

    try {
        // Fetch spot, expiries, analysis, news, AND asset details (lot size)
        const [spotResult, expiryResult, analysisResult, newsResult, detailsResult] = await Promise.allSettled([
            fetchNiftyPrice(activeAsset), // Fetches spot price
            fetchExpiries(activeAsset),   // Fetches expiry dates
            fetchAnalysis(activeAsset),   // Fetches stock analysis
            fetchNews(activeAsset),       // Fetches news
            fetchAPI(`/get_asset_details?asset=${encodeURIComponent(activeAsset)}`) // <--- FETCH LOT SIZE
        ]);

        // --- Process Lot Size Result ---
        if (detailsResult.status === 'fulfilled' && detailsResult.value?.lot_size) {
            currentLotSize = parseInt(detailsResult.value.lot_size, 10);
            if (!isNaN(currentLotSize) && currentLotSize > 0) {
                logger.info(`Successfully fetched lot size for ${activeAsset}: ${currentLotSize}`);
            } else {
                 logger.error(`Invalid lot size value received for ${activeAsset}: ${detailsResult.value.lot_size}`);
                 currentLotSize = null; // Treat invalid as null
            }
        } else {
            // Handle case where API failed or returned null lot size
            const reason = detailsResult.reason?.message || `Lot size missing in response for ${activeAsset}`;
            logger.error(`Failed to fetch or validate lot size for ${activeAsset}: ${reason}`);
            currentLotSize = null;
            // Optionally inform user if lot size is critical
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Warning: Could not determine lot size for ${activeAsset}. Strategy calculations may be affected.`);
             setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
        }
        // -------------------------------


        // Check essential data (Spot + Expiry are often most critical for chain)
        let hasCriticalError = spotResult.status === 'rejected' || expiryResult.status === 'rejected';

        // Log other non-critical failures
        if (analysisResult.status === 'rejected') logger.error(`Analysis fetch failed: ${analysisResult.reason?.message}`);
        if (newsResult.status === 'rejected') logger.error(`News fetch failed: ${newsResult.reason?.message}`);
        // Lot size failure already logged above

        if (!hasCriticalError) {
            logger.info(`Essential data loaded for ${activeAsset}. Starting auto-refresh.`);
            startAutoRefresh();
        } else {
            logger.error(`Failed loading essential market data (Spot/Expiry) for ${activeAsset}. Refresh NOT started.`);
            setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed loading market data for ${activeAsset}.`);
            // Ensure dependent UI shows error state correctly
            if (spotResult.status === 'rejected') setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error');
            if (expiryResult.status === 'rejected') {
                setElementState(SELECTORS.expiryDropdown, 'error', 'Error Expiries');
                setElementState(SELECTORS.optionChainTableBody, 'error', `Failed loading expiries.`);
            }
        }
    } catch (err) { // Catch unexpected errors during Promise.allSettled or processing
        logger.error(`Unexpected error in handleAssetChange for ${activeAsset}:`, err);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed loading data: ${err.message}`);
        stopAutoRefresh(); // Stop refresh on major error
        // Set multiple elements to error state
        setElementState(SELECTORS.expiryDropdown, 'error');
        setElementState(SELECTORS.optionChainTableBody, 'error');
        setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error');
        setElementState(SELECTORS.analysisResultContainer, 'error');
        setElementState(SELECTORS.newsResultContainer, 'error');
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
function handleOptionChainClick(event) {
    // Log the initial click target for debugging delegation
    logger.debug("[handleOptionChainClick] Delegated click fired! Target:", event.target);

    // Use closest() to find the relevant TD from the actual click target
    const priceCell = event.target.closest('td.clickable.price');
    if (!priceCell) {
        // logger.debug("[handleOptionChainClick] Click target was not inside a clickable price TD.");
        return; // Ignore clicks not within desired cells
    }
    logger.debug("[handleOptionChainClick] Found relevant priceCell via closest():", priceCell);

    const row = priceCell.closest('tr');
    if (!row || !row.dataset.strike) {
        logger.error("[handleOptionChainClick] Could not find parent row or row missing data-strike attribute.");
        return;
    }
    logger.debug("[handleOptionChainClick] Found row:", row);

    // --- Extract Data ---
    const strike = parseFloat(row.dataset.strike);
    const type = priceCell.classList.contains('call') ? 'CE' : 'PE';

    // Get price: Prefer data-price, fallback to textContent, default to 0
    const priceText = priceCell.dataset.price !== undefined ? priceCell.dataset.price : (priceCell.textContent || '0');
    const price = parseFloat(String(priceText).replace(/[₹,]/g, '')) || 0;

    // Find IV cell
    const ivSelector = priceCell.classList.contains('call') ? '.call.iv' : '.put.iv';
    const ivCell = row.querySelector(ivSelector);
    let iv = null;
    if (ivCell) {
        const ivRaw = ivCell.dataset.iv !== undefined ? ivCell.dataset.iv : ivCell.textContent;
        let ivNum = parseFloat(String(ivRaw).replace(/[%]/g, ''));
        if (!isNaN(ivNum)) {
             if (ivCell.dataset.iv !== undefined && ivCell.dataset.iv !== '') {
                  iv = ivNum; // Already decimal
             } else if (String(ivRaw).includes('%')) {
                  iv = ivNum / 100.0; // Convert % text to decimal
             } else if (ivRaw !== '-') {
                  iv = ivNum; // Assume decimal if no % and not '-'
                  logger.warn(`[handleOptionChainClick] Ambiguous IV source for ${type} ${strike}. Parsed as ${ivNum} from '${ivRaw}'. Assuming decimal.`);
             }
        }
    }

    logger.debug('[handleOptionChainClick] Extracted Data:', { strike, type, price, iv });

    // --- Validate Critical Data ---
     if (isNaN(strike) || strike <= 0) {
         logger.error("[handleOptionChainClick] Invalid strike price extracted:", row.dataset.strike); return;
     }
     if (isNaN(price) || price < 0) {
          logger.error("[handleOptionChainClick] Invalid price extracted:", priceText); return;
     }
     if (iv !== null && (isNaN(iv) || iv < 0)) {
         logger.error(`[handleOptionChainClick] Invalid IV extracted: ${iv}. Setting to null.`); iv = null;
     }

    // --- Log & Add ---
    logger.info(`[handleOptionChainClick] Attempting to add position: Strike=${strike}, Type=${type}, Price=${price}, IV=${iv === null ? 'N/A' : iv.toFixed(4)}`);
    addPosition(strike, type, price, iv); // Call addPosition (ensure it's the robust version)
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

    // --- Input Controls ---
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
    if (assetDropdown) {
        if (typeof handleAssetChange === 'function') {
            assetDropdown.addEventListener("change", handleAssetChange);
        } else { logger.error("handleAssetChange not defined for listener!"); }
    } else { logger.warn("Asset dropdown not found for listener."); }

    const expiryDropdown = document.querySelector(SELECTORS.expiryDropdown);
    if (expiryDropdown) {
        if (typeof handleExpiryChange === 'function') {
            expiryDropdown.addEventListener("change", handleExpiryChange);
        } else { logger.error("handleExpiryChange not defined for listener!"); }
    } else { logger.warn("Expiry dropdown not found for listener."); }

    // --- Action Buttons ---
    const updateButton = document.querySelector(SELECTORS.updateChartButton);
    if (updateButton) {
        if (typeof fetchPayoffChart === 'function') {
            updateButton.addEventListener("click", fetchPayoffChart);
        } else { logger.error("fetchPayoffChart not defined for listener!"); }
    } else { logger.warn("Update chart button not found for listener."); }

    const clearButton = document.querySelector(SELECTORS.clearPositionsButton);
    if (clearButton) {
        if (typeof clearAllPositions === 'function') {
            clearButton.addEventListener("click", clearAllPositions);
        } else { logger.error("clearAllPositions not defined for listener!"); }
    } else { logger.warn("Clear positions button not found for listener."); }

    // --- Strategy Table Interaction (Event Delegation Recommended Here Too) ---
    const strategyTable = document.getElementById('strategyTable'); // Use ID for direct access
    if (strategyTable) {
        logger.debug("[DEBUG] Attaching listeners to #strategyTable:", strategyTable);
        // Use event delegation for clicks (remove, toggle)
        if (typeof handleStrategyTableClick === 'function') {
            strategyTable.addEventListener('click', handleStrategyTableClick);
        } else { logger.error("handleStrategyTableClick not defined!"); }

        // Use event delegation for input changes (lots)
        if (typeof handleStrategyTableChange === 'function') {
            strategyTable.addEventListener('input', handleStrategyTableChange);
        } else { logger.error("handleStrategyTableChange not defined!"); }
    } else {
        logger.warn("Strategy table (#strategyTable) not found for listeners.");
    }

    // --- Option Chain Table Interaction (EVENT DELEGATION FIX) ---
    // Listen on the TABLE element itself, which is stable
    const optionChainTable = document.getElementById('optionChainTable'); // Use ID for direct access
    if (optionChainTable) {
        logger.debug("[DEBUG] Attaching click listener to #optionChainTable (delegation):", optionChainTable);
        if (typeof handleOptionChainClick === 'function') {
            // Attach listener to the TABLE, not the TBODY
            optionChainTable.addEventListener('click', handleOptionChainClick);
        } else {
            logger.error("handleOptionChainClick not defined!");
        }
    } else {
        logger.warn("Option chain table (#optionChainTable) not found for listener.");
    }

    logger.info("Event listener setup complete.");
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

function onAssetSelected(assetName) {
    fetch('/update_selected_asset/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset: assetName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Optionally, fetch new option chain or update UI
        } else {
            alert("Failed to update asset: " + data.message);
        }
    })
    .catch(err => {
        alert("Network error updating asset: " + err);
    });
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