// ===============================================================
// Configuration & Constants
// ===============================================================
// const API_BASE = "http://localhost:8000"; // For Local Hosting
const API_BASE = "https://option-strategy-website.onrender.com"; // For Production

const SELECTORS = {
    assetDropdown: "#asset",
    expiryDropdown: "#expiry",
    spotPriceDisplay: "#niftyPrice",
    optionChainTableBody: "#optionChainTable tbody",
    strategyTableBody: "#strategyTable tbody",
    updateChartButton: "#updateChartBtn",
    clearPositionsButton: "#clearStrategyBtn",
    payoffChartContainer: "#payoffChartContainer",
    analysisResultContainer: "#analysisResult",
    maxProfitDisplay: "#maxProfit",
    maxLossDisplay: "#maxLoss",
    breakevenDisplay: "#breakeven",
    rewardToRiskDisplay: "#rewardToRisk",
    netPremiumDisplay: "#netPremium",
    costBreakdownContainer: "#costBreakdownContainer",
    costBreakdownList: "#costBreakdownList",
    taxInfoContainer: "#taxInfo",
    greeksTableBody: "#greeksTable tbody",
    greeksTable: "#greeksTable",
    globalErrorDisplay: "#globalError",
};

const logger = { /* ... logger remains the same ... */
    debug: (...args) => console.debug('[DEBUG]', ...args),
    info: (...args) => console.log('[INFO]', ...args),
    warn: (...args) => console.warn('[WARN]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
};

// ===============================================================
// Global State (Keep as is)
// ===============================================================
let currentSpotPrice = 0;
let strategyPositions = [];
let activeAsset = null;

// ===============================================================
// Utility Functions (Minor improvements)
// ===============================================================

/** Safely formats a number or returns a fallback string */
function formatNumber(value, decimals = 2, fallback = "N/A") {
    if (value === null || typeof value === 'undefined') { return fallback; }
    if (typeof value === 'string') {
        // Handle specific string representations from backend (like infinity)
        if (["∞", "Infinity"].includes(value)) return "∞";
        if (["-∞", "-Infinity"].includes(value)) return "-∞";
        if (["N/A", "Undefined", "Loss"].includes(value)) return value; // Pass through specific statuses
        // Attempt to convert other strings
    }
    const num = Number(value);
    if (!isNaN(num)) {
        if (num === Infinity) return "∞";
        if (num === -Infinity) return "-∞";
        return num.toLocaleString(undefined, { // Use locale string for commas
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    return fallback;
}

/** Safely formats currency */
function formatCurrency(value, decimals = 2, fallback = "N/A", prefix = "₹") {
     // Handle specific non-numeric strings first
    if (typeof value === 'string' && ["∞", "-∞", "N/A", "Undefined", "Loss"].includes(value)) {
        return value;
    }
    const formattedNumber = formatNumber(value, decimals, null); // Use null fallback
    if (formattedNumber !== null && !["∞", "-∞"].includes(formattedNumber)) { // Don't prefix infinity
        return `${prefix}${formattedNumber}`;
    }
    return formattedNumber === null ? fallback : formattedNumber; // Return ∞/-∞ as is, or fallback
}

/** Helper to display formatted metric/value in a UI element */
function displayMetric(value, targetElementSelector, prefix = '', suffix = '', decimals = 2, isCurrency = false) {
     const element = document.querySelector(targetElementSelector);
     if (!element) {
        logger.warn(`displayMetric: Element not found for selector "${targetElementSelector}"`);
        return;
     }
     const formattedValue = isCurrency
        ? formatCurrency(value, decimals, "N/A", "") // Let prefix handle ₹ if needed
        : formatNumber(value, decimals, "N/A");

     // Handle cases where value itself is N/A, ∞ etc.
     if (["N/A", "∞", "-∞", "Undefined", "Loss"].includes(formattedValue)) {
         element.textContent = `${prefix}${formattedValue}${suffix}`;
     } else {
         element.textContent = `${prefix}${isCurrency ? '₹' : ''}${formattedValue}${suffix}`;
     }
}

/** Sets the loading/error/content state for an element. */
function setElementState(selector, state, message = 'Loading...') {
    const element = document.querySelector(selector);
    if (!element) { logger.warn(`setElementState: Element not found for "${selector}"`); return; }
    const isSelect = element.tagName === 'SELECT';
    const isButton = element.tagName === 'BUTTON';
    const isTbody = element.tagName === 'TBODY';
    const isContainer = element.tagName === 'DIV' || element.tagName === 'SECTION' || element.id === SELECTORS.payoffChartContainer.substring(1);
    const isSpan = element.tagName === 'SPAN'; // e.g., spot price part

    element.classList.remove('loading', 'error', 'loaded');
    if (isSelect || isButton) element.disabled = false;
    element.style.display = ''; // Default display
    if (element.classList.contains('error-message')) element.style.color = ''; // Reset error color

    switch (state) {
        case 'loading':
            element.classList.add('loading');
            if (isSelect) element.innerHTML = `<option>${message}</option>`;
            else if (isTbody) element.innerHTML = `<tr><td colspan="7" class="loading-text">${message}</td></tr>`; // Adjust colspan if needed
            else if (!isButton && !isSpan) element.textContent = message;
            else if(isSpan) element.textContent = message; // Set loading text for spans too
            if (isSelect || isButton) element.disabled = true;
            break;
        case 'error':
            element.classList.add('error');
            const displayMessage = `Error: ${message}`;
            if (isSelect) { element.innerHTML = `<option>${displayMessage}</option>`; element.disabled = true; }
            else if (isTbody) { element.innerHTML = `<tr><td colspan="7" class="error-message">${displayMessage}</td></tr>`; }
            else if (isContainer) { element.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`; } // Display error within container
            else if (isSpan) { element.textContent = 'Error'; element.classList.add('error-message'); } // Concise error for spans
            else { element.textContent = displayMessage; element.classList.add('error-message');}
            break;
        case 'content':
            element.classList.add('loaded');
            if(element.classList.contains('error-message')) element.style.color = '';
            // Calling function will set the actual content after this call
            break;
         case 'hidden':
             element.style.display = 'none';
            break;
    }
}

/** Populates a dropdown select element. */
function populateDropdown(selector, items, placeholder = "-- Select --", defaultSelectFirst = false) {
    const selectElement = document.querySelector(selector);
    if (!selectElement) return;
    const currentValue = selectElement.value;
    selectElement.innerHTML = "";
    if (!items || items.length === 0) {
        selectElement.innerHTML = `<option value="">-- No options available --</option>`;
        selectElement.disabled = true;
        return;
    }
    // Optional placeholder
    if (placeholder) {
        selectElement.innerHTML = `<option value="">${placeholder}</option>`;
    }
    items.forEach(item => {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });
    // Try to restore previous selection or select first item
    if (items.includes(currentValue)) {
        selectElement.value = currentValue;
    } else if (defaultSelectFirst && items.length > 0) {
         selectElement.value = items[0];
    }
    selectElement.disabled = false;
}

/** Fetches data from the API with error handling. */
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    options.headers = { ...defaultHeaders, ...options.headers };
    const method = options.method || 'GET';
    logger.debug(`fetchAPI Request: ${method} ${url}`, options.body ? JSON.parse(options.body) : '');
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Hide previous global errors

    try {
        const response = await fetch(url, options);
        let responseData = null;
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
             responseData = await response.json(); // Attempt to parse JSON
        } else if (response.status !== 204) { // Handle non-JSON, non-empty responses if necessary
             // responseData = await response.text(); // Or handle as text
             logger.warn(`Received non-JSON response from ${method} ${url} (Status: ${response.status})`);
        }

        logger.debug(`fetchAPI Response Status: ${response.status} for ${method} ${url}`);

        if (!response.ok) {
            const errorMessage = responseData?.detail || response.statusText || `HTTP error ${response.status}`;
            logger.error(`API Error (${method} ${url} - ${response.status}): ${errorMessage}`, responseData);
            throw new Error(errorMessage); // Throw standardized error
        }

         logger.debug(`fetchAPI Response Data:`, responseData);
        return responseData; // Return parsed data (or null for 204)

    } catch (error) {
        logger.error(`Network/Fetch Error (${method} ${url}):`, error);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Network Error: ${error.message || 'Failed to fetch'}`);
        throw error; // Re-throw for specific UI handling if needed
    }
}

// ===============================================================
// Initialization & Event Listeners
// ===============================================================

document.addEventListener("DOMContentLoaded", () => {
    logger.info("DOM Ready. Initializing...");
    initializePage();
    setupEventListeners();
    loadMarkdownParser(); // Load markdown parser early
});

async function initializePage() {
    logger.info("Initializing page data...");
    resetResultsUI(); // Start with clean results area
    await loadAssets(); // This will trigger subsequent data loads
}

function setupEventListeners() {
    logger.info("Setting up event listeners...");
    document.querySelector(SELECTORS.assetDropdown)?.addEventListener("change", handleAssetChange);
    document.querySelector(SELECTORS.expiryDropdown)?.addEventListener("change", handleExpiryChange);
    document.querySelector(SELECTORS.updateChartButton)?.addEventListener("click", fetchPayoffChart);
    // Use the corrected selector for the clear button
    document.querySelector(SELECTORS.clearPositionsButton)?.addEventListener("click", clearAllPositions);

    // Auto-refresh listeners (keep commented if not used)
    // document.querySelector(SELECTORS.autoRefreshCheckbox)?.addEventListener('change', setupAutoRefresh);
    // document.querySelector(SELECTORS.refreshIntervalSelect)?.addEventListener('change', setupAutoRefresh);

    // Add listener for strategy table interaction (event delegation)
    const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (strategyTableBody) {
        strategyTableBody.addEventListener('change', handleStrategyTableChange);
        strategyTableBody.addEventListener('click', handleStrategyTableClick);
    }

    // Scroll persistence (keep as is)
    // window.addEventListener("scroll", () => localStorage.setItem("scrollPosition", window.scrollY));
    // window.addEventListener("load", () => {
    //     const savedScrollTop = localStorage.getItem("scrollPosition");
    //     if (savedScrollTop !== null) window.scrollTo(0, parseInt(savedScrollTop));
    // });
}

function loadMarkdownParser() {
    if (typeof marked === 'undefined') {
        try {
            const script = document.createElement("script");
            script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
            script.onload = () => logger.info("Markdown parser (marked.js) loaded.");
            script.onerror = () => logger.error("Failed to load Markdown parser (marked.js). Analysis rendering may fail.");
            document.head.appendChild(script);
        } catch (e) {
             logger.error("Error creating script tag for marked.js", e);
        }
    }
}

// ===============================================================
// Event Handlers & Data Fetching Logic
// ===============================================================

/** Handles asset dropdown change */
async function handleAssetChange() {
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
    const asset = assetDropdown?.value;
    activeAsset = asset; // Update global state

    if (!asset) {
        // Reset dependent UI if no asset selected
        populateDropdown(SELECTORS.expiryDropdown, [], "-- Select Asset First --");
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Asset</td></tr>`;
        document.querySelector(SELECTORS.analysisResultContainer).innerHTML = '';
        document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
        resetResultsUI();
        return;
    }

    logger.info(`Asset changed to: ${asset}. Fetching data...`);
    setElementState(SELECTORS.expiryDropdown, 'loading');
    setElementState(SELECTORS.optionChainTableBody, 'loading');
    setElementState(SELECTORS.analysisResultContainer, 'loading');
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...');
    resetResultsUI(); // Clear previous results on asset change

    // --- Call Debug Endpoint for Flawed Background Task (Corrected POST with body) ---
    try {
        // Use the correct endpoint URL and send asset in the body
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST',
             body: JSON.stringify({ asset: asset }) // Send asset in JSON body
        });
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        // Log error but continue page load
        logger.error("Failed to send debug asset selection:", debugErr.message);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Debug Sync Failed: ${debugErr.message}`);
    }
    // --- End Debug Call ---

    try {
        // Fetch core data in parallel
        const [spotResult, expiryResult, analysisResult] = await Promise.allSettled([
            fetchNiftyPrice(asset), // Changed name
            fetchExpiries(asset),
            fetchAnalysis(asset)
        ]);

        // Log any errors from parallel fetches
        if (spotResult.status === 'rejected') logger.error(`Error fetching spot price: ${spotResult.reason}`);
        if (expiryResult.status === 'rejected') logger.error(`Error fetching expiries: ${expiryResult.reason}`);
        if (analysisResult.status === 'rejected') logger.error(`Error fetching analysis: ${analysisResult.reason}`);

        // fetchOptionChain will be called by fetchExpiries if successful
    } catch (err) {
        // This catch might not be reached if Promise.allSettled is used, but kept for safety
        logger.error(`Error fetching initial data for ${asset}:`, err);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load page data for ${asset}.`);
    }
}

/** Handles expiry dropdown change */
async function handleExpiryChange() {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) {
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Expiry</td></tr>`;
        return;
    }
    logger.info(`Expiry changed to: ${expiry}. Fetching option chain...`);
    await fetchOptionChain(true); // Fetch new chain and scroll to ATM
}

/** Fetches and populates the asset dropdown */
async function loadAssets() {
    setElementState(SELECTORS.assetDropdown, 'loading');
    try {
        const data = await fetchAPI("/get_assets");
        const assets = data?.assets || [];
        populateDropdown(SELECTORS.assetDropdown, assets, "-- Select Asset --");
        setElementState(SELECTORS.assetDropdown, 'content');

        // --- Default to NIFTY ---
        const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
        let defaultAsset = null;
        // Check if NIFTY is present in the fetched list
        if (assets.includes("NIFTY")) {
            defaultAsset = "NIFTY";
        } else if (assets.length > 0) {
            // Fallback to the first available asset if NIFTY is not present
            defaultAsset = assets[0];
             logger.warn(`"NIFTY" not found in assets, defaulting to first asset: ${defaultAsset}`);
        }

        if (defaultAsset && assetDropdown) {
            assetDropdown.value = defaultAsset; // Set the value in the dropdown
            logger.info(`Defaulting asset selection to: ${defaultAsset}`);
            // IMPORTANT: Explicitly call the handler AFTER setting the value
            // to load data for the default asset. Use await.
            await handleAssetChange();
        } else {
            // Handle case where no assets are loaded at all
             logger.warn("No assets found in dropdown after fetch. Cannot set default.");
             // Call handler anyway to potentially clear dependent fields
             await handleAssetChange();
        }
        // --- End Default to NIFTY ---

    } catch (error) {
        setElementState(SELECTORS.assetDropdown, 'error', `Assets Error: ${error.message}`);
        // Set dependent elements to error state if assets fail
        setElementState(SELECTORS.expiryDropdown, 'error', 'Asset load failed');
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Asset load failed');
    }
}

/** Fetches stock analysis for the selected asset */
async function fetchAnalysis(asset) {
    if (!asset) return;
    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Fetching analysis...');
    try {
        // Wait briefly if marked isn't loaded yet
        for (let i=0; i<5 && typeof marked === 'undefined'; i++) {
            await new Promise(resolve => setTimeout(resolve, 300));
        }

        if (typeof marked === 'undefined') {
            logger.error("Markdown parser (marked.js) is not loaded. Cannot render analysis.");
            throw new Error("Markdown parser failed to load.");
        }

        const data = await fetchAPI("/get_stock_analysis", {
            method: "POST", body: JSON.stringify({ asset })
        });
        const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
        if (analysisContainer) {
            // Sanitize potentially harmful HTML before parsing markdown
            // (Using a library like DOMPurify is recommended for production)
            // Simple basic 'sanitization' (replace potential script tags) for now:
            const rawAnalysis = data?.analysis || "*No analysis content received.*";
            const potentiallySanitized = rawAnalysis.replace(/<script.*?>.*?<\/script>/gi, ''); // Basic removal
            analysisContainer.innerHTML = marked.parse(potentiallySanitized);
            setElementState(SELECTORS.analysisResultContainer, 'content');
        }
    } catch (error) {
        setElementState(SELECTORS.analysisResultContainer, 'error', `Analysis Error: ${error.message}`);
    }
}


/** Fetches and populates expiry dates */
async function fetchExpiries(asset) {
    if (!asset) return;
    setElementState(SELECTORS.expiryDropdown, 'loading');
    try {
        const data = await fetchAPI(`/expiry_dates?asset=${encodeURIComponent(asset)}`);
        const expiries = data?.expiry_dates || [];
        // Select first expiry by default when populating
        populateDropdown(SELECTORS.expiryDropdown, expiries, "-- Select Expiry --", true); // Pass true for defaultSelectFirst
        setElementState(SELECTORS.expiryDropdown, 'content');

        // *** CHANGE HERE: Trigger chain load for the selected expiry ***
        const selectedExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
        if (selectedExpiry) {
            // Explicitly call the handler or fetch chain directly
            // Calling handleExpiryChange is safer if it does more than just fetch chain
             await handleExpiryChange(); // Call the existing handler
            // OR await fetchOptionChain(true); // Fetch directly if handler only does this
        } else {
            // Handle case where no expiries are returned
            setElementState(SELECTORS.optionChainTableBody, 'content');
            document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">No expiry dates found for ${asset}</td></tr>`;
        }
        // *** END CHANGE ***

    } catch (error) {
        setElementState(SELECTORS.expiryDropdown, 'error', `Expiries Error: ${error.message}`);
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Could not load option chain (expiry fetch failed)');
    }
}


/** Fetches and displays the spot price */
async function fetchNiftyPrice(asset) { // Renamed function
    if (!asset) return;
    const priceElement = document.querySelector(SELECTORS.spotPriceDisplay);
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...'); // Use state function
    try {
        const data = await fetchAPI(`/get_spot_price?asset=${encodeURIComponent(asset)}`);
        currentSpotPrice = data?.spot_price ?? 0; // Store fetched price
        if (priceElement) {
            priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}`;
            setElementState(SELECTORS.spotPriceDisplay, 'content');
        }
    } catch (error) {
         currentSpotPrice = 0; // Reset on error
         setElementState(SELECTORS.spotPriceDisplay, 'error', `Spot Price Error`); // Use state function
         logger.error(`Spot Price Error (${asset}):`, error.message);
    }
}

/** Fetches and displays the option chain */
async function fetchOptionChain(scrollToATM = false) {
    const asset = document.querySelector(SELECTORS.assetDropdown)?.value;
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    const tableBody = document.querySelector(SELECTORS.optionChainTableBody);

    if (!asset || !expiry) {
         if(tableBody) tableBody.innerHTML = `<tr><td colspan="7">Select Asset and Expiry</td></tr>`;
         setElementState(SELECTORS.optionChainTableBody, 'content'); // Set state
        return;
    }
    setElementState(SELECTORS.optionChainTableBody, 'loading');

    try {
        // Ensure spot price is available for ATM calculation before fetching chain
        if(currentSpotPrice <= 0) {
            logger.info("Spot price is 0, fetching before option chain...");
            await fetchNiftyPrice(asset); // Wait for spot price fetch
        }
        // Check again if still 0 after attempting fetch
        if(currentSpotPrice <= 0 && scrollToATM) {
            logger.warn("Spot price unavailable after fetch, cannot calculate ATM strike accurately.");
            scrollToATM = false; // Disable ATM scrolling if spot price is missing
        }

        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);

        if (!data?.option_chain || Object.keys(data.option_chain).length === 0) {
            if(tableBody) tableBody.innerHTML = `<tr><td colspan="7">No option chain data available for ${asset} on ${expiry}</td></tr>`;
            setElementState(SELECTORS.optionChainTableBody, 'content');
            return;
        }

        const strikes = Object.keys(data.option_chain).map(Number).sort((a, b) => a - b);
        const atmStrike = currentSpotPrice > 0 ? findATMStrike(strikes, currentSpotPrice) : null;
        const newTbody = document.createElement("tbody");

        strikes.forEach(strike => {
            const optionData = data.option_chain[strike];
            const tr = document.createElement("tr");
             tr.dataset.strike = strike; // Add strike to dataset for easier access

            // Add ATM class if applicable
            if (atmStrike !== null && Math.abs(strike - atmStrike) < 0.01) { // Float comparison
                tr.classList.add("atm-strike");
            }

            const call = optionData.call || {}; // Default to empty object if null/undefined
            const put = optionData.put || {};   // Default to empty object

            // Add data attributes to the price cells for easier access in addPosition
            // Use data-* attributes for strike, type, and price
            tr.innerHTML = `
                <td class="call clickable" data-type="CE" data-price="${call.last_price || 0}">${formatNumber(call.last_price, 2, '-')}</td>
                <td class="call oi">${formatNumber(call.open_interest, 0, '-')}</td>
                <td class="call iv">${formatNumber(call.implied_volatility, 2, '-')} %</td>
                <td class="strike">${strike}</td>
                <td class="put iv">${formatNumber(put.implied_volatility, 2, '-')} %</td>
                <td class="put oi">${formatNumber(put.open_interest, 0, '-')}</td>
                <td class="put clickable" data-type="PE" data-price="${put.last_price || 0}">${formatNumber(put.last_price, 2, '-')}</td>
            `;
            newTbody.appendChild(tr);
        });

        // Replace entire tbody for efficiency and event listener management
        if (tableBody) tableBody.parentNode.replaceChild(newTbody, tableBody);
        setElementState(SELECTORS.optionChainTableBody, 'content'); // Set state AFTER replacement

        // Add event listeners to the new tbody (event delegation)
        newTbody.addEventListener('click', handleOptionChainClick);

        // Scroll after new tbody is in the DOM and rendered
        if (scrollToATM && atmStrike !== null) {
            // Use setTimeout to ensure rendering is complete before scrolling
            setTimeout(() => {
                const atmRow = newTbody.querySelector(".atm-strike");
                if (atmRow) {
                    atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
                } else {
                    logger.warn("ATM strike row not found after rendering for scrolling.");
                }
            }, 100); // Small delay
        }

    } catch (error) {
        setElementState(SELECTORS.optionChainTableBody, 'error', `Chain Error: ${error.message}`);
    }
}

// ===============================================================
// Event Delegation Handlers
// ===============================================================

/** Handles clicks within the option chain table body */
function handleOptionChainClick(event) {
    const targetCell = event.target.closest('td.clickable'); // Find the clicked cell or its clickable parent
    if (!targetCell) return; // Exit if click wasn't on a clickable cell

    const row = targetCell.closest('tr');
    if (!row || !row.dataset.strike) return; // Exit if row or strike data is missing

    const strike = parseFloat(row.dataset.strike);
    const type = targetCell.dataset.type; // 'CE' or 'PE'
    const price = parseFloat(targetCell.dataset.price); // Price from the clicked cell

    if (!isNaN(strike) && type && !isNaN(price)) {
         addPosition(strike, type, price); // Pass the fetched price
    } else {
        logger.warn('Could not add position - invalid data from clicked cell', { strike, type, price });
        alert('Could not retrieve option details. Please try again.');
    }
}

/** Handles clicks within the strategy table body (for remove buttons) */
function handleStrategyTableClick(event) {
     const removeButton = event.target.closest('button.remove-btn');
     if (removeButton && removeButton.dataset.index) {
         const index = parseInt(removeButton.dataset.index, 10);
         if (!isNaN(index)) {
             removePosition(index);
         }
     }
}

/** Handles changes within the strategy table body (for lots input) */
function handleStrategyTableChange(event) {
    const lotsInput = event.target.closest('input.lots-input');
     if (lotsInput && lotsInput.dataset.index) {
        const index = parseInt(lotsInput.dataset.index, 10);
        if (!isNaN(index)) {
            updateLots(index, lotsInput.value);
        }
    }
}


// ===============================================================
// Strategy Management UI Logic
// ===============================================================

/** Adds a position (called by handleOptionChainClick) */
function addPosition(strike, type, price) { // Added price parameter
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) { alert("Please select an expiry date first."); return; }

    // Price is now passed directly from the clicked cell data attribute
    const lastPrice = (typeof price === 'number' && !isNaN(price)) ? price : 0;

    const newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type, // 'CE' or 'PE'
        lots: 1,           // Default to 1 lot (Buy)
        last_price: lastPrice, // Use the price from the clicked cell
    };
    strategyPositions.push(newPosition);
    updateStrategyTable();
    logger.info("Added position:", newPosition);
     // Optional: Automatically update chart on adding? Might be too slow/frequent.
     // fetchPayoffChart();
}

/** Updates the strategy table in the UI */
function updateStrategyTable() {
    const tableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (!tableBody) return;
    tableBody.innerHTML = ""; // Clear existing rows

    if (strategyPositions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7">No positions added. Click option prices in the chain to add.</td></tr>';
        resetResultsUI();
        return;
    }

    strategyPositions.forEach((pos, index) => {
        const isLong = pos.lots >= 0;
        const positionType = isLong ? "BUY" : "SELL";
        const positionClass = isLong ? "long-position" : "short-position";
        const buttonClass = isLong ? "button-buy" : "button-sell"; // Class for toggle button styling

        const row = document.createElement("tr");
        row.className = positionClass;
        row.dataset.index = index; // Add index to row for easier updates potentially

        // Match thead order: Type, Strike, Expiry, Lots, Action, Premium, Remove
        row.innerHTML = `
            <td>${pos.option_type}</td>
            <td>${pos.strike_price}</td>
            <td>${pos.expiry_date}</td>
            <td>
                <input type="number" value="${pos.lots}"
                    data-index="${index}"
                    min="-100" max="100" step="1"
                    class="lots-input"
                    aria-label="Lots for position ${index+1}">
            </td>
            <td>
                <!-- ADDED Buy/Sell Toggle Button -->
                <button class="toggle-buy-sell ${buttonClass}" data-index="${index}" title="Click to switch between Buy and Sell">
                    ${positionType}
                </button>
            </td>
            <td>${formatCurrency(pos.last_price, 2)}</td>
            <td>
                <button class="remove-btn" data-index="${index}" aria-label="Remove position ${index+1}">
                    ×
                </button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

/** Updates the number of lots for a position (called by delegation) */
function updateLots(index, value) {
    if (index < 0 || index >= strategyPositions.length) return;
    const newLots = parseInt(value, 10);

    if (isNaN(newLots)) {
         alert("Please enter a valid integer for lots.");
         const inputElement = document.querySelector(`${SELECTORS.strategyTableBody} input.lots-input[data-index="${index}"]`);
         if(inputElement) inputElement.value = strategyPositions[index].lots;
         return;
    }

    if (newLots === 0) {
        logger.info(`Lots set to 0 for index ${index}, removing position.`);
        removePosition(index);
    } else {
        const previousLots = strategyPositions[index].lots;
        strategyPositions[index].lots = newLots;

        // Update Buy/Sell button and row class if the sign changed
        const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
        const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);

        if (toggleButton && row) {
             const isNowLong = newLots >= 0;
             const positionType = isNowLong ? "BUY" : "SELL";
             const buttonClass = isNowLong ? "button-buy" : "button-sell";

            toggleButton.textContent = positionType;
            toggleButton.classList.remove("button-buy", "button-sell");
            toggleButton.classList.add(buttonClass);
            row.className = isNowLong ? "long-position" : "short-position"; // Update row class
        } else {
             updateStrategyTable(); // Fallback if elements not found
        }

        logger.info(`Updated lots for index ${index} from ${previousLots} to ${newLots}`);
    }
}


function toggleBuySell(index) {
    if (index < 0 || index >= strategyPositions.length) return;

    // Flip the sign of lots
    const previousLots = strategyPositions[index].lots;
    strategyPositions[index].lots *= -1;
    const newLots = strategyPositions[index].lots;

    // If lots were 0, toggling makes it -0 which is 0. Default to buying 1 lot? Or selling -1? Let's default to BUY 1.
     if (newLots === 0 && previousLots === 0) {
         strategyPositions[index].lots = 1;
         logger.info(`Lots were 0 for index ${index}, setting to 1 (BUY).`);
     } else if (newLots === 0) { // If toggling resulted in zero (e.g., from 0), set to opposite of previous non-zero or default
         // This case should ideally not happen if updateLots removes 0-lot positions
         // But as a safeguard, set to Buy 1
         strategyPositions[index].lots = 1;
         logger.warn(`Toggling resulted in 0 lots for index ${index}, setting to 1 (BUY).`);
     }


    logger.info(`Toggled Buy/Sell for index ${index}. New lots: ${strategyPositions[index].lots}`);

    // Update the specific row UI instead of full table refresh
    const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
    const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);
    const lotsInput = row?.querySelector(`input.lots-input[data-index="${index}"]`);

    if (row && toggleButton && lotsInput) {
        const isLong = strategyPositions[index].lots >= 0;
        const positionType = isLong ? "BUY" : "SELL";
        const buttonClass = isLong ? "button-buy" : "button-sell";

        toggleButton.textContent = positionType;
        toggleButton.classList.remove("button-buy", "button-sell");
        toggleButton.classList.add(buttonClass);
        row.className = isLong ? "long-position" : "short-position"; // Update row class
        lotsInput.value = strategyPositions[index].lots; // Update the number input field as well
    } else {
        updateStrategyTable(); // Fallback to full refresh if specific elements not found
    }
     // Optional: Trigger chart update immediately after toggle?
     // fetchPayoffChart();
}


function handleStrategyTableClick(event) {
     // Check for Remove button click
     const removeButton = event.target.closest('button.remove-btn');
     if (removeButton?.dataset.index) {
         const index = parseInt(removeButton.dataset.index, 10);
         if (!isNaN(index)) { removePosition(index); }
         return; // Stop further processing if remove button was clicked
     }

     // Check for Toggle Buy/Sell button click
     const toggleButton = event.target.closest('button.toggle-buy-sell');
     if (toggleButton?.dataset.index) {
          const index = parseInt(toggleButton.dataset.index, 10);
         if (!isNaN(index)) { toggleBuySell(index); }
         return; // Stop further processing if toggle button was clicked
     }
}

/** Removes a position from the strategy (called by delegation) */
function removePosition(index) {
     if (index < 0 || index >= strategyPositions.length) return;
    strategyPositions.splice(index, 1);
    updateStrategyTable(); // Update table UI
    logger.info("Removed position at index", index);
    // Trigger chart update only if there are remaining positions
    if (strategyPositions.length > 0) {
        fetchPayoffChart(); // Update results after removal
    } else {
        resetResultsUI(); // Clear results if strategy is now empty
    }
}

/** Clears all positions and resets UI */
function clearAllPositions() {
    logger.info("Clearing all positions...");
    strategyPositions = []; // Reset the local state array
    updateStrategyTable(); // Clear the strategy UI table
    resetResultsUI();      // Clear the chart and metrics UI
    logger.info("Strategy cleared.");
}

/** Resets the chart and results UI to initial state */
/** Resets the chart and results UI to initial state */
function resetResultsUI() {
     const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
     if (chartContainer) {
         try {
             // Clear any existing Plotly chart instance from the div
             if (typeof Plotly !== 'undefined') {
                 Plotly.purge(chartContainer);
             }
         } catch (e) {
             logger.error("Error purging Plotly chart:", e);
         } finally {
            // Restore placeholder text regardless of purge success/failure
            chartContainer.innerHTML = '<div class="placeholder-text">Add positions and click "Update" to see the payoff chart.</div>';
            setElementState(SELECTORS.payoffChartContainer, 'content'); // Reset state
         }
     }

     // Reset other elements
     document.querySelector(SELECTORS.taxInfoContainer).innerHTML = "";
     const greeksTable = document.querySelector(SELECTORS.greeksTable);
     if (greeksTable) {
         const greekBody = greeksTable.querySelector('tbody'); if(greekBody) greekBody.innerHTML = "";
         const greekFoot = greeksTable.querySelector('tfoot'); if(greekFoot) greekFoot.innerHTML = "";
     }
     document.querySelector(SELECTORS.costBreakdownList).innerHTML = "";
     setElementState(SELECTORS.costBreakdownContainer, 'hidden');

     // Reset metrics display to "N/A"
     displayMetric("N/A", SELECTORS.maxProfitDisplay, "Max Profit: ");
     displayMetric("N/A", SELECTORS.maxLossDisplay, "Max Loss: ");
     displayMetric("N/A", SELECTORS.breakevenDisplay, "Breakeven: ");
     displayMetric("N/A", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
     displayMetric("N/A", SELECTORS.netPremiumDisplay, "Net Premium: ");

     // Reset loading/error states for these sections too
     setElementState(SELECTORS.taxInfoContainer, 'content'); // Set back to content (empty)
     setElementState(SELECTORS.greeksTable, 'content');     // Set back to content (empty)
}


// ===============================================================
// Payoff Chart & Results Logic
// ===============================================================

/** Fetches calculation results and displays the interactive payoff chart using Plotly.js */
async function fetchPayoffChart() {
    const asset = document.querySelector(SELECTORS.assetDropdown)?.value;
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    const updateButton = document.querySelector(SELECTORS.updateChartButton);

    // --- Input Validation (Keep as is) ---
    if (!asset) { alert("Please select an asset."); return; }
    if (strategyPositions.length === 0) { resetResultsUI(); alert("Please add positions to the strategy first."); return; }
    if (!chartContainer) { logger.error("Payoff chart container element not found."); return; }

    // --- Set Loading States (Keep as is) ---
    setElementState(SELECTORS.payoffChartContainer, 'loading', 'Generating results...');
    setElementState(SELECTORS.taxInfoContainer, 'loading', 'Calculating...');
    setElementState(SELECTORS.greeksTable, 'loading', 'Calculating...');
    setElementState(SELECTORS.costBreakdownContainer, 'hidden');
    displayMetric("...", SELECTORS.maxProfitDisplay, "Max Profit: ");
    displayMetric("...", SELECTORS.maxLossDisplay, "Max Loss: ");
    displayMetric("...", SELECTORS.breakevenDisplay, "Breakeven: ");
    displayMetric("...", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
    displayMetric("...", SELECTORS.netPremiumDisplay, "Net Premium: ");
    if (updateButton) updateButton.disabled = true;

    // --- Prepare Request Data (Keep as is) ---
    const requestData = {
        asset: asset,
        strategy: strategyPositions.map(pos => ({
            option_type: pos.option_type,
            strike_price: String(pos.strike_price),
            option_price: String(pos.last_price),
            expiry_date: pos.expiry_date,
            lots: String(Math.abs(pos.lots)),
            tr_type: pos.lots >= 0 ? "b" : "s",
        }))
    };
    logger.debug("Sending request to /get_payoff_chart:", requestData);

    // --- API Call and Response Handling ---
    try {
        // Fetch data from the backend
        const data = await fetchAPI('/get_payoff_chart', {
            method: 'POST', body: JSON.stringify(requestData)
        });
        logger.debug("Received response from /get_payoff_chart:", data);

        // *** UPDATED ERROR CHECKING ***
        // 1. Check if data object itself exists
        if (!data) {
            throw new Error("Received empty response from server.");
        }
        // 2. Check the success flag explicitly
        if (data.success === false) {
            // If success is false, use the detail message if provided, otherwise a generic failure message
            throw new Error(data.detail || data.message || "Server indicated calculation failed without details.");
        }
        // 3. Check for essential data PRESENCE (expecting image_base64 now)
        if (!data.image_base64 || !data.metrics || !data.charges || data.greeks === undefined) { // Check greeks exists (can be empty list [])
             logger.warn("Response missing expected data fields (image_base64, metrics, charges, or greeks). Data:", data);
             // Provide a more specific "incomplete" message
             throw new Error("Server response was incomplete (missing chart/metrics/data).");
        }
        // If we pass all checks, proceed assuming data is valid

        // --- Render Chart using image_base64 ---
        if (chartContainer && data.image_base64) {
            chartContainer.innerHTML = ""; // Clear loading/placeholder
            const img = document.createElement("img");
            img.src = `data:image/png;base64,${data.image_base64}`;
            img.alt = `Option Strategy Payoff Chart for ${asset}`;
            img.className = "payoff-chart-image";
            chartContainer.appendChild(img);
            setElementState(SELECTORS.payoffChartContainer, 'content');
             logger.info("Successfully rendered Matplotlib chart image.");
        } else if (chartContainer) {
             // This case should ideally be caught by the check above, but added defensively
             setElementState(SELECTORS.payoffChartContainer, 'error', 'Chart image missing in response');
             logger.error("Payoff chart image_base64 missing in successful response.");
             // Allow proceeding to show other metrics even if image missing somehow
        }


        // --- Display Metrics, Breakdown, Taxes, Greeks (Keep as corrected previously) ---
        // Safely display metrics
        if (data.metrics?.metrics) {
            const metrics = data.metrics.metrics;
            displayMetric(metrics.max_profit, SELECTORS.maxProfitDisplay, "Max Profit: ", "", 2, true);
            displayMetric(metrics.max_loss, SELECTORS.maxLossDisplay, "Max Loss: ", "", 2, true);
            const breakevens = Array.isArray(metrics.breakeven_points) && metrics.breakeven_points.length > 0
                ? metrics.breakeven_points.map(p => formatCurrency(p, 2, 'N/A')).join(', ') : "None";
            displayMetric(breakevens, SELECTORS.breakevenDisplay, "Breakeven: ");
            displayMetric(metrics.reward_to_risk_ratio, SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
            const netPremiumPrefix = metrics.net_premium >= 0 ? "Net Credit: " : "Net Debit: ";
            displayMetric(Math.abs(metrics.net_premium), SELECTORS.netPremiumDisplay, netPremiumPrefix, "", 2, true);
        } else {
            logger.warn("Metrics data missing or invalid structure in response.");
            displayMetric("N/A", SELECTORS.maxProfitDisplay, "Max Profit: "); // Show N/A if missing
            displayMetric("N/A", SELECTORS.maxLossDisplay, "Max Loss: ");
            displayMetric("N/A", SELECTORS.breakevenDisplay, "Breakeven: ");
            displayMetric("N/A", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
            displayMetric("N/A", SELECTORS.netPremiumDisplay, "Net Premium: ");
        }

        // Display Cost Breakdown
        const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
        const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
        const costBreakdownData = data.metrics?.cost_breakdown_per_leg;
        if (breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) {
            breakdownList.innerHTML = ""; // Clear previous breakdown
            costBreakdownData.forEach(item => {
                const li = document.createElement("li");
                const premiumEffect = item.effect === 'Paid' ? `(Paid ${formatCurrency(Math.abs(item.total_premium))})` : `(Received ${formatCurrency(item.total_premium)})`;
                li.textContent = `${item.action} ${item.quantity} x ${item.type} @ ${item.strike} ${premiumEffect}`;
                breakdownList.appendChild(li);
            });
            breakdownContainer.style.display = "block";
            breakdownContainer.open = false;
            setElementState(SELECTORS.costBreakdownContainer, 'content');
        } else if (breakdownContainer) {
            setElementState(SELECTORS.costBreakdownContainer, 'hidden');
        }

        // Display Taxes/Charges
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
        if (taxContainer && data.charges) {
            renderTaxTable(taxContainer, data.charges);
            setElementState(SELECTORS.taxInfoContainer, 'content');
        } else if (taxContainer) {
             taxContainer.innerHTML = "<p>Charge data unavailable.</p>";
             setElementState(SELECTORS.taxInfoContainer, 'content');
        }

        // Display Greeks
        const greeksTable = document.querySelector(SELECTORS.greeksTable);
        // Check if data.greeks is an array (can be empty, which is valid)
        if (greeksTable && Array.isArray(data.greeks)) {
             renderGreeksTable(greeksTable, data.greeks); // Handles empty array internally
             setElementState(SELECTORS.greeksTable, 'content');
        } else if (greeksTable) {
            greeksTable.innerHTML = '<thead><tr><th>Greeks</th></tr></thead><tbody><tr><td>Greeks data unavailable.</td></tr></tbody>';
            setElementState(SELECTORS.greeksTable, 'content');
        }


    } catch (error) {
        logger.error("Error fetching or displaying payoff results:", error);
        // Display specific error message from the thrown error
        setElementState(SELECTORS.payoffChartContainer, 'error', `Calculation Error: ${error.message}`);
        // Reset other fields to show error occurred
        displayMetric("Error", SELECTORS.maxProfitDisplay, "Max Profit: ");
        displayMetric("Error", SELECTORS.maxLossDisplay, "Max Loss: ");
        displayMetric("Error", SELECTORS.breakevenDisplay, "Breakeven: ");
        displayMetric("Error", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
        displayMetric("Error", SELECTORS.netPremiumDisplay, "Net Premium: ");
        setElementState(SELECTORS.taxInfoContainer, 'error', 'Calculation Failed');
        setElementState(SELECTORS.greeksTable, 'error', 'Calculation Failed');
        setElementState(SELECTORS.costBreakdownContainer, 'hidden');

    } finally {
         if (updateButton) updateButton.disabled = false; // Re-enable button
    }
}

// --- Rendering Helpers for Payoff Results ---

function renderTaxTable(containerElement, taxData) {
    if (!taxData || !taxData.breakdown_per_leg || !taxData.charges_summary) {
        containerElement.innerHTML = "<p>Charge calculation data not available.</p>";
        return;
    }
    // Create a details/summary element for collapsibility
    containerElement.innerHTML = `
        <details class="results-details tax-details">
            <summary><strong>Estimated Charges Breakdown (Total: ${formatCurrency(taxData.total_estimated_cost, 2)})</strong></summary>
            <div class="table-wrapper"></div>
        </details>
    `;

    const tableWrapper = containerElement.querySelector('.table-wrapper');
    const table = document.createElement("table");
    table.className = "results-table charges-table";
    const charges = taxData.charges_summary || {};
    const breakdown = taxData.breakdown_per_leg || [];

    // Generate table rows safely accessing properties
    const tableBody = breakdown.map(t => `
        <tr>
            <td>${t.transaction_type || '?'}</td>
            <td>${t.option_type || '?'}</td>
            <td>${formatNumber(t.strike, 2, '-')}</td>
            <td>${formatNumber(t.lots, 0, '-')}</td>
            <td>${formatNumber(t.premium_per_share, 2, '-')}</td>
            <td>${formatNumber(t.stt, 2, '0.00')}</td>
            <td>${formatNumber(t.stamp_duty, 2, '0.00')}</td>
            <td>${formatNumber(t.sebi_fee, 4, '0.0000')}</td>
            <td>${formatNumber(t.txn_charge, 4, '0.0000')}</td>
            <td>${formatNumber(t.brokerage, 2, '0.00')}</td>
            <td>${formatNumber(t.gst, 2, '0.00')}</td>
            <td class="note" title="${t.stt_note || ''}">${(t.stt_note || '').substring(0, 15)}...</td>
        </tr>`).join('');

    // Note: Adjust colspan in tfoot to match the number of columns (12)
    table.innerHTML = `
        <thead>
            <tr><th>Act</th><th>Type</th><th>Strike</th><th>Lots</th><th>Premium</th><th>STT</th><th>Stamp</th><th>SEBI</th><th>Txn</th><th>Broker</th><th>GST</th><th title="Securities Transaction Tax Note">STT Note</th></tr>
        </thead>
        <tbody>${tableBody}</tbody>
        <tfoot>
            <tr class="totals-row">
                <td colspan="5">Total</td>
                <td>${formatCurrency(charges.stt, 2)}</td>
                <td>${formatCurrency(charges.stamp_duty, 2)}</td>
                <td>${formatCurrency(charges.sebi_fee, 4)}</td>
                <td>${formatCurrency(charges.txn_charges, 4)}</td>
                <td>${formatCurrency(charges.brokerage, 2)}</td>
                <td>${formatCurrency(charges.gst, 2)}</td>
                <td>Total: ${formatCurrency(taxData.total_estimated_cost, 2)}</td>
            </tr>
        </tfoot>`;
    tableWrapper.appendChild(table);
}

function renderGreeksTable(tableElement, greeksResult) {
    // Expects greeksResult = { greeks_data: [...], skipped_count: N, error: null | str }

    // --- Check for complete failure or empty data ---
    if (!greeksResult || !greeksResult.greeks_data || typeof greeksResult.skipped_count === 'undefined') {
        logger.warn("Greeks data structure is invalid or missing.");
        tableElement.innerHTML = `
            <caption class="table-caption">Portfolio Option Greeks</caption>
            <thead><tr><th>Info</th></tr></thead>
            <tbody><tr><td class="error-message">Greeks data unavailable or invalid format received.</td></tr></tbody>`;
        return;
    }

    const greeksData = greeksResult.greeks_data; // The list of leg results
    const skippedCount = greeksResult.skipped_count;
    const totalLegs = greeksData.length; // Total legs attempted

    // --- Case 1: All legs were skipped ---
    if (totalLegs > 0 && skippedCount === totalLegs) {
        logger.info(`All ${totalLegs} legs skipped for Greeks calculation (likely missing IV).`);
        tableElement.innerHTML = `
            <caption class="table-caption">Portfolio Option Greeks</caption>
            <thead><tr><th>Info</th></tr></thead>
            <tbody><tr><td class="skipped-reason">Greeks calculation skipped for all legs (likely due to missing IV).</td></tr></tbody>`;
        return; // Don't render the rest of the table
    }

    // --- Case 2: No legs were processed at all (empty list initially) ---
    if (totalLegs === 0) {
         logger.info("No legs were available for Greeks calculation.");
         tableElement.innerHTML = `
            <caption class="table-caption">Portfolio Option Greeks</caption>
            <thead><tr><th>Info</th></tr></thead>
            <tbody><tr><td>No strategy legs to calculate Greeks for.</td></tr></tbody>`;
         return;
    }

    // --- Proceed with rendering if at least one leg was processed (some might be skipped) ---
    let hasCalculatedGreeks = false; // Flag remains useful

    // --- Calculate Totals (Logic remains the same) ---
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    const scaling_factor = 100.0;
    greeksData.forEach(g => {
        if (g && g.calculated_greeks) {
            // ... (total calculation logic remains the same) ...
            const gv = g.calculated_greeks;
            const leg = g.input_data;
            if (!gv || !leg || typeof leg.lot !== 'number' || typeof leg.lot_size !== 'number' || leg.lot <= 0 || leg.lot_size <= 0) return;
            const quantity = leg.lot * leg.lot_size;
            if (typeof gv.delta === 'number') totals.delta += (gv.delta / scaling_factor) * quantity;
            if (typeof gv.gamma === 'number') totals.gamma += (gv.gamma / scaling_factor) * quantity;
            if (typeof gv.theta === 'number') totals.theta += (gv.theta / scaling_factor) * quantity;
            if (typeof gv.vega === 'number') totals.vega += (gv.vega / scaling_factor) * quantity;
            if (typeof gv.rho === 'number') totals.rho += (gv.rho / scaling_factor) * quantity;
            hasCalculatedGreeks = true;
        }
    });

    // --- Render Table (Logic remains the same, includes skipped rows) ---
    tableElement.className = "results-table greeks-table";
    const tableBodyContent = greeksData.map(g => {
        const leg = g?.input_data || {}; const gv = g?.calculated_greeks; const action = (leg.tr_type || '?').toUpperCase(); const lotsDisplay = `${leg.lot || '?'}x${leg.lot_size || '?'}`;
        if (gv) { /* Render calculated row */
            return `<tr class="greeks-calculated"><td>${action}</td><td>${lotsDisplay}</td><td>${(leg.op_type || '?').toUpperCase()}</td><td>${leg.strike}</td><td>${formatNumber(gv.delta, 2, '-')}</td><td>${formatNumber(gv.gamma, 4, '-')}</td><td>${formatNumber(gv.theta, 2, '-')}</td><td>${formatNumber(gv.vega, 2, '-')}</td><td>${formatNumber(gv.rho, 2, '-')}</td></tr>`;
        } else { /* Render skipped row */
            const reason = g?.error || "Skipped"; return `<tr class="greeks-skipped"><td>${action}</td><td>${lotsDisplay}</td><td>${(leg.op_type || '?').toUpperCase()}</td><td>${leg.strike}</td><td colspan="5" class="skipped-reason" title="${reason}">Greeks N/A (${reason.split(':')[0]})</td></tr>`;
        }
    }).join('');
    const captionText = skippedCount > 0 ? `Portfolio Option Greeks (Skipped for ${skippedCount}/${totalLegs} leg${skippedCount > 1 ? 's' : ''})` : `Portfolio Option Greeks`;
    tableElement.innerHTML = `
        <caption class="table-caption">${captionText}</caption>
        <thead><tr><th>Action</th><th>Quantity</th><th>Type</th><th>Strike</th><th title="Scaled Delta per share">Δ Delta</th><th title="Scaled Gamma per share">Γ Gamma</th><th title="Scaled Theta per share (Daily)">Θ Theta/Day</th><th title="Scaled Vega per share (per 1% IV)">Vega</th><th title="Scaled Rho per share (per 1% Rate)">Ρ Rho</th></tr></thead>
        <tbody>${tableBodyContent}</tbody>
        <tfoot>${hasCalculatedGreeks ? `<tr class="totals-row"><td colspan="4">Total Portfolio Greeks (Calc. Legs)</td><td>${formatNumber(totals.delta, 4)}</td><td>${formatNumber(totals.gamma, 4)}</td><td>${formatNumber(totals.theta, 4)}</td><td>${formatNumber(totals.vega, 4)}</td><td>${formatNumber(totals.rho, 4)}</td></tr>` : `<tr class="totals-row"><td colspan="9">No Greeks calculated for totals.</td></tr>`}</tfoot>`;
}

// ===============================================================
// Misc Helpers
// ===============================================================

/** Finds the strike closest to the current spot price */
function findATMStrike(strikes = [], spotPrice) {
    if (!Array.isArray(strikes) || strikes.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         return null;
    }
    // Ensure strikes are numbers and filter out any NaNs
    const numericStrikes = strikes.map(Number).filter(n => !isNaN(n));
    if(numericStrikes.length === 0) return null;

    // Find the strike with the minimum absolute difference from the spot price
    return numericStrikes.reduce((prev, curr) =>
        Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev
    );
}