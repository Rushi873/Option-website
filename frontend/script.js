// ===============================================================
// Configuration & Constants
// ===============================================================
// const API_BASE = "http://localhost:8000"; // For Local Hosting
const API_BASE = "https://option-strategy-website.onrender.com"; // For Production (Ensure this matches your deployed backend)
const REFRESH_INTERVAL_MS = 3000; // Auto-refresh interval (3 seconds )
const HIGHLIGHT_DURATION_MS = 1500; // How long highlights last

const SELECTORS = {
    assetDropdown: "#asset",
    expiryDropdown: "#expiry",
    spotPriceDisplay: "#spotPriceDisplay", // CORRECTED ID to match HTML
    optionChainTableBody: "#optionChainTable tbody",
    strategyTableBody: "#strategyTable tbody",
    updateChartButton: "#updateChartBtn",
    clearPositionsButton: "#clearStrategyBtn", // Corrected ID (was clearPositionsButton)
    payoffChartContainer: "#payoffChartContainer",
    analysisResultContainer: "#analysisResult",
    maxProfitDisplay: "#maxProfit .metric-value", // Target the value span
    maxLossDisplay: "#maxLoss .metric-value",     // Target the value span
    breakevenDisplay: "#breakeven .metric-value", // Target the value span
    rewardToRiskDisplay: "#rewardToRisk .metric-value", // Target the value span
    netPremiumDisplay: "#netPremium .metric-value",   // Target the value span
    costBreakdownContainer: "#costBreakdownContainer",
    costBreakdownList: "#costBreakdownList",
    newsResultContainer: "#newsResult",
    taxInfoContainer: "#taxInfo",
    greeksTable: "#greeksTable", // Selector for the entire table
    greeksTableBody: "#greeksTable tbody", // Added for direct access if needed
    globalErrorDisplay: "#globalError",
    greeksAnalysisSection: '#greeksAnalysisSection',
    greeksAnalysisResultContainer: '#greeksAnalysisResult',
    greeksTableContainer: '#greeksSection', 
    metricsList: '.metrics-list',
    statusMessageContainer: '#statusMessage',
    warningContainer: '#warningMessage', 
    
};

// Ensure 'strategyPositions' is declared in a scope accessible by resetResultsUI

// Basic Logger
const logger = {
    debug: (...args) => console.debug('[DEBUG]', ...args),
    info: (...args) => console.log('[INFO]', ...args),
    warn: (...args) => console.warn('[WARN]', ...args),
    error: (...args) => console.error('[ERROR]', ...args),
};

// ===============================================================
// Global State
// ===============================================================
let currentSpotPrice = 0;
let strategyPositions = []; // Holds { strike_price, expiry_date, option_type, lots, last_price, iv, days_to_expiry }
let activeAsset = null;
let autoRefreshIntervalId = null; // Timer ID for auto-refresh
let previousOptionChainData = {}; // Store previous chain data for highlighting
let previousSpotPrice = 0; // Store previous spot price for highlighting


// ===============================================================
// Utility Functions (Enhanced)
// ===============================================================

/** Safely formats a number or returns a fallback string, handling backend specials */
function formatNumber(value, decimals = 2, fallback = "N/A") {
    if (value === null || typeof value === 'undefined') { return fallback; }
    // Handle specific string representations from backend (like infinity, loss)
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
        if (["∞", "INFINITY"].includes(upperVal)) return "∞";
        if (["-∞", "-INFINITY"].includes(upperVal)) return "-∞";
        if (["N/A", "UNDEFINED", "LOSS"].includes(upperVal)) return value; // Pass through specific statuses
        // Attempt to convert other strings (e.g., numbers sent as strings)
    }
    const num = Number(value);
    if (!isNaN(num)) {
        if (num === Infinity) return "∞";
        if (num === -Infinity) return "-∞";
        // Use locale string for commas and specified decimals
        return num.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    // If it's a string we didn't specifically handle and couldn't convert to number
    if (typeof value === 'string') return value; // Return the original string (might be a label)
    return fallback; // Fallback for other non-numeric types
}

/** Safely formats currency, handling backend specials */
function formatCurrency(value, decimals = 2, fallback = "N/A", prefix = "₹") {
     // Handle specific non-numeric strings first
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
         if (["∞", "INFINITY", "-∞", "-INFINITY", "N/A", "UNDEFINED", "LOSS"].includes(upperVal)) {
             return value; // Return these special strings directly
         }
    }
    // Try formatting as number, use null fallback to distinguish between valid 0 and error
    const formattedNumber = formatNumber(value, decimals, null);

    if (formattedNumber !== null && !["∞", "-∞"].includes(formattedNumber)) { // Don't prefix infinity
        // Check if the value was negative before formatting potentially removed the sign (though toLocaleString usually handles it)
        const originalNum = Number(value);
        const displayPrefix = (originalNum < 0) ? `-${prefix}` : prefix;
        // Use the absolute value for formatting if prefixing negative sign manually (usually not needed)
        return `${displayPrefix}${formatNumber(Math.abs(originalNum), decimals, 'Error')}`;
        // Simpler version if toLocaleString handles negative signs correctly:
        // return `${prefix}${formattedNumber}`;
    }
    // Return ∞/-∞ as is, or the fallback if formatting failed
    return formattedNumber === null ? fallback : formattedNumber;
}


/** Helper to display formatted metric/value in a UI element */
function displayMetric(value, targetElementSelector, prefix = '', suffix = '', decimals = 2, isCurrency = false) {
     const element = document.querySelector(targetElementSelector);
     if (!element) {
        logger.warn(`displayMetric: Element not found for selector "${targetElementSelector}"`);
        return;
     }
     const formatFunc = isCurrency ? formatCurrency : formatNumber;
     const formattedValue = formatFunc(value, decimals, "N/A", ""); // Get base formatted value

     // Construct final string
     element.textContent = `${prefix}${formattedValue}${suffix}`;
}

/** Sets the loading/error/content state for an element. */
function setElementState(selector, state, message = 'Loading...') {
    const element = document.querySelector(selector);
    if (!element) { logger.warn(`setElementState: Element not found for "${selector}"`); return; }

    const isSelect = element.tagName === 'SELECT';
    const isButton = element.tagName === 'BUTTON';
    const isTbody = element.tagName === 'TBODY';
    const isTable = element.tagName === 'TABLE';
    const isContainer = element.tagName === 'DIV' || element.tagName === 'SECTION' || element.classList.contains('chart-container');
    const isSpan = element.tagName === 'SPAN'; // e.g., spot price part
    const isErrorMessage = element.id === SELECTORS.globalErrorDisplay.substring(1);

    // Reset states and styles
    element.classList.remove('loading', 'error', 'loaded', 'hidden');
    if (isSelect || isButton) element.disabled = false;
    element.style.display = ''; // Default display
    if (isErrorMessage) {
        element.style.color = ''; // Reset error color
        element.style.display = 'none'; // Default hidden for global error
    }
    if (isContainer && state !== 'error') { // Clear previous content unless setting new error
       // element.innerHTML = ''; // Clear only if not setting error
    }


    switch (state) {
        case 'loading':
            element.classList.add('loading');
            if (isSelect) element.innerHTML = `<option>${message}</option>`;
            else if (isTbody) element.innerHTML = `<tr><td colspan="7" class="loading-text">${message}</td></tr>`; // Adjust colspan if needed
            else if (isTable) { // Handle table loading state
                const caption = element.querySelector('caption');
                if (caption) caption.textContent += ' (Loading...)';
                const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="9" class="loading-text">${message}</td></tr>`; // Colspan for greeks
                const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
            }
            else if (isContainer) element.innerHTML = `<div class="loading-text" style="padding: 20px; text-align: center;">${message}</div>`;
            else if (isSpan) element.textContent = message; // Set loading text for spans too
            else if (!isButton && !isErrorMessage) element.textContent = message; // Other text elements
            if (isSelect || isButton) element.disabled = true;
            if (isErrorMessage) element.style.display = 'block'; // Show loading if used for global error
            break;
        case 'error':
            element.classList.add('error');
            const displayMessage = `${message}`; // No "Error:" prefix, let context show it's an error
            if (isSelect) { element.innerHTML = `<option>${displayMessage}</option>`; element.disabled = true; }
            else if (isTbody) { element.innerHTML = `<tr><td colspan="7" class="error-message">${displayMessage}</td></tr>`; }
            else if (isTable) { // Handle table error state
                 const caption = element.querySelector('caption');
                 if (caption) caption.textContent += ' (Error)';
                 const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="9" class="error-message">${displayMessage}</td></tr>`; // Colspan for greeks
                 const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = '';
            }
            else if (isContainer) { element.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`; } // Display error within container
            else if (isSpan) { element.textContent = 'Error'; element.classList.add('error-message'); } // Concise error for spans
            else { element.textContent = displayMessage; element.classList.add('error-message');}
            if (isErrorMessage) element.style.display = 'block'; // Ensure global error is visible
            break;
        case 'content':
            element.classList.add('loaded');
            // Calling function will set the actual content after this call
            if (isErrorMessage) element.style.display = 'none'; // Hide global error when setting other content
            break;
         case 'hidden':
             element.classList.add('hidden'); // Use class for hidden state
             element.style.display = 'none';
            break;
    }
}


/** Populates a dropdown select element. */
function populateDropdown(selector, items, placeholder = "-- Select --", defaultSelectFirst = false) {
    const selectElement = document.querySelector(selector);
    if (!selectElement) return;
    const currentValue = selectElement.value; // Store current value before clearing
    selectElement.innerHTML = ""; // Clear existing options

    if (!items || items.length === 0) {
        selectElement.innerHTML = `<option value="">-- No options available --</option>`;
        selectElement.disabled = true;
        return;
    }

    // Optional placeholder
    if (placeholder) {
        const placeholderOption = document.createElement("option");
        placeholderOption.value = "";
        placeholderOption.textContent = placeholder;
        placeholderOption.disabled = true; // Make placeholder unselectable if needed
        // placeholderOption.selected = true; // Select it initially
        selectElement.appendChild(placeholderOption);
    }

    items.forEach(item => {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });

    // Try to restore previous selection or select first item
    let valueSet = false;
    if (items.includes(currentValue)) {
        selectElement.value = currentValue;
        valueSet = true;
    } else if (defaultSelectFirst && items.length > 0) {
         selectElement.value = items[0];
         valueSet = true;
    }

     // If no value was set (e.g., placeholder was added but nothing else selected)
     if (!valueSet && placeholder) {
         selectElement.value = ""; // Ensure placeholder is selected
     }

    selectElement.disabled = false;
}

/** Fetches data from the API with error handling. */
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    options.headers = { ...defaultHeaders, ...options.headers };
    const method = options.method || 'GET';
    const requestBody = options.body ? JSON.parse(options.body) : '';
    logger.debug(`fetchAPI Request: ${method} ${url}`, requestBody);

    try {
        const response = await fetch(url, options);
        let responseData = null;
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
             try {
                 responseData = await response.json(); // Attempt to parse JSON
             } catch (jsonError) {
                  logger.error(`API Error (${method} ${url} - ${response.status}): Failed to parse JSON response.`, jsonError);
                  // Throw a specific error for JSON parsing failure
                  throw new Error(`Invalid JSON response from server (Status: ${response.status})`);
             }
        } else if (response.status !== 204) { // Handle non-JSON, non-empty responses if necessary
             logger.warn(`Received non-JSON response from ${method} ${url} (Status: ${response.status}). Body might be plain text.`);
             // Optionally try reading as text: responseData = await response.text();
        }

        logger.debug(`fetchAPI Response Status: ${response.status} for ${method} ${url}`);

        if (!response.ok) {
            // Try to get detail from JSON, fallback to statusText, then generic message
            const errorMessage = responseData?.detail // FastAPI typical error field
                              || responseData?.message // Other potential error field
                              || response.statusText   // Standard HTTP status text
                              || `HTTP error ${response.status}`; // Generic fallback
            logger.error(`API Error (${method} ${url} - ${response.status}): ${errorMessage}`, responseData);
            // Display global error for backend errors (like 500s)
            setElementState(SELECTORS.globalErrorDisplay, 'error', `Server Error (${response.status}): ${errorMessage}`);
            throw new Error(errorMessage); // Throw standardized error message
        }

         // Clear global error on successful API call
         setElementState(SELECTORS.globalErrorDisplay, 'hidden');
         logger.debug(`fetchAPI Response Data:`, responseData);
        return responseData; // Return parsed data (or null for 204)

    } catch (error) {
        // Catch network errors or errors thrown above (JSON parse, backend error)
        logger.error(`Fetch/Network Error or API Error (${method} ${url}):`, error);
        // Display global error, using the error message generated above or from network failure
        setElementState(SELECTORS.globalErrorDisplay, 'error', `${error.message || 'Network error or invalid response'}`);
        throw error; // Re-throw for specific UI handling if needed by caller
    }
}


/** Applies a temporary highlight effect to an element */
function highlightElement(element) {
    if (!element) return;
    element.classList.add('value-changed');
    // Use requestAnimationFrame for potentially smoother removal
    setTimeout(() => {
        requestAnimationFrame(() => {
            element?.classList.remove('value-changed'); // Optional chaining for safety
        });
    }, HIGHLIGHT_DURATION_MS);
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
    logger.info("Initializing page: Setting loading states...");
    // ... (set initial loading states) ...
    resetResultsUI();
    setupEventListeners();
    loadMarkdownParser(); // Load markdown early

    try {
        // Load assets and get the default asset value
        const defaultAsset = await loadAssets(); // loadAssets populates dropdown and returns default

        if (defaultAsset) {
            logger.info(`Default asset determined: ${defaultAsset}. Fetching initial static data...`);
            // Set loading states for news/analysis
            setElementState(SELECTORS.analysisResultContainer, 'loading', 'Loading analysis...');
            setElementState(SELECTORS.newsResultContainer, 'loading', 'Loading news...');

            // Fetch initial News & Analysis concurrently
            const initialStaticFetches = Promise.allSettled([
                fetchAnalysis(defaultAsset),
                fetchNews(defaultAsset)
            ]);

            // Trigger the *market data* load for the default asset
            // This will update activeAsset and fetch Spot/Expiry/Chain
            const initialMarketFetch = handleAssetChange(); // Don't await here yet if we want static to run in parallel

            // Await both static data fetches AND the market data fetch
            const [staticResults, marketResult] = await Promise.allSettled([
                 initialStaticFetches,
                 initialMarketFetch // Await the handleAssetChange completion
            ]);

            // Log errors from static fetches
            if (staticResults.status === 'fulfilled') {
                const [analysisRes, newsRes] = staticResults.value; // results of Promise.allSettled inside
                 if (analysisRes.status === 'rejected') { logger.error(`Initial analysis fetch failed: ${analysisRes.reason?.message || analysisRes.reason}`); }
                 if (newsRes.status === 'rejected') { logger.error(`Initial news fetch failed: ${newsRes.reason?.message || newsRes.reason}`); }
            } else {
                 logger.error("Error settling initial static fetches:", staticResults.reason);
            }

            // Log error from market data fetch (handleAssetChange) if it occurred
            // Note: handleAssetChange has its own internal error handling/logging
            if (marketResult.status === 'rejected') {
                logger.error(`Initial market data fetch (handleAssetChange) failed: ${marketResult.reason?.message || marketResult.reason}`);
                // Maybe set global error here if market data fails critically
                setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load initial market data for ${defaultAsset}.`);
            } else {
                 logger.info(`Initial market data fetch (handleAssetChange) completed for ${defaultAsset}.`);
            }

        } else {
            // Keep the existing logic for when no assets are found
            logger.warn("No default asset set. Waiting for user selection.");
            // ... (set placeholders for analysis, news, expiry, chain, spot) ...
        }

    } catch (error) {
        // Keep existing catch block for initialization failures
        logger.error("Page Initialization failed:", error);
        // ... (set error states) ...
    }
    // This log now accurately reflects the end of the entire sequence
    logger.info("Initialization sequence complete.");
}

/** Loads assets, populates dropdown, sets default, and returns the default asset value. */
async function loadAssets() {
    logger.info("Loading assets...");
    setElementState(SELECTORS.assetDropdown, 'loading');
    let defaultAsset = null; // Variable to store the default asset

    try {
        const data = await fetchAPI("/get_assets"); // Your endpoint to get assets
        const assets = data?.assets || [];
        const defaultSelection = assets.includes("NIFTY") ? "NIFTY" : (assets[0] || null);

        populateDropdown(SELECTORS.assetDropdown, assets, "-- Select Asset --", defaultSelection);
        setElementState(SELECTORS.assetDropdown, 'content');

        const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
        if (assetDropdown && assetDropdown.value) {
             defaultAsset = assetDropdown.value; // Get the actual selected default value
             logger.info(`Assets loaded. Default selected: ${defaultAsset}`);
        } else if (assets.length === 0) {
             logger.warn("No assets found in database.");
             setElementState(SELECTORS.assetDropdown, 'error', 'No assets');
        } else {
            logger.warn("Asset dropdown populated, but no default value could be determined.");
            // Keep defaultAsset as null
        }

    } catch (error) {
        logger.error("Failed to load assets:", error);
        populateDropdown(SELECTORS.assetDropdown, [], "-- Error Loading --");
        setElementState(SELECTORS.assetDropdown, 'error', `Error`);
        // Don't set global error here, let initializePage handle it
        throw error; // Re-throw error to be caught by initializePage
    }
    return defaultAsset; // Return the determined default asset
}


function setupEventListeners() {
    logger.info("Setting up event listeners...");
    document.querySelector(SELECTORS.assetDropdown)?.addEventListener("change", handleAssetChange);
    document.querySelector(SELECTORS.expiryDropdown)?.addEventListener("change", handleExpiryChange);
    document.querySelector(SELECTORS.updateChartButton)?.addEventListener("click", fetchPayoffChart);
    document.querySelector(SELECTORS.clearPositionsButton)?.addEventListener("click", clearAllPositions);

    // Event delegation for strategy table interaction
    const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (strategyTableBody) {
        strategyTableBody.addEventListener('input', handleStrategyTableChange); // Use 'input' for immediate feedback on number change
        strategyTableBody.addEventListener('click', handleStrategyTableClick);
    }

    // Event delegation for option chain table clicks (moved from fetchOptionChain)
     const optionChainTableBody = document.querySelector(SELECTORS.optionChainTableBody);
     if (optionChainTableBody) {
         optionChainTableBody.addEventListener('click', handleOptionChainClick);
     }
}


function loadMarkdownParser() {
    // Check if marked is already loaded (e.g., if script tag was already in HTML)
    if (typeof marked !== 'undefined') {
        logger.info("Markdown parser (marked.js) already available.");
        return;
    }
    // Dynamically load if not present
    try {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
        script.async = true; // Load asynchronously
        script.onload = () => logger.info("Markdown parser (marked.js) loaded dynamically.");
        script.onerror = () => logger.error("Failed to load Markdown parser (marked.js). Analysis rendering may fail.");
        document.head.appendChild(script);
    } catch (e) {
         logger.error("Error creating script tag for marked.js", e);
    }
}

// ===============================================================
// Auto-Refresh Logic
// ===============================================================

async function refreshLiveData() {
    // Check if asset and expiry are selected before refreshing
    if (!activeAsset) {
        logger.warn("Auto-refresh called with no active asset. Stopping.");
        stopAutoRefresh();
        return;
    }
    const currentExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!currentExpiry) {
        logger.debug("Auto-refresh skipped: No expiry selected.");
        return; // Don't refresh chain if no expiry selected
    }

    logger.debug(`Auto-refreshing live data (Spot & Chain) for ${activeAsset}...`); // Clarify log

    // Fetch ONLY dynamic data concurrently
    const results = await Promise.allSettled([
        fetchNiftyPrice(activeAsset, true), // Pass true for refresh call (handles spot display)
        fetchOptionChain(false, true)       // No scroll, is refresh call (handles chain display)
        // --- DO NOT FETCH NEWS OR ANALYSIS HERE ---
    ]);

    // Log errors from settled promises if any
    results.forEach((result, index) => {
        if (result.status === 'rejected') {
            const source = index === 0 ? 'Spot price' : 'Option chain';
             // Log as warning, as refresh might fail temporarily
             logger.warn(`Auto-refresh: ${source} fetch failed: ${result.reason?.message || result.reason}`);
             // Optionally display a subtle error indicator for the failing component
        }
    });
     logger.debug(`Auto-refresh cycle finished for ${activeAsset}.`);
}


function startAutoRefresh() {
    stopAutoRefresh(); // Clear any existing timer first
    if (!activeAsset) {
        logger.info("No active asset, auto-refresh not started.");
        return;
    }
    // Assuming REFRESH_INTERVAL_MS is defined elsewhere (e.g., const REFRESH_INTERVAL_MS = 30000;)
    logger.info(`Starting auto-refresh every ${REFRESH_INTERVAL_MS / 1000}s for ${activeAsset}`);
    // Store previous data *before* starting the interval if needed by fetch functions
    previousSpotPrice = currentSpotPrice;
    // previousOptionChainData is likely updated within fetchOptionChain itself now
    autoRefreshIntervalId = setInterval(refreshLiveData, REFRESH_INTERVAL_MS);
}

function stopAutoRefresh() {
    if (autoRefreshIntervalId) {
        clearInterval(autoRefreshIntervalId);
        autoRefreshIntervalId = null;
        logger.info("Auto-refresh stopped.");
    }
}

async function refreshLiveData() {
    if (!activeAsset) {
        logger.warn("Auto-refresh called with no active asset. Stopping.");
        stopAutoRefresh();
        return;
    }
    const currentExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!currentExpiry) {
        logger.debug("Auto-refresh skipped: No expiry selected.");
        return; // Don't refresh chain if no expiry selected
    }

    logger.debug(`Auto-refreshing data for ${activeAsset}...`);
    // Fetch data concurrently using Promise.allSettled to avoid stopping if one fails
    const results = await Promise.allSettled([
        fetchNiftyPrice(activeAsset, true), // Pass true to indicate it's a refresh call
        fetchOptionChain(false, true) // No scroll, but is refresh call
    ]);

    // Log errors from settled promises if any
    results.forEach((result, index) => {
        if (result.status === 'rejected') {
            const source = index === 0 ? 'Spot price' : 'Option chain';
             logger.warn(`Auto-refresh: ${source} fetch failed: ${result.reason?.message || result.reason}`);
        }
    });
}


// ===============================================================
// Event Handlers & Data Fetching Logic
// ===============================================================

/** Handles asset dropdown change */
async function handleAssetChange() {
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
    const asset = assetDropdown?.value;

    // Prevent running if asset hasn't actually changed or is empty
    // This check prevents redundant fetches if the same asset is re-selected
    if (!asset || asset === activeAsset) {
        if (!asset) {
            // Handle the case where the selection becomes empty (e.g., "-- Select --")
             logger.info("Asset selection cleared.");
             activeAsset = null; // Clear active asset
             stopAutoRefresh();
             // Reset UI completely
             populateDropdown(SELECTORS.expiryDropdown, [], "-- Select Asset First --");
             setElementState(SELECTORS.optionChainTableBody, 'content');
             document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Asset</td></tr>`;
             setElementState(SELECTORS.spotPriceDisplay, 'content');
             document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
             resetResultsUI(); // This clears strategy and all results including news/analysis
             setElementState(SELECTORS.globalErrorDisplay, 'hidden');
        } else {
            logger.debug(`handleAssetChange skipped: Asset unchanged (${asset}).`);
        }
        return;
    }


    logger.info(`Asset changed to: ${asset}. Fetching Spot, Expiry, News, Analysis...`); // Updated log
    activeAsset = asset; // Update global state
    stopAutoRefresh(); // Stop refresh for the old asset

    // Clear previous option chain interaction data
    previousOptionChainData = {};
    previousSpotPrice = 0;
    currentSpotPrice = 0;

    // Reset results UI AND strategy input table (using the combined function)
    resetResultsUI();

    // Set loading states for ALL sections being fetched for the new asset
    setElementState(SELECTORS.expiryDropdown, 'loading');
    setElementState(SELECTORS.optionChainTableBody, 'loading');
    setElementState(SELECTORS.analysisResultContainer, 'loading'); // Fetching Analysis
    setElementState(SELECTORS.newsResultContainer, 'loading');    // Fetching News
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...');
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear global error

    // --- Optional Debug Call ---
    // (Keep as is)
    try {
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST', body: JSON.stringify({ asset: asset })
        });
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        logger.error("Failed to send debug asset selection:", debugErr.message);
    }
    // --- End Debug Call ---

    try {
        // Fetch Spot, Expiry, News, and Analysis concurrently for the NEW asset
        const [spotResult, expiryResult, analysisResult, newsResult] = await Promise.allSettled([
            fetchNiftyPrice(asset), // Initial fetch for this asset
            fetchExpiries(asset),
            fetchAnalysis(asset),   // <<< Fetch analysis for the new asset
            fetchNews(asset)        // <<< Fetch news for the new asset
        ]);

        let hasCriticalError = false;

        // Process critical results (Spot/Expiry)
        if (spotResult.status === 'rejected') {
            logger.error(`Error fetching spot price for ${asset}: ${spotResult.reason?.message || spotResult.reason}`);
            hasCriticalError = true;
            setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error'); // Show error on spot display
        }
        if (expiryResult.status === 'rejected') {
            logger.error(`Error fetching expiries for ${asset}: ${expiryResult.reason?.message || expiryResult.reason}`);
            hasCriticalError = true;
            setElementState(SELECTORS.expiryDropdown, 'error', 'Failed');
            setElementState(SELECTORS.optionChainTableBody, 'error', 'Failed to load expiries');
        }

        // Log non-critical failures (News/Analysis) - errors are handled within their functions
        if (analysisResult.status === 'rejected') {
            logger.error(`Error fetching analysis for ${asset} during asset change: ${analysisResult.reason?.message || analysisResult.reason}`);
        }
         if (newsResult.status === 'rejected') {
            logger.error(`Error fetching news for ${asset} during asset change: ${newsResult.reason?.message || newsResult.reason}`);
        }

        // Start auto-refresh ONLY if critical data loaded
        if (!hasCriticalError) {
            logger.info(`Essential data loaded for ${asset}. Starting auto-refresh.`);
            startAutoRefresh(); // Start refresh for the new asset
        } else {
             logger.error(`Failed to load essential data (spot/expiries) for ${asset}. Auto-refresh NOT started.`);
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load essential market data for ${asset}. Option chain unavailable.`);
        }

    } catch (err) {
        // Catch unexpected errors during orchestration
        logger.error(`Unexpected error during handleAssetChange for ${asset}:`, err);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load data for ${asset}. ${err.message}`);
        stopAutoRefresh(); // Stop refresh on major load error
        // Set multiple areas to error state
        setElementState(SELECTORS.expiryDropdown, 'error', 'Failed');
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Failed');
        setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error');
        setElementState(SELECTORS.analysisResultContainer, 'error', 'Failed to load'); // Also indicate error here
        setElementState(SELECTORS.newsResultContainer, 'error', 'Failed to load'); // And here
    }
}


/** Handles expiry dropdown change */
async function handleExpiryChange() {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    // Clear previous chain data when expiry changes
    previousOptionChainData = {};
    if (!expiry) {
        setElementState(SELECTORS.optionChainTableBody, 'content');
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Expiry</td></tr>`;
        return;
    }
    logger.info(`Expiry changed to: ${expiry}. Fetching option chain...`);
    await fetchOptionChain(true); // Fetch new chain and scroll to ATM
}

/** Fetches and populates the asset dropdown */
async function loadAssets() {
    logger.info("Loading assets...");
    setElementState(SELECTORS.assetDropdown, 'loading');
    let defaultAsset = null;

    try {
        const data = await fetchAPI("/get_assets");
        const assets = data?.assets || [];
        // Determine default *before* populating if possible
        const potentialDefault = assets.includes("NIFTY") ? "NIFTY" : (assets[0] || null);

        // Populate dropdown, setting the determined default
        populateDropdown(SELECTORS.assetDropdown, assets, "-- Select Asset --", potentialDefault);
        setElementState(SELECTORS.assetDropdown, 'content');

        const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
        if (assetDropdown && assetDropdown.value) {
             defaultAsset = assetDropdown.value; // Confirm the actual selected default value
             logger.info(`Assets loaded. Default selected: ${defaultAsset}`);
        } else if (assets.length === 0) {
             logger.warn("No assets found in database.");
             setElementState(SELECTORS.assetDropdown, 'error', 'No assets');
        } else {
            logger.warn("Asset dropdown populated, but no default value could be determined.");
        }

    } catch (error) {
        // Keep existing catch block
        logger.error("Failed to load assets:", error);
        // ... (set error states) ...
        throw error; // Re-throw
    }
    return defaultAsset; // Return the determined default asset value
}

/** Fetches stock analysis for the selected asset */
async function fetchAnalysis(asset) {
    const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
    if (!analysisContainer) {
         logger.warn("Analysis container element not found.");
         return; // Exit if container doesn't exist
    }
    if (!asset) {
         analysisContainer.innerHTML = 'Select an asset to load analysis...';
         setElementState(SELECTORS.analysisResultContainer, 'content');
         return; // Exit if no asset
    }

    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Fetching analysis...');
    logger.debug(`Fetching analysis for ${asset}...`);

    try {
        // Ensure marked.js is loaded
        let attempts = 0;
        while (typeof marked === 'undefined' && attempts < 10) {
            logger.debug("Waiting for marked.js...");
            await new Promise(resolve => setTimeout(resolve, 200)); // Wait 200ms
            attempts++;
        }
        if (typeof marked === 'undefined') {
            throw new Error("Markdown parser (marked.js) failed to load.");
        }
        logger.debug("marked.js loaded.");

        // Call backend endpoint (unchanged call structure)
        const data = await fetchAPI("/get_stock_analysis", {
            method: "POST", body: JSON.stringify({ asset })
        });
        logger.debug(`Received analysis data for ${asset}`);

        const rawAnalysis = data?.analysis || "*No analysis content received.*";
        // Basic sanitization (consider a more robust library like DOMPurify if complex HTML is possible)
        const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
        // Use marked.parse()
        analysisContainer.innerHTML = marked.parse(potentiallySanitized);
        setElementState(SELECTORS.analysisResultContainer, 'content');
        logger.info(`Successfully rendered analysis for ${asset}`);

    } catch (error) {
        logger.error(`Error fetching or rendering analysis for ${asset}:`, error);
        let displayMessage = `Analysis Error: ${error.message}`;

        // === UPDATED ERROR HANDLING ===
        // Check specifically for the 404-like error message from the backend
        // Use .includes() for flexibility as symbol/region might vary
        if (error.message && error.message.includes("Essential stock data not found")) {
            // Display the specific 404 message clearly inside the container
            displayMessage = error.message; // Use the detailed message from backend
            analysisContainer.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`;
            setElementState(SELECTORS.analysisResultContainer, 'content'); // Set state to content, but show error message inside
        } else if (error.message && error.message.includes("Analysis blocked by content filter")) {
             // Handle content filter blocks specifically
             displayMessage = `Analysis generation failed due to content restrictions.`;
             analysisContainer.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`;
             setElementState(SELECTORS.analysisResultContainer, 'content'); // Set state to content, but show error message inside
        } else if (error.message && error.message.includes("Analysis generation failed")) {
             // Handle other specific generation failures from the backend
             displayMessage = error.message; // Show the specific failure reason
             analysisContainer.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`;
             setElementState(SELECTORS.analysisResultContainer, 'content'); // Set state to content, but show error message inside
        }
         else {
            // For other errors (network, server 500s, JSON parse, etc.), show generic error message
            setElementState(SELECTORS.analysisResultContainer, 'error', displayMessage);
        }
        // Avoid setting global error for analysis-specific issues
        // throw error; // Do not re-throw unless handleAssetChange needs to react specifically
    }
}

/** Fetches and renders news for the selected asset */
async function fetchNews(asset) {
    if (!asset) return;
    const newsContainer = document.querySelector(SELECTORS.newsResultContainer);
    if (!newsContainer) {
        logger.warn("News container element not found.");
        return;
    }
    setElementState(SELECTORS.newsResultContainer, 'loading', 'Fetching news...');

    try {
        // Call the new backend endpoint
        const data = await fetchAPI(`/get_news?asset=${encodeURIComponent(asset)}`);
        const newsItems = data?.news; // Expects { news: [...] }

        if (Array.isArray(newsItems)) {
            renderNews(newsContainer, newsItems); // Render the fetched items
            setElementState(SELECTORS.newsResultContainer, 'content');
        } else {
            logger.error("Invalid news data format received:", data);
            throw new Error("Invalid news data format from server.");
        }
    } catch (error) {
        logger.error(`Error fetching or rendering news for ${asset}:`, error);
        // Display error within the news container
        setElementState(SELECTORS.newsResultContainer, 'error', `News Error: ${error.message}`);
        // Re-throw error so handleAssetChange knows about the failure (optional)
        // throw error;
    }
}


/** Renders the news items into the specified container */
function renderNews(containerElement, newsData) {
    containerElement.innerHTML = ""; // Clear previous content (loading/error)

    if (!newsData || newsData.length === 0) {
        containerElement.innerHTML = '<p>No recent news found for this asset.</p>';
        return;
    }

    // Handle potential error messages returned within the newsData array
    if (newsData.length === 1 && newsData[0].headline.startsWith("Error fetching news")) {
         containerElement.innerHTML = `<p class="error-message">${newsData[0].headline}</p>`;
         return;
    }
     if (newsData.length === 1 && newsData[0].headline.startsWith("No recent news found")) {
         containerElement.innerHTML = `<p>${newsData[0].headline}</p>`;
         return;
    }


    const ul = document.createElement("ul");
    ul.className = "news-list"; // Add class for styling

    newsData.forEach(item => {
        const li = document.createElement("li");
        li.className = "news-item"; // Add class for styling

        const headline = document.createElement("div");
        headline.className = "news-headline";
        const link = document.createElement("a");
        link.href = item.link || "#";
        link.textContent = item.headline || "No Title";
        link.target = "_blank"; // Open in new tab
        link.rel = "noopener noreferrer";
        headline.appendChild(link);

        const summary = document.createElement("p");
        summary.className = "news-summary";
        summary.textContent = item.summary || "No summary available.";

        li.appendChild(headline);
        li.appendChild(summary);
        ul.appendChild(li);
    });

    containerElement.appendChild(ul);
}


/** Fetches and populates expiry dates */
async function fetchExpiries(asset) {
    if (!asset) return;
    setElementState(SELECTORS.expiryDropdown, 'loading');
    try {
        const data = await fetchAPI(`/expiry_dates?asset=${encodeURIComponent(asset)}`);
        const expiries = data?.expiry_dates || [];
        populateDropdown(SELECTORS.expiryDropdown, expiries, "-- Select Expiry --", true); // Select first expiry
        setElementState(SELECTORS.expiryDropdown, 'content');

        const selectedExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
        if (selectedExpiry) {
            // handleExpiryChange will be called automatically by the change event if value changed
            // If the value didn't change (e.g., only one expiry), manually trigger chain load
            if (document.querySelector(SELECTORS.expiryDropdown).selectedIndex === (expiries.length > 0 ? 1 : 0)) { // Check if first actual option selected
                 await fetchOptionChain(true); // Manually load chain for the single/first expiry
            }
        } else {
             logger.warn(`No expiry dates found for asset: ${asset}`);
            setElementState(SELECTORS.optionChainTableBody, 'content');
            document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">No expiry dates found for ${asset}</td></tr>`;
        }

    } catch (error) {
        setElementState(SELECTORS.expiryDropdown, 'error', `Expiries Error: ${error.message}`);
        // Don't set chain error here, let the caller (handleAssetChange) manage overall state
        throw error; // Re-throw so handleAssetChange knows about the failure
    }
}


/** Fetches and displays the spot price using the correct endpoint */
async function fetchNiftyPrice(asset, isRefresh = false) {
    if (!asset) return;
    const priceElement = document.querySelector(SELECTORS.spotPriceDisplay);

    if (!isRefresh) {
        setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...');
    }

    try {
        const data = await fetchAPI(`/get_spot_price?asset=${encodeURIComponent(asset)}`);
        const newSpotPrice = data?.spot_price;
        // const timestamp = data?.timestamp; // We don't need timestamp anymore

        if (newSpotPrice === null || typeof newSpotPrice === 'undefined') {
             throw new Error("Spot price not available from API.");
        }

        if (!isRefresh || previousSpotPrice === 0) {
            previousSpotPrice = currentSpotPrice;
        }
        currentSpotPrice = newSpotPrice;

        if (priceElement) {
            // ***** CHANGE THIS LINE *****
            // const timeText = timestamp ? ` (as of ${new Date(timestamp).toLocaleTimeString()})` : ''; // REMOVE
            // priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}${timeText}`; // REMOVE
            priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}`; // USE THIS
            // ***** END OF CHANGE *****

             if (!isRefresh) {
                 setElementState(SELECTORS.spotPriceDisplay, 'content');
             }
            if (isRefresh && currentSpotPrice !== previousSpotPrice && previousSpotPrice !== 0) {
                 logger.debug(`Spot price changed: ${previousSpotPrice} -> ${currentSpotPrice}`);
                 highlightElement(priceElement);
                 previousSpotPrice = currentSpotPrice;
            } else if (isRefresh) {
                 previousSpotPrice = currentSpotPrice;
            }
        }
    } catch (error) {
         // ... (keep error handling as is) ...
         if (!isRefresh) {
             currentSpotPrice = 0;
             previousSpotPrice = 0;
             setElementState(SELECTORS.spotPriceDisplay, 'error', `Spot Price Error`);
         } else {
             logger.warn(`Spot Price refresh Error (${asset}):`, error.message);
         }
         if (!isRefresh) throw error; // Re-throw only on initial load failure
    }
}


/** Fetches and displays the option chain, optionally highlights changes */
/** Fetches and displays the option chain, optionally highlights changes */
async function fetchOptionChain(scrollToATM = false, isRefresh = false) {
    const asset = document.querySelector(SELECTORS.assetDropdown)?.value;
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;

    // Use direct tbody selector as before
    const currentTbody = document.querySelector(SELECTORS.optionChainTableBody);
    if (!currentTbody) { // Check tbody directly
        logger.error("Option chain tbody element not found.");
        return;
    }

    // --- Initial Checks & Loading State ---
    if (!asset || !expiry) {
        currentTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">Select Asset and Expiry</td></tr>`; // Use placeholder class
        if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
        return;
    }
    if (!isRefresh) {
        setElementState(SELECTORS.optionChainTableBody, 'loading', 'Loading Chain...'); // Loading text is handled by setElementState
    }

    try {
        // --- Fetch Spot Price if Needed ---
        if (currentSpotPrice <= 0 && scrollToATM && !isRefresh) { // Added !isRefresh check
            logger.info("Spot price unavailable, fetching before option chain for ATM scroll...");
            try { await fetchNiftyPrice(asset); } catch (spotError) { logger.warn("Failed to fetch spot price for ATM calculation:", spotError.message); }
            if (currentSpotPrice <= 0) { logger.warn("Spot price still unavailable, cannot calculate ATM strike accurately."); scrollToATM = false; }
        }

        // --- Fetch Option Chain Data ---
        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);
        const currentChainData = data?.option_chain;

        // --- Handle Empty/Invalid Data ---
        if (!currentChainData || typeof currentChainData !== 'object' || currentChainData === null || Object.keys(currentChainData).length === 0) {
            logger.warn(`No option chain data available for ${asset} on ${expiry}.`); // Simplified log
            currentTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">No option chain data available for ${asset} on ${expiry}</td></tr>`; // Use placeholder class
            if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
            previousOptionChainData = {};
            return;
        }

        // --- Render Table & Handle Highlights ---
        const strikeStringKeys = Object.keys(currentChainData).sort((a, b) => Number(a) - Number(b));
        // Find the ATM strike STRING key (defined in this scope)
        const atmStrikeObjectKey = currentSpotPrice > 0 ? findATMStrikeAsStringKey(strikeStringKeys, currentSpotPrice) : null;

        currentTbody.innerHTML = ''; // Clear existing tbody

        // Iterate using the STRING keys
        strikeStringKeys.forEach((strikeStringKey) => {
            // ... (Keep the exact row rendering logic from your previous version) ...
            const optionDataForStrike = currentChainData[strikeStringKey];
            const optionData = (typeof optionDataForStrike === 'object' && optionDataForStrike !== null)
                                ? optionDataForStrike
                                : { call: null, put: null };
            const call = optionData.call || {};
            const put = optionData.put || {};
            const strikeNumericValue = Number(strikeStringKey);
            const prevOptionData = previousOptionChainData[strikeStringKey] || { call: {}, put: {} };
            const prevCall = prevOptionData.call || {};
            const prevPut = prevOptionData.put || {};
            const tr = document.createElement("tr");
            tr.dataset.strike = strikeNumericValue;
            if (atmStrikeObjectKey !== null && strikeStringKey === atmStrikeObjectKey) {
                tr.classList.add("atm-strike");
            }
            const columns = [
                { class: 'call clickable price', type: 'CE', key: 'last_price', format: val => formatNumber(val, 2, '-') },
                { class: 'call oi', key: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'call iv', key: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'strike', key: 'strike', isStrike: true, format: val => formatNumber(val, val % 1 === 0 ? 0 : 2) },
                { class: 'put iv', key: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'put oi', key: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'put clickable price', type: 'PE', key: 'last_price', format: val => formatNumber(val, 2, '-') },
            ];
            columns.forEach(col => {
                try {
                    const td = document.createElement('td');
                    td.className = col.class;
                    let currentValue;
                    if (col.isStrike) { currentValue = strikeNumericValue; }
                    else if (col.class.includes('call')) { currentValue = (typeof call === 'object' && call !== null) ? call[col.key] : undefined; }
                    else { currentValue = (typeof put === 'object' && put !== null) ? put[col.key] : undefined; }
                    td.textContent = col.format(currentValue);
                    if (col.type) {
                        td.dataset.type = col.type;
                        let sourceObj = col.class.includes('call') ? call : put;
                        if(typeof sourceObj === 'object' && sourceObj !== null) {
                            const ivValue = sourceObj['implied_volatility'];
                            const priceValue = sourceObj['last_price'];
                            if (ivValue !== null && ivValue !== undefined && !isNaN(parseFloat(ivValue))) { td.dataset.iv = ivValue; }
                             // Use ternary for price dataset, ensure it's a string '0' if invalid
                            td.dataset.price = (priceValue !== null && priceValue !== undefined && !isNaN(parseFloat(priceValue))) ? priceValue : '0';
                        } else { td.dataset.price = '0'; }
                    }
                     // Keep highlight logic exactly as you had it
                    if (isRefresh && !col.isStrike) {
                        let prevDataObject = col.class.includes('call') ? prevCall : prevPut;
                        if(typeof prevDataObject === 'object' && prevDataObject !== null) {
                            let previousValue = prevDataObject[col.key]; let changed = false;
                            const currentExists = currentValue !== null && typeof currentValue !== 'undefined';
                            const previousExists = previousValue !== null && typeof previousValue !== 'undefined';
                            if (currentExists && previousExists) { if (typeof currentValue === 'number' && typeof previousValue === 'number') { changed = Math.abs(currentValue - previousValue) > 0.001; } else { changed = currentValue !== previousValue; } }
                            else if (currentExists !== previousExists) { changed = true; }
                            if (changed) { highlightElement(td); }
                        } else if (currentValue !== null && typeof currentValue !== 'undefined'){ // Highlight if new value appears
                            highlightElement(td);
                        }
                    }
                    tr.appendChild(td);
                 } catch (cellError) {
                     logger.error(`Error rendering cell for Strike: ${strikeStringKey}, Column Key: ${col.key}`, cellError);
                     const errorTd = document.createElement('td'); errorTd.textContent = 'ERR'; errorTd.className = col.class + ' error-message'; tr.appendChild(errorTd);
                 }
            });
            currentTbody.appendChild(tr);
        }); // End strikes.forEach

        if (!isRefresh) {
            setElementState(SELECTORS.optionChainTableBody, 'content');
        }
        previousOptionChainData = currentChainData;

        // --- Scroll logic (Minimal Fix for Scope Error Applied HERE) ---
        if (scrollToATM && atmStrikeObjectKey !== null && !isRefresh) {
             // *** FIX: Pass the variable into the setTimeout callback ***
             setTimeout((atmKeyToUse) => { // Renamed parameter for clarity inside callback
                try {
                     // Use the passed 'atmKeyToUse' instead of the outer scope 'atmStrikeObjectKey'
                     const numericStrikeToFind = Number(atmKeyToUse);
                     if (isNaN(numericStrikeToFind)) {
                          logger.warn(`Invalid ATM key passed to scroll timeout: ${atmKeyToUse}`);
                          return;
                     }
                     logger.debug(`Scroll Timeout: Finding ATM row data-strike="${numericStrikeToFind}"`);
                     const atmRow = currentTbody.querySelector(`tr[data-strike="${numericStrikeToFind}"]`);
                     if (atmRow) {
                         atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
                         logger.debug(`Scrolled to ATM strike key: ${atmKeyToUse}`); // Log the key used
                     } else {
                          // Log using the key *and* the number it converted to
                          logger.warn(`ATM strike row for key (${atmKeyToUse} / ${numericStrikeToFind}) not found for scrolling.`);
                      }
                 } catch (e) {
                     logger.error("Error inside scroll timeout:", e);
                 }
             }, 250, atmStrikeObjectKey); // Pass the outer scope variable here as the 3rd argument
             // *** END FIX ***
         }

    } catch (error) { // Outer catch
        logger.error(`Error during fetchOptionChain execution for ${activeAsset}/${expiry}:`, error);
        if (currentTbody) {
            currentTbody.innerHTML = `<tr><td colspan="7" class="error-message">Chain Error: ${error.message}</td></tr>`;
        }
        if (!isRefresh) { setElementState(SELECTORS.optionChainTableBody, 'error', `Chain Error`); } // Show simple error state
        else { logger.warn(`Option Chain refresh failed: ${error.message}`); }
        previousOptionChainData = {};
    }
}

// ===============================================================
// Event Delegation Handlers
// ===============================================================
function findATMStrikeAsStringKey(strikeStringKeys = [], spotPrice) {
    // Assume logger is defined globally or passed as an argument if needed
    const logger = window.logger || window.console;

    if (!Array.isArray(strikeStringKeys) || strikeStringKeys.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         logger.warn("Cannot find ATM strike key: Invalid input.", { numKeys: strikeStringKeys?.length, spotPrice });
         return null; // Return null if input is invalid
    }

    let closestKey = null;
    let minDiff = Infinity;

    for (const key of strikeStringKeys) {
        // Convert string key to number for comparison
        const numericStrike = Number(key);
        if (!isNaN(numericStrike)) { // Ensure conversion is valid
            const diff = Math.abs(numericStrike - spotPrice);
            if (diff < minDiff) {
                minDiff = diff;
                closestKey = key; // Store the ORIGINAL STRING KEY
            }
        } else {
             logger.warn(`Skipping non-numeric strike key '${key}' during ATM calculation.`);
        }
    }

    // Log the result before returning
    logger.debug(`Calculated ATM strike key: ${closestKey} (Min diff: ${minDiff.toFixed(4)}) for spot price: ${spotPrice.toFixed(4)}`);
    return closestKey; // Return the STRING key
}


/** Handles clicks within the option chain table body */
function handleOptionChainClick(event) {
    const targetCell = event.target.closest('td.clickable');
    if (!targetCell) return;

    const row = targetCell.closest('tr');
    if (!row || !row.dataset.strike) return;

    const strike = parseFloat(row.dataset.strike);
    const type = targetCell.dataset.type; // 'CE' or 'PE'
    const price = parseFloat(targetCell.dataset.price); // Price from the cell's dataset
    const iv = parseFloat(targetCell.dataset.iv); // IV from the cell's dataset

    if (!isNaN(strike) && type && !isNaN(price)) {
         // Pass IV to addPosition, can be NaN if not found/parsed
         addPosition(strike, type, price, iv);
    } else {
        logger.warn('Could not add position - invalid data from clicked cell', { strike, type, price, iv });
        alert('Could not retrieve complete option details (strike, type, price). Please try again.');
    }
}

/** Handles clicks within the strategy table body (remove/toggle buttons) */
function handleStrategyTableClick(event) {
     // Check for Remove button click
     const removeButton = event.target.closest('button.remove-btn');
     if (removeButton?.dataset.index) { // Optional chaining
         const index = parseInt(removeButton.dataset.index, 10);
         if (!isNaN(index)) { removePosition(index); }
         return; // Stop further processing
     }

     // Check for Toggle Buy/Sell button click
     const toggleButton = event.target.closest('button.toggle-buy-sell');
     if (toggleButton?.dataset.index) { // Optional chaining
          const index = parseInt(toggleButton.dataset.index, 10);
         if (!isNaN(index)) { toggleBuySell(index); }
         return; // Stop further processing
     }
}

/** Handles input changes within the strategy table body (for lots input) */
function handleStrategyTableChange(event) {
    // Target input elements of type number with class lots-input
    if (event.target.matches('input[type="number"].lots-input') && event.target.dataset.index) {
        const index = parseInt(event.target.dataset.index, 10);
        if (!isNaN(index)) {
            // Use event.target.value directly
            updateLots(index, event.target.value);
        }
    }
}


// ===============================================================
// Strategy Management UI Logic
// ===============================================================

/** Adds a position (called by handleOptionChainClick) */
function addPosition(strike, type, price, iv) { // Added iv parameter
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) { alert("Please select an expiry date first."); return; }

    const lastPrice = (typeof price === 'number' && !isNaN(price)) ? price : 0;
    const impliedVol = (typeof iv === 'number' && !isNaN(iv) && iv > 0) ? iv : null; // Use fetched IV, null if invalid/missing
    const dte = calculateDaysToExpiry(expiry);

    if (impliedVol === null) {
        logger.warn(`Adding position ${type} ${strike} @ ${expiry} without valid IV (${iv}). Greeks calculation may fail for this leg.`);
        // Optionally alert user? alert(`Warning: Could not find Implied Volatility for ${type} ${strike}. Greeks may be inaccurate.`);
    }
     if (dte === null) {
        logger.error(`Could not calculate Days to Expiry for ${expiry}. Greeks calculation will fail.`);
        alert(`Error: Invalid expiry date ${expiry} provided.`);
        return; // Don't add position if DTE calculation fails
    }


    const newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type, // 'CE' or 'PE'
        lots: 1,           // Default to 1 lot (BUY)
        tr_type: 'b',      // Default to buy ('b')
        last_price: lastPrice,
        // Store IV and DTE needed for Greeks payload
        iv: impliedVol, // Store fetched or null IV
        days_to_expiry: dte, // Store calculated DTE
    };
    strategyPositions.push(newPosition);
    updateStrategyTable(); // Update UI
    logger.info("Added position:", newPosition);
    // Optional: Automatically update chart on add?
    // fetchPayoffChart();
}

/** Helper to get a specific value (like IV) from the currently rendered option chain table */
// DEPRECATED: IV is now passed directly from the clicked cell's dataset in handleOptionChainClick
/* function getOptionValueFromTable(strike, type, cellSelectorSuffix) { ... } */

/** Helper to calculate days to expiry from YYYY-MM-DD string */
function calculateDaysToExpiry(expiryDateStr) {
    try {
        // Ensure the date string is valid before creating a Date object
        if (!/^\d{4}-\d{2}-\d{2}$/.test(expiryDateStr)) {
             throw new Error("Invalid date format. Expected YYYY-MM-DD.");
        }

        // Create date objects at UTC midnight to avoid timezone issues affecting day difference
        const expiryDate = new Date(expiryDateStr + 'T00:00:00Z');
        const today = new Date();
        // Get today's date at UTC midnight
        const todayUTC = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));

         // Check if dates are valid after creation
        if (isNaN(expiryDate.getTime()) || isNaN(todayUTC.getTime())) {
            throw new Error("Could not parse dates.");
        }

        const diffTime = expiryDate - todayUTC; // Difference in milliseconds

        // Calculate days, rounding up. DTE=0 for today's expiry.
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        // Return 0 if expiry is in the past according to UTC midnight comparison
        return Math.max(0, diffDays);

    } catch (e) {
        logger.error("Error calculating days to expiry for", expiryDateStr, e);
        return null; // Indicate error
    }
}


/** Updates the strategy table in the UI */
function updateStrategyTable() {
    const tableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (!tableBody) return;
    tableBody.innerHTML = ""; // Clear previous rows

    if (strategyPositions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7">No positions added. Click option prices in the chain to add.</td></tr>';
        // Don't reset results here, let clearAllPositions or fetchPayoffChart handle it
        return;
    }

    strategyPositions.forEach((pos, index) => {
        // Determine buy/sell based on lots sign
        const isLong = pos.lots >= 0;
        pos.tr_type = isLong ? 'b' : 's'; // Update internal state if needed (though updateLots should handle this)
        const positionType = isLong ? "BUY" : "SELL";
        const positionClass = isLong ? "long-position" : "short-position";
        const buttonClass = isLong ? "button-buy" : "button-sell";

        const row = document.createElement("tr");
        row.className = positionClass;
        row.dataset.index = index; // Add index for easy access

        row.innerHTML = `
            <td>${pos.option_type}</td>
            <td>${formatNumber(pos.strike_price, 2)}</td>
            <td>${pos.expiry_date}</td>
            <td>
                <input type="number" value="${pos.lots}" data-index="${index}" min="-100" max="100" step="1" class="lots-input" aria-label="Lots for position ${index+1}">
            </td>
            <td>
                <button class="toggle-buy-sell ${buttonClass}" data-index="${index}" title="Click to switch between Buy and Sell">${positionType}</button>
            </td>
            <td>${formatCurrency(pos.last_price, 2)}</td>
            <td>
                <button class="remove-btn" data-index="${index}" aria-label="Remove position ${index+1}" title="Remove this leg">×</button>
            </td>
        `;
        tableBody.appendChild(row);
    });
}

/** Updates the number of lots for a position */
function updateLots(index, value) {
    if (index < 0 || index >= strategyPositions.length) return;

    const newLots = parseInt(value, 10);

    if (isNaN(newLots)) {
         // Optionally provide feedback to the user
         // alert("Please enter a valid integer for lots.");
         // Revert input value to the current state
         const inputElement = document.querySelector(`${SELECTORS.strategyTableBody} input.lots-input[data-index="${index}"]`);
         if(inputElement) inputElement.value = strategyPositions[index].lots;
         logger.warn(`Invalid lots input for index ${index}: "${value}"`);
         return;
    }

    if (newLots === 0) {
        logger.info(`Lots set to 0 for index ${index}, removing position.`);
        removePosition(index); // Remove position if lots become zero
    } else {
        const previousLots = strategyPositions[index].lots;
        strategyPositions[index].lots = newLots;
        strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's'; // Update buy/sell type

        // Update UI elements for the specific row for immediate feedback
        const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
        const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);

        if (row && toggleButton) {
             const isNowLong = newLots >= 0;
             const positionType = isNowLong ? "BUY" : "SELL";
             const buttonClass = isNowLong ? "button-buy" : "button-sell";

            toggleButton.textContent = positionType;
            toggleButton.classList.remove("button-buy", "button-sell");
            toggleButton.classList.add(buttonClass);
            row.className = isNowLong ? "long-position" : "short-position";
        } else {
            logger.warn(`Could not find row/button elements for index ${index} during lot update, doing full table refresh.`);
             updateStrategyTable(); // Fallback if specific elements not found
        }
        logger.info(`Updated lots for index ${index} from ${previousLots} to ${newLots}`);
    }
    // User must click "Update" to see calculation changes
}


/** Toggles a position between Buy and Sell */
function toggleBuySell(index) {
    if (index < 0 || index >= strategyPositions.length) return;

    const previousLots = strategyPositions[index].lots;
    let newLots = -previousLots; // Flip the sign

    // Handle case where flipping results in 0 (e.g., if it was 0) -> default to Buy 1
     if (newLots === 0) {
         newLots = 1;
         logger.info(`Lots were 0 for index ${index}, toggling to 1 (BUY).`);
     }

    strategyPositions[index].lots = newLots;
    strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's'; // Update buy/sell type

    logger.info(`Toggled Buy/Sell for index ${index}. Prev lots: ${previousLots}, New lots: ${newLots}`);

    // Update the specific row UI
    const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
    const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);
    const lotsInput = row?.querySelector(`input.lots-input[data-index="${index}"]`);

    if (row && toggleButton && lotsInput) {
        const isLong = newLots >= 0;
        const positionType = isLong ? "BUY" : "SELL";
        const buttonClass = isLong ? "button-buy" : "button-sell";

        toggleButton.textContent = positionType;
        toggleButton.classList.remove("button-buy", "button-sell");
        toggleButton.classList.add(buttonClass);
        row.className = isLong ? "long-position" : "short-position";
        lotsInput.value = newLots; // Update number input to reflect change
    } else {
        logger.warn(`Could not find row/button/input elements for index ${index} during toggle, doing full table refresh.`);
        updateStrategyTable(); // Fallback
    }
    // User must click "Update" to see calculation changes
}


/** Removes a position from the strategy */
function removePosition(index) {
    if (index < 0 || index >= strategyPositions.length) return;
    const removedPos = strategyPositions.splice(index, 1); // Remove and get the removed item
    logger.info("Removed position at index", index, removedPos[0]);
    updateStrategyTable(); // Update the table UI (indices will shift)

    // Update results only if there are remaining positions and chart isn't showing placeholder
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    const hasChartContent = chartContainer && !chartContainer.querySelector('.placeholder-text');

    if (strategyPositions.length > 0 && hasChartContent) {
        logger.info("Remaining positions exist, updating calculations...");
        fetchPayoffChart(); // Update results after removal
    } else if (strategyPositions.length === 0) {
        logger.info("Strategy is now empty, resetting results UI.");
        resetResultsUI(); // Clear results if strategy is now empty
    }
}


/** Clears all positions and resets UI */
function clearAllPositions() {
    if (strategyPositions.length === 0) return; // Do nothing if already empty
    if (confirm("Are you sure you want to clear all strategy legs?")) {
        logger.info("Clearing all positions...");
        strategyPositions = [];
        updateStrategyTable();
        resetResultsUI();
        // Don't stop auto-refresh here, asset/expiry are still selected
        logger.info("Strategy cleared.");
    }
}

function resetCalculationOutputsUI() {
     const logger = window.logger || window.console;
     logger.debug("Resetting calculation output UI elements...");

     // --- Reset Payoff Chart ---
     const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
     if (chartContainer) {
         if (typeof Plotly !== 'undefined' && chartContainer.layout) {
             try { Plotly.purge(chartContainer.id); } catch (e) { logger.warn("Plotly purge failed during reset:", e); }
         }
         chartContainer.innerHTML = '<div class="placeholder-text">Preparing calculation...</div>';
         setElementState(SELECTORS.payoffChartContainer, 'content');
     } else { logger.warn("Payoff chart container not found during output reset."); }

     // --- Reset Tax Container ---
     const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
     if (taxContainer) {
         taxContainer.innerHTML = '<p class="placeholder-text">Update strategy to calculate charges.</p>'; // Use placeholder
         setElementState(SELECTORS.taxInfoContainer, 'content');
     } else { logger.warn("Tax info container not found during output reset."); }

     // --- Reset Greeks Table ---
     const greeksTable = document.querySelector(SELECTORS.greeksTable);
     if (greeksTable) {
         const caption = greeksTable.querySelector('caption'); if (caption) caption.textContent = 'Portfolio Option Greeks';
         const greekBody = greeksTable.querySelector('tbody'); if (greekBody) greekBody.innerHTML = `<tr><td colspan="9" class="placeholder-text">Update strategy to calculate Greeks.</td></tr>`;
         const greekFoot = greeksTable.querySelector('tfoot'); if (greekFoot) greekFoot.innerHTML = "";
         setElementState(SELECTORS.greeksTable, 'content'); // Reset table state
         const greeksSection = document.querySelector(SELECTORS.greeksSection); if (greeksSection) setElementState(SELECTORS.greeksSection, 'content'); // Reset section state
     } else { logger.warn("Greeks table not found during output reset."); }

     // --- Reset Greeks Analysis Section ---
     const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection);
     const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
     if (greeksAnalysisSection) { setElementState(SELECTORS.greeksAnalysisSection, 'hidden'); } else { logger.warn("Greeks Analysis section not found during output reset."); }
     if (greeksAnalysisContainer) { greeksAnalysisContainer.innerHTML = ''; setElementState(SELECTORS.greeksAnalysisResultContainer, 'content'); } else { logger.warn("Greeks Analysis result container not found during output reset."); }

     // --- Reset Metrics Display ---
     displayMetric("N/A", SELECTORS.maxProfitDisplay); displayMetric("N/A", SELECTORS.maxLossDisplay); displayMetric("N/A", SELECTORS.breakevenDisplay); displayMetric("N/A", SELECTORS.rewardToRiskDisplay); displayMetric("N/A", SELECTORS.netPremiumDisplay);
     const metricsList = document.querySelector(SELECTORS.metricsList); if (metricsList) setElementState(SELECTORS.metricsList, 'content');

     // --- Reset Cost Breakdown ---
     const breakdownList = document.querySelector(SELECTORS.costBreakdownList); if (breakdownList) { breakdownList.innerHTML = ""; setElementState(SELECTORS.costBreakdownList, 'content'); }
     const detailsElement = document.querySelector(SELECTORS.costBreakdownContainer); if (detailsElement) { detailsElement.open = false; setElementState(SELECTORS.costBreakdownContainer, 'hidden'); detailsElement.style.display = 'none'; } else { logger.warn("Cost breakdown container not found during output reset."); }

     // --- Reset Warning Container ---
      const warningContainer = document.querySelector(SELECTORS.warningContainer); if (warningContainer) { warningContainer.textContent = ''; warningContainer.style.display = 'none'; setElementState(SELECTORS.warningContainer, 'hidden'); }

     // --- DO NOT Reset News or Stock Analysis Containers Here ---
     // --- DO NOT Clear strategyPositions or Strategy Table Here ---

     logger.debug("Calculation output UI elements reset complete.");
}

/** Resets the chart and results UI to initial state */
function resetResultsUI() {
    const logger = window.console; // Use console if no specific logger is set up
    logger.info("Resetting calculation output UI elements..."); // Updated log message

    // --- DO NOT Clear Strategy Data and Table HERE ---
    // The logic to clear strategyPositions and call updateStrategyTable
    // should be handled separately where needed (e.g., in handleAssetChange, clearAllPositions).

    // --- Reset Payoff Chart ---
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    if (chartContainer) {
        if (typeof Plotly !== 'undefined' && chartContainer.layout) {
            try {
                Plotly.purge(chartContainer.id);
                logger.debug("Purged Plotly chart container during reset.");
            } catch (e) {
                logger.warn("Failed to purge Plotly chart during reset:", e);
            }
        }
        // Set placeholder text
        chartContainer.innerHTML = '<div class="placeholder-text">Add positions and click "Update" to see the payoff chart.</div>'; // Placeholder reflects need for action
        setElementState(SELECTORS.payoffChartContainer, 'content'); // Reset state
    } else {
        logger.warn("Payoff chart container not found during reset.");
    }

    // --- Reset Tax Container ---
    const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
    if (taxContainer) {
        // Set placeholder text
        taxContainer.innerHTML = '<p class="placeholder-text">Update strategy to calculate charges.</p>';
        setElementState(SELECTORS.taxInfoContainer, 'content'); // Reset state
    } else {
        logger.warn("Tax info container not found during reset.");
    }

    // --- Reset Greeks Table ---
    const greeksTable = document.querySelector(SELECTORS.greeksTable);
    if (greeksTable) {
        const caption = greeksTable.querySelector('caption');
        if (caption) caption.textContent = 'Portfolio Option Greeks'; // Reset caption

        const greekBody = greeksTable.querySelector('tbody');
        if (greekBody) greekBody.innerHTML = `<tr><td colspan="9" class="placeholder-text">Update strategy to calculate Greeks.</td></tr>`; // Placeholder

        const greekFoot = greeksTable.querySelector('tfoot');
        if (greekFoot) greekFoot.innerHTML = ""; // Clear footer

        setElementState(SELECTORS.greeksTable, 'content'); // Reset state
        const greeksSection = document.querySelector(SELECTORS.greeksSection);
        if (greeksSection) setElementState(SELECTORS.greeksSection, 'content');
    } else {
        logger.warn("Greeks table not found during reset.");
    }

    // --- Reset Cost Breakdown ---
    const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
    if (breakdownList) {
        breakdownList.innerHTML = ""; // Clear list items
        setElementState(SELECTORS.costBreakdownList, 'content');
    } else {
        logger.warn("Cost breakdown list not found during reset.");
    }
    const detailsElement = document.querySelector(SELECTORS.costBreakdownContainer);
    if (detailsElement) {
        detailsElement.open = false;
        setElementState(SELECTORS.costBreakdownContainer, 'hidden');
        detailsElement.style.display = 'none';
    } else {
         logger.warn("Cost breakdown container not found during reset.");
    }

    // --- Reset Metrics Display ---
    // Use N/A as the default reset state for metrics
    displayMetric("N/A", SELECTORS.maxProfitDisplay);
    displayMetric("N/A", SELECTORS.maxLossDisplay);
    displayMetric("N/A", SELECTORS.breakevenDisplay);
    displayMetric("N/A", SELECTORS.rewardToRiskDisplay);
    displayMetric("N/A", SELECTORS.netPremiumDisplay);
    const metricsList = document.querySelector(SELECTORS.metricsList);
    if (metricsList) setElementState(SELECTORS.metricsList, 'content');

    // --- Reset News Container ---
    // This function should focus on calculation results, so we *don't* reset news here.
    // News container reset should happen in handleAssetChange or initializePage.
    // const newsContainer = document.querySelector(SELECTORS.newsResultContainer);
    // if (newsContainer) { /* ... */ }

    // --- Reset Stock Analysis Container ---
    // This function should focus on calculation results, so we *don't* reset analysis here.
    // Analysis container reset should happen in handleAssetChange or initializePage.
    // const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
    // if (analysisContainer) { /* ... */ }

    // --- Reset Greeks Analysis Section ---
    const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection);
    const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
    if (greeksAnalysisSection) {
        setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
        logger.debug("Reset Greeks Analysis Section to hidden.");
    } else {
        logger.warn("Greeks Analysis section not found during reset.");
    }
    if (greeksAnalysisContainer) {
         greeksAnalysisContainer.innerHTML = ''; // Clear content
         setElementState(SELECTORS.greeksAnalysisResultContainer, 'content');
         logger.debug("Reset Greeks Analysis Container content.");
    } else {
        logger.warn("Greeks Analysis result container not found during reset.");
    }

    // --- Reset Status/Warning Message Container ---
    const messageContainer = document.querySelector(SELECTORS.statusMessageContainer);
     if (messageContainer) {
          messageContainer.textContent = '';
          messageContainer.className = 'status-message';
          setElementState(messageContainer, 'hidden');
     }
     const warningContainer = document.querySelector(SELECTORS.warningContainer);
     if (warningContainer) {
          warningContainer.textContent = '';
          warningContainer.style.display = 'none';
          setElementState(warningContainer, 'hidden');
     }

     logger.info("Calculation Results UI elements have been reset."); // Updated log
}

// --- Define necessary variables and functions assumed by resetResultsUI ---


function renderCostBreakdown(listElement, costBreakdownData) {
    // Ensure logger and formatCurrency are available
    const logger = window.logger || window.console;
    const localFormatCurrency = window.formatCurrency || ((val) => `₹${Number(val).toFixed(2)}`); // Basic fallback

    if (!listElement) {
        logger.error("renderCostBreakdown: Target list element is null or undefined.");
        return;
    }
    if (!Array.isArray(costBreakdownData)) {
        logger.error("renderCostBreakdown: costBreakdownData is not a valid array.");
        listElement.innerHTML = '<li>Error displaying breakdown data.</li>';
        return;
    }

    // Clear previous list items
    listElement.innerHTML = "";

    if (costBreakdownData.length === 0) {
        listElement.innerHTML = '<li>No cost breakdown details available.</li>';
        return;
    }

    // Populate the list
    costBreakdownData.forEach((item, index) => {
        try {
            const li = document.createElement("li");

            // Safely access properties with fallbacks
            const action = item.action || 'N/A';
            const lots = item.lots !== undefined ? item.lots : '?';
            const quantity = item.quantity !== undefined ? item.quantity : '?';
            const type = item.type || 'N/A';
            const strike = item.strike !== undefined ? item.strike : '?';
            const premiumPerShare = item.premium_per_share !== undefined ? localFormatCurrency(item.premium_per_share) : 'N/A';
            const totalPremium = item.total_premium !== undefined ? Math.abs(item.total_premium) : null; // Use absolute value for display effect
            const effect = item.effect || 'N/A'; // 'Paid' or 'Received'

            let premiumEffectText = 'N/A';
            if (totalPremium !== null) {
                 premiumEffectText = effect === 'Paid' ? `Paid ${localFormatCurrency(totalPremium)}` : `Received ${localFormatCurrency(totalPremium)}`;
            }

            // Construct the text content for the list item
            li.textContent = `${action} ${lots} Lot(s) [${quantity} Qty] ${type} ${strike} @ ${premiumPerShare} (${premiumEffectText} Total)`;

            listElement.appendChild(li);
        } catch (e) {
            logger.error(`Error rendering breakdown item ${index}:`, e, item);
            const errorLi = document.createElement("li");
            errorLi.textContent = `Error processing breakdown leg ${index + 1}.`;
            errorLi.style.color = 'red';
            listElement.appendChild(errorLi);
        }
    });
}


// ===============================================================
// Payoff Chart & Results Logic
// ===============================================================

/** Fetches payoff chart data, metrics, taxes, greeks and triggers rendering */
async function fetchPayoffChart() {
    const logger = window.logger || window.console; // Use console if no specific logger is set up
    const updateButton = document.querySelector(SELECTORS.updateChartButton);

    logger.info("--- [fetchPayoffChart] START ---"); // Debug Start

    // 1. --- Get Asset and Strategy Legs FIRST ---
    const asset = activeAsset; // Use the globally tracked activeAsset
    logger.debug("[fetchPayoffChart] Step 1: Get Asset. Value:", asset); // Log asset value
    if (!asset) {
        alert("Error: No asset is currently selected.");
        logger.error("[fetchPayoffChart] Aborted: No active asset."); // Debug Log
        return;
    }

    // Gather strategy legs from the *current state* using the helper
    logger.debug("[fetchPayoffChart] Step 1b: Gathering strategy legs...");
    const currentStrategyLegs = gatherStrategyLegsFromTable(); // Assumes this reads UI/state
    // *** DEBUG: Log the raw gathered legs ***
    logger.debug("[fetchPayoffChart] Step 1c: Gathered legs raw data:", JSON.parse(JSON.stringify(currentStrategyLegs))); // Deep copy log

    if (!currentStrategyLegs || currentStrategyLegs.length === 0) {
        alert("Please add positions to the strategy first.");
        logger.warn("[fetchPayoffChart] Aborted: No strategy legs gathered or returned empty array."); // Debug Log
        resetCalculationOutputsUI(); // Reset only outputs if trying to calc empty strategy
        return;
    }
    logger.debug(`[fetchPayoffChart] Found ${currentStrategyLegs.length} legs in gathered data.`); // Changed log slightly

    // Check for chart container *early*
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    if (!chartContainer) {
        logger.error("[fetchPayoffChart] Aborted: Payoff chart container element not found."); // Debug Log
        return; // Stop execution if container is missing
    }

    // 2. --- Set Loading States & Reset OUTPUT UI ---
    logger.debug("[fetchPayoffChart] Step 2: Resetting outputs and setting loading states..."); // Debug Log
    resetCalculationOutputsUI(); // <<< Call the OUTPUTS ONLY reset function here
    setElementState(SELECTORS.payoffChartContainer, 'loading', 'Generating chart data...');
    setElementState(SELECTORS.taxInfoContainer, 'loading', 'Calculating charges...');
    setElementState(SELECTORS.greeksSection, 'loading', 'Calculating Greeks...');
    setElementState(SELECTORS.greeksTable, 'loading');
    setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
    setElementState(SELECTORS.metricsList, 'loading');
    setElementState(SELECTORS.costBreakdownContainer, 'hidden');
    if (updateButton) updateButton.disabled = true;


    // 3. --- Prepare Request Data (Map and Validate gathered legs) ---
    logger.debug("[fetchPayoffChart] Step 3: Mapping and validating gathered legs..."); // Debug Log
    let dataIsValid = true; // Flag to track overall validity
    let mappingErrors = 0; // Count legs failing validation
    const requestStrategy = currentStrategyLegs.map((pos, index) => {
        const legNum = index + 1;
        logger.debug(`[fetchPayoffChart] Mapping leg ${legNum}:`, JSON.parse(JSON.stringify(pos))); // Log raw leg data from gathered array

        // Use the DTE already calculated in gatherStrategyLegsFromTable
        const currentDTE = pos.days_to_expiry; // Assumes gather func added this
        let legValidationMessages = [];

        // --- Perform Local Validation ---
        // Validate DTE existence first (should be added by gather function)
        if (currentDTE === null || currentDTE === undefined || isNaN(parseInt(currentDTE)) || parseInt(currentDTE) < 0) {
            legValidationMessages.push(`Invalid or missing DTE (${currentDTE})`); dataIsValid = false;
        }
        // Check other required fields (use || for checks, not &&)
        if (!pos.strike_price || isNaN(parseFloat(pos.strike_price)) || parseFloat(pos.strike_price) <= 0) { legValidationMessages.push(`Invalid strike (${pos.strike_price})`); dataIsValid = false; }
        if (pos.last_price === null || pos.last_price === undefined || isNaN(parseFloat(pos.last_price)) || parseFloat(pos.last_price) < 0) { legValidationMessages.push(`Invalid premium (${pos.last_price})`); dataIsValid = false; }
        if (!pos.lots || isNaN(parseInt(pos.lots)) || parseInt(pos.lots) === 0) { legValidationMessages.push(`Invalid lots (${pos.lots})`); dataIsValid = false; }
        // IV is optional, just log warning if missing/invalid
        if (pos.iv === null || pos.iv === undefined || pos.iv === '' || isNaN(parseFloat(pos.iv))) { logger.warn(`[fetchPayoffChart] Leg ${legNum}: Missing/invalid IV.`); }
        // Lot size is optional but validate if provided
        if (pos.lot_size && (isNaN(parseInt(pos.lot_size)) || parseInt(pos.lot_size) <= 0)) { logger.warn(`[fetchPayoffChart] Leg ${legNum}: Invalid explicit lot size (${pos.lot_size}).`); }

        // Determine backend op_type ('c' or 'p')
        let backendOpType = '';
        if (pos.option_type && typeof pos.option_type === 'string') {
            const upperOptType = pos.option_type.toUpperCase();
            if (upperOptType === 'CE') { backendOpType = 'c'; }
            else if (upperOptType === 'PE') { backendOpType = 'p'; }
            else { legValidationMessages.push(`Invalid option_type (${pos.option_type})`); dataIsValid = false; }
        } else {
            legValidationMessages.push(`Missing/invalid option_type`); dataIsValid = false;
        }
        // --- End Local Validation ---

        // If leg failed validation, log and return null
        if (legValidationMessages.length > 0) {
            logger.error(`[fetchPayoffChart] Leg ${legNum} failed validation: ${legValidationMessages.join('; ')}`);
            mappingErrors++;
            // Ensure dataIsValid flag stays false if it was already set
            dataIsValid = false;
            return null; // Mark leg as invalid for filtering
        }

        // Return object matching backend Pydantic model if valid
        const mappedLeg = {
            op_type: backendOpType,
            strike: String(pos.strike_price),
            tr_type: parseInt(pos.lots) >= 0 ? "b" : "s",
            op_pr: String(pos.last_price),
            lot: String(Math.abs(parseInt(pos.lots))),
            lot_size: pos.lot_size ? String(pos.lot_size) : null,
            iv: (pos.iv !== null && pos.iv !== undefined && pos.iv !== '' && !isNaN(parseFloat(pos.iv))) ? parseFloat(pos.iv) : null,
            days_to_expiry: currentDTE,
            expiry_date: pos.expiry_date,
        };
        logger.debug(`[fetchPayoffChart] Leg ${legNum} mapped successfully:`, mappedLeg);
        return mappedLeg;

    }).filter(leg => leg !== null); // Filter out legs that failed validation

    logger.debug(`[fetchPayoffChart] Mapping complete. ${requestStrategy.length} valid legs found, ${mappingErrors} legs failed validation.`);

    // Abort if dataIsValid flag is false OR if filtering resulted in an empty array
    if (!dataIsValid || requestStrategy.length === 0) {
        const errorReason = !dataIsValid ? `Invalid data found in ${mappingErrors} leg(s)` : "No valid legs remained after mapping/filtering";
        alert(`Error: ${errorReason}. Please check console for details or correct the strategy.`);
        logger.error(`[fetchPayoffChart] Aborted before API call. Reason: ${errorReason}.`);
        resetCalculationOutputsUI(); // Reset results UI only
        if (updateButton) updateButton.disabled = false;
        return;
    }

    // Prepare final request payload
    const requestData = { asset: asset, strategy: requestStrategy }; // Use the filtered, validated legs
    // *** DEBUG: Log the final payload EXACTLY as it will be sent ***
    logger.debug("[fetchPayoffChart] Step 4: Final requestData object:", JSON.parse(JSON.stringify(requestData)));
    logger.debug("[fetchPayoffChart] Step 4b: Body to be sent:", JSON.stringify(requestData));


    // --- Fetch API and Handle Response ---
    logger.debug("[fetchPayoffChart] Step 5: Calling fetchAPI('/get_payoff_chart')...");
    try {
        const data = await fetchAPI('/get_payoff_chart', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData) // Send the correctly built data
        });

        // *** DEBUG: Log the raw response data ***
        logger.debug("[fetchPayoffChart] Step 6: Received response data from /get_payoff_chart:", JSON.parse(JSON.stringify(data))); // Deep copy log

        // 5. Validate Backend Response
        if (!data || !data.success) {
            const errorMessage = data?.message || "Calculation failed on the server.";
            logger.error(`[fetchPayoffChart] Backend reported failure: ${errorMessage}`); // Log backend failure message
            resetCalculationOutputsUI(); // Reset outputs to clear loading
            setElementState(SELECTORS.payoffChartContainer, 'error', `Chart Error: ${errorMessage}`);
            setElementState(SELECTORS.taxInfoContainer, 'error', 'Calculation Failed');
            setElementState(SELECTORS.greeksSection, 'error', 'Calculation Failed');
            setElementState(SELECTORS.greeksTable, 'error');
            setElementState(SELECTORS.metricsList, 'error');
            setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
            // Show global error for calculation failures reported by backend
            setElementState(SELECTORS.globalErrorDisplay, 'error', `Calculation Error: ${errorMessage}`);
            return; // Stop processing on backend failure
        }
        logger.debug("[fetchPayoffChart] Backend response indicates success=true."); // Log success

        // 6. --- Render Results ---
        logger.debug("[fetchPayoffChart] Step 7: Rendering results...");

        // --- DEBUG: Log data sections before rendering ---
        logger.debug("[fetchPayoffChart] Metrics data:", data?.metrics);
        logger.debug("[fetchPayoffChart] Charges data:", data?.charges);
        logger.debug("[fetchPayoffChart] Greeks data:", data?.greeks);
        logger.debug("[fetchPayoffChart] Chart JSON present:", !!data?.chart_figure_json);
        // --- End DEBUG ---

        // ... (Keep the rest of the rendering logic as before) ...
         // Metrics
         const metricsContainer = data?.metrics;
         const metricsData = metricsContainer?.metrics;
         if (metricsData) { /* ... render metrics ... */ setElementState(SELECTORS.metricsList, 'content');}
         else { logger.warn("[fetchPayoffChart] Metrics data missing from successful response."); /* ... handle missing metrics ... */ setElementState(SELECTORS.metricsList, 'content');}

         // Cost Breakdown
         const costBreakdownData = metricsContainer?.cost_breakdown_per_leg;
         const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
         const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
         if (breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) { renderCostBreakdown(breakdownList, costBreakdownData); setElementState(SELECTORS.costBreakdownContainer, 'content'); breakdownContainer.style.display = ""; breakdownContainer.open = false;}
         else if (breakdownContainer) { logger.warn("[fetchPayoffChart] Cost breakdown data missing or empty."); setElementState(SELECTORS.costBreakdownContainer, 'hidden'); }

         // Tax Table
         const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
         if (taxContainer) { if (data?.charges) { renderTaxTable(taxContainer, data.charges); setElementState(SELECTORS.taxInfoContainer, 'content'); } else { logger.warn("[fetchPayoffChart] Charges data missing from successful response."); taxContainer.innerHTML = "<p class='text-muted'>Charge data unavailable.</p>"; setElementState(SELECTORS.taxInfoContainer, 'content'); } }

         // Chart
         const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
         const chartDataKey = "chart_figure_json";
         if (chartContainer && data[chartDataKey]) { renderPayoffChart(chartContainer, data[chartDataKey]); setElementState(SELECTORS.payoffChartContainer, 'content'); }
         else if (chartContainer) { logger.warn("[fetchPayoffChart] Chart JSON missing from successful response."); chartContainer.innerHTML = '<div class="placeholder-text">Chart could not be generated.</div>'; setElementState(SELECTORS.payoffChartContainer, 'error'); }

         // Greeks Table & Analysis Trigger
         const greeksTableElement = document.querySelector(SELECTORS.greeksTable);
         const greeksSectionElement = document.querySelector(SELECTORS.greeksSection);
         if (greeksTableElement && greeksSectionElement && data.greeks && Array.isArray(data.greeks)) {
             logger.debug("[fetchPayoffChart] Rendering Greeks table...");
             const calculatedTotals = renderGreeksTable(greeksTableElement, data.greeks); // Capture totals
             setElementState(greeksSectionElement, 'content');
             if (calculatedTotals) {
                 logger.info("[fetchPayoffChart] Greeks totals calculated, fetching Greeks analysis...", calculatedTotals);
                 fetchAndDisplayGreeksAnalysis(asset, calculatedTotals); // Trigger analysis
             } else {
                 logger.warn("[fetchPayoffChart] Greeks table rendered, but totals were null/invalid. Skipping Greeks analysis fetch.");
                 const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer); if(greeksAnalysisSection && greeksAnalysisContainer) { setElementState(greeksAnalysisSection, 'content'); greeksAnalysisContainer.innerHTML = '<p>Could not calculate portfolio totals needed for Greeks analysis.</p>'; setElementState(greeksAnalysisContainer, 'content'); }
             }
         } else {
             logger.warn("[fetchPayoffChart] Greeks data missing/invalid or table element not found. Skipping Greeks table/analysis.");
             if (greeksSectionElement) { greeksSectionElement.innerHTML = '<h3 class="section-subheader">Options Greeks</h3><p>Greeks data not available.</p>'; setElementState(greeksSectionElement, 'content'); }
             const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); if(greeksAnalysisSection) setElementState(greeksAnalysisSection, 'hidden');
         }
         logger.debug("[fetchPayoffChart] Step 7: Rendering complete.");

    } catch (error) { // Catch errors from fetchAPI or rendering failures AFTER successful fetch
        logger.error(`[fetchPayoffChart] Fatal error during API call or result processing: ${error.message}`, error); // Log full error
        resetCalculationOutputsUI(); // Reset results UI fully on error
        let errorMsg = `Error: ${error.message || 'Failed to process calculation result.'}`;
        // Update based on common error types caught by fetchAPI or thrown locally
        if (error.message.includes("Invalid JSON response")) { errorMsg = "Error: Invalid response from server."; }
        else if (error.message.includes("Failed to fetch")) { errorMsg = "Network Error: Could not reach calculation server."; }
        // Display specific errors if needed, otherwise generic
        setElementState(SELECTORS.payoffChartContainer, 'error', errorMsg);
        setElementState(SELECTORS.taxInfoContainer, 'error', 'Error');
        setElementState(SELECTORS.greeksSection, 'error', 'Error');
        setElementState(SELECTORS.greeksTable, 'error');
        setElementState(SELECTORS.metricsList, 'error');
        setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Calculation Error: ${error.message || 'Unknown processing error'}`);

    } finally {
        if (updateButton) updateButton.disabled = false; // Always re-enable button
        logger.info("--- [fetchPayoffChart] END ---"); // Mark the end
    }
}

// --- Rendering Helpers for Payoff Results ---

function renderTaxTable(containerElement, taxData) {
    // Assume logger, formatCurrency, formatNumber are defined globally or imported
    const logger = window.console; // Use console if no specific logger

    // Guard against null/undefined taxData or missing nested properties
    if (!taxData || !taxData.breakdown_per_leg || !taxData.charges_summary || !Array.isArray(taxData.breakdown_per_leg)) {
        containerElement.innerHTML = '<p class="error-message">Charge calculation data is incomplete or unavailable.</p>';
        logger.warn("renderTaxTable called with invalid or incomplete taxData:", taxData);
        return;
    }

    containerElement.innerHTML = ""; // Clear previous content

    const details = document.createElement('details');
    details.className = "results-details tax-details";
    details.open = false; // Default closed

    const summary = document.createElement('summary');
    // Use formatCurrency for the total in the summary for consistency
    summary.innerHTML = `<strong>Estimated Charges Breakdown (Total: ${formatCurrency(taxData.total_estimated_cost, 2)})</strong>`;
    details.appendChild(summary);

    const tableWrapper = document.createElement('div');
    tableWrapper.className = 'table-wrapper thin-scrollbar'; // Add scrollbar class
    details.appendChild(tableWrapper);

    const table = document.createElement("table");
    table.className = "results-table charges-table data-table"; // Keep consistent classes
    const charges = taxData.charges_summary || {};
    const breakdown = taxData.breakdown_per_leg;

    // --- Generate Table Body with Mapped Values ---
    const tableBody = breakdown.map(t => {
        // Map Transaction Type ('B'/'S' to 'BUY'/'SELL')
        let actionDisplay = '?'; // Default placeholder
        const actionRaw = (t.transaction_type || '').toUpperCase();
        if (actionRaw === 'B') {
            actionDisplay = 'BUY';
        } else if (actionRaw === 'S') {
            actionDisplay = 'SELL';
        }

        // Map Option Type ('C'/'P' to 'CE'/'PE')
        let typeDisplay = '?'; // Default placeholder
        const typeRaw = (t.option_type || '').toUpperCase();
        if (typeRaw === 'C') {
            typeDisplay = 'CE';
        } else if (typeRaw === 'P') {
            typeDisplay = 'PE';
        }

        // Ensure all expected keys exist in breakdown items, providing defaults
        // Use the mapped display values in the first two columns
        return `
        <tr>
            <td>${actionDisplay}</td>
            <td>${typeDisplay}</td>
            <td>${formatNumber(t.strike, 2, '-')}</td>
            <td>${formatNumber(t.lots, 0, '-')}</td>
            <td>${formatNumber(t.premium_per_share, 2, '-')}</td>
            <td>${formatNumber(t.stt, 2, '0.00')}</td>
            <td>${formatNumber(t.stamp_duty, 2, '0.00')}</td>
            <td>${formatNumber(t.sebi_fee, 4, '0.0000')}</td>
            <td>${formatNumber(t.txn_charge, 4, '0.0000')}</td>
            <td>${formatNumber(t.brokerage, 2, '0.00')}</td>
            <td>${formatNumber(t.gst, 2, '0.00')}</td>
            <td class="note" title="${t.stt_note || ''}">${((t.stt_note || '').substring(0, 15))}${ (t.stt_note || '').length > 15 ? '...' : ''}</td>
        </tr>`;
    }).join('');

    // --- Prepare Footer Totals ---
    // Use nullish coalescing (??) for safer defaults
    const total_stt = charges.stt ?? 0;
    const total_stamp = charges.stamp_duty ?? 0;
    const total_sebi = charges.sebi_fee ?? 0;
    const total_txn = charges.txn_charges ?? 0; // Match backend key if it's 'txn_charges'
    const total_brokerage = charges.brokerage ?? 0;
    const total_gst = charges.gst ?? 0;
    const overall_total = taxData.total_estimated_cost ?? 0;

    // --- Assemble Table HTML ---
    // Header column count = 12
    table.innerHTML = `
        <thead>
            <tr>
                <th>Act</th>
                <th>Type</th>
                <th>Strike</th>
                <th>Lots</th>
                <th>Premium</th>
                <th>STT</th>
                <th>Stamp</th>
                <th>SEBI</th>
                <th>Txn</th>
                <th>Broker</th>
                <th>GST</th>
                <th title="Securities Transaction Tax Note">STT Note</th>
            </tr>
        </thead>
        <tbody>${tableBody}</tbody>
        <tfoot>
            <tr class="totals-row">
                <td colspan="5" style="text-align: right; font-weight: bold;">Total Estimated Charges</td>
                <td>${formatCurrency(total_stt, 2)}</td>
                <td>${formatCurrency(total_stamp, 2)}</td>
                <td>${formatCurrency(total_sebi, 4)}</td>
                <td>${formatCurrency(total_txn, 4)}</td>
                <td>${formatCurrency(total_brokerage, 2)}</td>
                <td>${formatCurrency(total_gst, 2)}</td>
                <td style="font-weight: bold;">${formatCurrency(overall_total, 2)}</td>
            </tr>
        </tfoot>`;

    tableWrapper.appendChild(table);
    containerElement.appendChild(details);
}


/**
 * Renders the Greeks table based on the list provided by the backend.
 * Calculates portfolio totals on the frontend.
 * @param {HTMLElement} tableElement - The table element to render into.
 * @param {Array | null | undefined} greeksList - The list of Greek results per leg like: [{leg_index: N, input_data: {...}, calculated_greeks_per_share: {...}}, ...].
 */
function renderGreeksTable(tableElement, greeksList) {
    const logger = window.console; // Use standard console if no custom logger
    tableElement.innerHTML = ''; // Clear previous content (header, body, footer)

    // Ensure tableElement is valid
    if (!tableElement || !(tableElement instanceof HTMLTableElement)) {
        logger.error("renderGreeksTable: Invalid tableElement provided.");
        return null;
    }

    const caption = tableElement.createCaption();
    caption.className = "table-caption";
    caption.textContent = "Portfolio Option Greeks";

    // Initialize totals
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    let hasCalculatedGreeks = false; // Flag to check if any totals were calculated

    // --- Input Validation ---
    if (!Array.isArray(greeksList)) {
        logger.error("renderGreeksTable: Input greeksList is not an array.");
        caption.textContent = "Error: Invalid Greeks data format received.";
        // Use the CORRECTED setElementState call with a selector string
        setElementState(SELECTORS.greeksTable, 'error'); // Pass selector string
        return null; // Return null indicating failure
    }

    const totalLegsInput = greeksList.length;
    if (totalLegsInput === 0) {
        caption.textContent = "Portfolio Option Greeks (No legs in strategy)";
        const tbody = tableElement.createTBody();
        // Use placeholder class for consistency
        tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">No option legs found in the strategy.</td></tr>`;
        // Use the CORRECTED setElementState call with a selector string
        setElementState(SELECTORS.greeksTable, 'content'); // Pass selector string
        return totals; // Return zeroed totals
    }

    caption.textContent = `Portfolio Option Greeks (${totalLegsInput} Leg${totalLegsInput > 1 ? 's' : ''} Input)`;

    // --- Create Table Header ---
    const thead = tableElement.createTHead();
    // Header shows per-share context in tooltips
    thead.innerHTML = `
        <tr>
            <th>Action</th>
            <th>Lots</th>
            <th>Type</th>
            <th>Strike</th>
            <th title="Delta per Share">Δ Delta</th>
            <th title="Gamma per Share">Γ Gamma</th>
            <th title="Theta per Share (per Day)">Θ Theta</th>
            <th title="Vega per Share (per 1% IV)">Vega</th>
            <th title="Rho per Share (per 1% Rate)">Ρ Rho</th>
        </tr>`;

    // --- Populate Table Body ---
    const tbody = tableElement.createTBody();
    let skippedLegsCount = 0;
    let processedLegsCount = 0;

    greeksList.forEach((g, index) => {
        const row = tbody.insertRow();
        const inputData = g?.input_data;
        const gv_per_share = g?.calculated_greeks_per_share; // Greeks per share for row display

        // --- Validate data for this leg ---
        if (!inputData || typeof inputData !== 'object' || !gv_per_share || typeof gv_per_share !== 'object') {
             logger.warn(`renderGreeksTable: Malformed data structure for leg ${index + 1}. Skipping.`);
             skippedLegsCount++;
             // Add a row indicating skipped leg
             row.innerHTML = `<td colspan="9" style="font-style: italic; color: #888;">Leg ${index + 1}: Invalid data received</td>`;
             row.classList.add('skipped-leg'); // Add class for potential styling
             return; // Skip to next iteration
        }

        // --- Extract and Format Leg Details ---
        const actionDisplay = (inputData.tr_type === 'b') ? 'BUY' : (inputData.tr_type === 's' ? 'SELL' : '?');
        const typeDisplay = (inputData.op_type === 'c') ? 'CE' : (inputData.op_type === 'p' ? 'PE' : '?');
        // Ensure lots and lotSize are parsed correctly as integers
        const lots = parseInt(inputData.lots || '0', 10);
        const lotSize = parseInt(inputData.lot_size || '0', 10); // Needed for totals calculation
        const strike = inputData.strike; // Keep as number for formatting
        const lotsDisplay = (lots > 0) ? `${lots}` : 'N/A';

        // --- Fill Table Cells (Displaying Per-Share Greeks) ---
        // Assume formatNumber helper exists: formatNumber(value, precision, fallback = 'N/A')
        row.insertCell().textContent = actionDisplay;
        row.insertCell().textContent = lotsDisplay;
        row.insertCell().textContent = typeDisplay;
        row.insertCell().textContent = formatNumber(strike, 2);
        row.insertCell().textContent = formatNumber(gv_per_share.delta, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.gamma, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.theta, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.vega, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.rho, 4, '-');

        // --- Accumulate PORTFOLIO Totals (Using Per-Share * Quantity) ---
        // Check if all required values are valid numbers for calculation
        const isValidForTotal = lots > 0 && lotSize > 0 &&
                                typeof gv_per_share.delta === 'number' && isFinite(gv_per_share.delta) &&
                                typeof gv_per_share.gamma === 'number' && isFinite(gv_per_share.gamma) &&
                                typeof gv_per_share.theta === 'number' && isFinite(gv_per_share.theta) &&
                                typeof gv_per_share.vega === 'number' && isFinite(gv_per_share.vega) &&
                                typeof gv_per_share.rho === 'number' && isFinite(gv_per_share.rho);

        if (isValidForTotal) {
            const quantity = lots * lotSize; // Total quantity (shares) for the leg
            // Accumulate totals
            totals.delta += gv_per_share.delta * quantity;
            totals.gamma += gv_per_share.gamma * quantity;
            totals.theta += gv_per_share.theta * quantity;
            totals.vega += gv_per_share.vega * quantity;
            totals.rho += gv_per_share.rho * quantity;
            hasCalculatedGreeks = true; // Mark that totals were successfully updated at least once
            processedLegsCount++;
            row.classList.add('greeks-calculated'); // Optional styling for valid rows
        } else {
            // Log why calculation was skipped for this leg
            logger.warn(`renderGreeksTable: Skipping leg ${index + 1} from total calculation due to invalid data (lots=${lots}, lotSize=${lotSize}, delta=${gv_per_share.delta}, etc.)`);
            skippedLegsCount++;
            row.classList.add('greeks-skipped'); // Optional styling for skipped rows
            // You might want to visually indicate these rows differently (e.g., greyed out slightly)
            row.style.opacity = '0.6';
            row.style.fontStyle = 'italic';
        }
    }); // End forEach leg

    // Update caption with processed/skipped count
    caption.textContent = `Portfolio Option Greeks (${processedLegsCount} Leg${processedLegsCount !== 1 ? 's' : ''} Processed, ${skippedLegsCount} Skipped)`;

    // --- Create Table Footer with Totals ---
    const tfoot = tableElement.createTFoot();
    const footerRow = tfoot.insertRow();
    footerRow.className = 'totals-row'; // Class for styling totals

    if (hasCalculatedGreeks) { // Only show totals if at least one leg contributed
        const headerCell = footerRow.insertCell();
        headerCell.colSpan = 4; // Span first 4 columns (Action, Lots, Type, Strike)
        // Clarify Total represents the entire portfolio value change / exposure
        headerCell.textContent = 'Total Portfolio Exposure'; // Or "Portfolio Totals"
        headerCell.style.textAlign = 'right';
        headerCell.style.fontWeight = 'bold';

        // Display the final calculated totals, rounded
        footerRow.insertCell().textContent = formatNumber(totals.delta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.gamma, 4);
        footerRow.insertCell().textContent = formatNumber(totals.theta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.vega, 4);
        footerRow.insertCell().textContent = formatNumber(totals.rho, 4);

        // Use the CORRECTED setElementState call with a selector string
        setElementState(SELECTORS.greeksTable, 'content'); // Pass selector string

    } else if (totalLegsInput > 0) { // Legs existed, but none were valid for calculation
        const cell = footerRow.insertCell();
        cell.colSpan = 9; // Span all columns
        cell.textContent = 'Could not calculate portfolio totals due to invalid or missing leg data.';
        cell.style.textAlign = 'center';
        cell.style.fontStyle = 'italic';
        // Use the CORRECTED setElementState call with a selector string
        setElementState(SELECTORS.greeksTable, 'content'); // Still 'content', but showing message
    }
    // If totalLegsInput was 0, the tbody already has the message, and state was set earlier.

    // --- Return the calculated totals ---
    // Round totals before returning for consistency if needed elsewhere
    const finalTotals = {
        delta: roundToPrecision(totals.delta, 4),
        gamma: roundToPrecision(totals.gamma, 4),
        theta: roundToPrecision(totals.theta, 4),
        vega: roundToPrecision(totals.vega, 4),
        rho: roundToPrecision(totals.rho, 4)
    };

    logger.info(`renderGreeksTable: Rendered ${processedLegsCount} valid legs. Totals: ${JSON.stringify(finalTotals)}`);
    return finalTotals; // Return the dictionary of calculated totals
}

/**
 * Helper function to format numbers, handling null/undefined/NaN.
 * (Ensure you have this function available in your frontend code)
 *
 * @param {number|string|null|undefined} value The number to format.
 * @param {number} precision Number of decimal places.
 * @param {string} nanPlaceholder String to display if value is not a valid number. Defaults to ''.
 * @returns {string} Formatted number string or placeholder.
 */
function formatNumber(value, precision = 2, nanPlaceholder = '') {
    const num = parseFloat(value);
    if (isNaN(num) || !isFinite(num)) {
        return nanPlaceholder;
    }
    return num.toFixed(precision);
}


function roundToPrecision(num, precision) {
    if (typeof num !== 'number' || !isFinite(num)) {
        return null; // Or 0 depending on how you want to handle invalid inputs
    }
    const factor = Math.pow(10, precision);
    return Math.round(num * factor) / factor;
}



async function fetchAndDisplayGreeksAnalysis(asset, portfolioGreeksData) {
    const container = document.querySelector(SELECTORS.greeksAnalysisResultContainer); // Define this selector
    const section = document.querySelector(SELECTORS.greeksAnalysisSection); // Define this selector

    if (!container || !section) {
        logger.error("Greeks analysis container or section not found in DOM.");
        return;
    }
    if (!asset || !portfolioGreeksData || typeof portfolioGreeksData !== 'object') {
        logger.warn("Greeks analysis skipped: Missing asset or valid Greeks data.");
        setElementState(section, 'hidden'); // Hide the section if no data
        return;
    }
    // Check if all greek values are effectively zero or N/A (might happen if only stock is added)
     const allZeroOrNull = Object.values(portfolioGreeksData).every(v => v === null || v === 0 || !isFinite(v));
    if (allZeroOrNull) {
        logger.info("Greeks analysis skipped: All portfolio Greeks are zero or N/A.");
         container.innerHTML = '<p>No option Greeks to analyze for this position.</p>';
         setElementState(section, 'content'); // Show the section with the message
         setElementState(container, 'content');
         return;
    }


    logger.info(`Fetching Greeks analysis for ${asset}...`);
    setElementState(section, 'content'); // Show the section
    setElementState(container, 'loading', 'Fetching Greeks analysis...');

    try {
        // Ensure marked.js is loaded (similar check as in fetchAnalysis)
        let attempts = 0;
        while (typeof marked === 'undefined' && attempts < 10) {
            await new Promise(resolve => setTimeout(resolve, 200));
            attempts++;
        }
        if (typeof marked === 'undefined') {
            throw new Error("Markdown parser (marked.js) failed to load.");
        }

        const requestBody = {
            asset_symbol: asset,
            portfolio_greeks: portfolioGreeksData // Pass the totals dictionary
        };

        const data = await fetchAPI("/get_greeks_analysis", {
            method: "POST",
            body: JSON.stringify(requestBody)
        });

        const rawAnalysis = data?.greeks_analysis || "*No Greeks analysis content received.*";
        const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
        container.innerHTML = marked.parse(potentiallySanitized);
        setElementState(container, 'content');
        logger.info(`Successfully rendered Greeks analysis for ${asset}`);

    } catch (error) {
        logger.error(`Error fetching or rendering Greeks analysis for ${asset}:`, error);
        // Display error within the Greeks analysis container
        setElementState(container, 'error', `Greeks Analysis Error: ${error.message}`);
        // Maybe hide the whole section on error? Or show the error message inline.
        // Example: Keep section visible but show error in container
        // setElementState(section, 'content');
    }
}





// ===============================================================
// Misc Helpers
// ===============================================================
function gatherStrategyLegsFromTable() {
      const logger = window.logger || window.console;
      logger.debug("--- [gatherStrategyLegs] START ---"); // Mark start

      if (!Array.isArray(strategyPositions)) {
          logger.error("[gatherStrategyLegs] Aborted: strategyPositions is not an array or not defined.");
          return [];
      }
      if (strategyPositions.length === 0) {
           logger.warn("[gatherStrategyLegs] Aborted: strategyPositions array is empty.");
           return [];
      }

      const validLegs = [];
      let invalidLegCount = 0;

      strategyPositions.forEach((pos, index) => {
           logger.debug(`[gatherStrategyLegs] Processing index ${index}, raw data:`, JSON.parse(JSON.stringify(pos))); // <<< Log raw pos data

           let legIsValid = true;
           let validationError = null;

           // --- DEBUG: Log each property value BEFORE validation ---
           logger.debug(`[gatherStrategyLegs] Index ${index} - Checking option_type:`, pos?.option_type);
           logger.debug(`[gatherStrategyLegs] Index ${index} - Checking strike_price:`, pos?.strike_price);
           logger.debug(`[gatherStrategyLegs] Index ${index} - Checking expiry_date:`, pos?.expiry_date);
           logger.debug(`[gatherStrategyLegs] Index ${index} - Checking lots:`, pos?.lots);
           logger.debug(`[gatherStrategyLegs] Index ${index} - Checking last_price:`, pos?.last_price);
           // --- End DEBUG logging ---

           // Validate essential data from the stored position object
           if (!pos || typeof pos !== 'object') {
                validationError = "Position data is not an object.";
                legIsValid = false;
           } else if (!pos.option_type || (pos.option_type !== 'CE' && pos.option_type !== 'PE')) {
                validationError = `Invalid option_type: ${pos.option_type}`;
                legIsValid = false;
           } else if (!pos.strike_price || isNaN(parseFloat(pos.strike_price)) || parseFloat(pos.strike_price) <= 0) {
                validationError = `Invalid strike_price: ${pos.strike_price}`;
                legIsValid = false;
           } else if (!pos.expiry_date || !/^\d{4}-\d{2}-\d{2}$/.test(pos.expiry_date)) {
               validationError = `Invalid expiry_date: ${pos.expiry_date}`;
               legIsValid = false;
           } else if (pos.lots === undefined || pos.lots === null || isNaN(parseInt(pos.lots)) || parseInt(pos.lots) === 0) {
                validationError = `Invalid lots: ${pos.lots}`;
                legIsValid = false;
           } else if (pos.last_price === undefined || pos.last_price === null || isNaN(parseFloat(pos.last_price)) || parseFloat(pos.last_price) < 0) {
                validationError = `Invalid last_price: ${pos.last_price}`;
                legIsValid = false;
           }

           // --- Recalculate DTE ---
           const currentDTE = calculateDaysToExpiry(pos?.expiry_date); // Add safe access
           if (currentDTE === null && legIsValid) {
               validationError = `Could not calculate DTE for expiry ${pos?.expiry_date}.`;
               legIsValid = false;
           }

           // --- Determine Backend Option Type ---
            let backendOpType = '';
            if (legIsValid) {
                // Use safe access again just in case
                const upperOptType = pos?.option_type?.toUpperCase();
                if (upperOptType === 'CE') { backendOpType = 'c'; }
                else if (upperOptType === 'PE') { backendOpType = 'p'; }
                else { validationError = `Invalid stored option_type: ${pos?.option_type}`; legIsValid = false; }
            }

            // --- Handle IV ---
            let ivToSend = null;
            if (legIsValid && pos.iv !== null && pos.iv !== undefined && pos.iv !== '') {
                const parsedIV = parseFloat(pos.iv);
                 if (!isNaN(parsedIV)) { ivToSend = parsedIV; }
                 else { logger.warn(`[gatherStrategyLegs] Invalid IV found for leg ${index} ('${pos.iv}'), sending null.`); }
            }

           // --- If Leg is Valid, Add to Array ---
           if (legIsValid) {
               validLegs.push({
                    // Data needed by backend
                    op_type: backendOpType,
                    strike: String(pos.strike_price),
                    tr_type: parseInt(pos.lots) >= 0 ? 'b' : 's',
                    op_pr: String(pos.last_price),
                    lot: String(Math.abs(parseInt(pos.lots))),
                    lot_size: pos.lot_size ? String(pos.lot_size) : null,
                    iv: ivToSend,
                    days_to_expiry: currentDTE,
                    expiry_date: pos.expiry_date,
               });
           } else {
                // Log the first validation error found
                logger.error(`[gatherStrategyLegs] Skipping invalid position data at index ${index}. Reason: ${validationError || 'Unknown validation failure'}. Data:`, JSON.parse(JSON.stringify(pos)));
                invalidLegCount++;
           }
      }); // End forEach

      if (invalidLegCount > 0) {
         // Alert moved to fetchPayoffChart to avoid multiple alerts if called elsewhere
         // alert(`Error: ${invalidLegCount} invalid leg(s) found...`);
         logger.warn(`[gatherStrategyLegs] Found ${invalidLegCount} invalid legs.`);
      }

      logger.debug(`[gatherStrategyLegs] Returning ${validLegs.length} valid legs. (Ignored ${invalidLegCount})`);
      logger.debug("--- [gatherStrategyLegs] END ---");
      return validLegs; // Return ONLY the valid legs
 }
const numericATMStrike = Number(atmStrikeObjectKey);
logger.debug(`Attempting to find ATM row with data-strike="${numericATMStrike}". Tbody has ${currentTbody.rows.length} rows.`);
const atmRow = currentTbody.querySelector(`tr[data-strike="${numericATMStrike}"]`);
if (atmRow) {
  // Scroll logic here...
} else {
  logger.debug(`ATM strike row for key (${atmStrikeObjectKey} / ${numericATMStrike}) not found for scrolling.`);
}