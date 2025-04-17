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
};

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
    logger.info("Initializing page data...");
    resetResultsUI(); // Start with clean results area
    await loadAssets(); // This will trigger subsequent data loads
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

function startAutoRefresh() {
    stopAutoRefresh(); // Clear any existing timer first
    if (!activeAsset) {
        logger.info("No active asset, auto-refresh not started.");
        return;
    }
    logger.info(`Starting auto-refresh every ${REFRESH_INTERVAL_MS}ms for ${activeAsset}`);
    // Store previous data *before* starting the interval
    previousSpotPrice = currentSpotPrice;
    // previousOptionChainData should be populated by the initial fetchOptionChain call
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
    activeAsset = asset; // Update global state
    stopAutoRefresh(); // Stop refresh when changing asset

    // Clear previous data used for highlighting and state
    previousOptionChainData = {};
    previousSpotPrice = 0;
    currentSpotPrice = 0; // Reset current spot price

    if (!asset) {
        // Reset dependent UI if no asset selected
        populateDropdown(SELECTORS.expiryDropdown, [], "-- Select Asset First --");
        setElementState(SELECTORS.optionChainTableBody, 'content');
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Asset</td></tr>`;
        setElementState(SELECTORS.analysisResultContainer, 'content');
        document.querySelector(SELECTORS.analysisResultContainer).innerHTML = 'Select an asset to load analysis...';
        setElementState(SELECTORS.spotPriceDisplay, 'content'); // Set state
        document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
        resetResultsUI();
        setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Hide global error
        return;
    }

    logger.info(`Asset changed to: ${asset}. Fetching data...`);
    setElementState(SELECTORS.expiryDropdown, 'loading');
    setElementState(SELECTORS.optionChainTableBody, 'loading');
    setElementState(SELECTORS.analysisResultContainer, 'loading');
    // Ensure news container state is also set to loading
    setElementState(SELECTORS.newsResultContainer, 'loading');
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...');
    resetResultsUI(); // Clear previous results on asset change
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear global error on new asset load

    // --- Call Debug Endpoint ---
    try {
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST', body: JSON.stringify({ asset: asset })
        });
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        logger.error("Failed to send debug asset selection:", debugErr.message);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Debug Sync Failed: ${debugErr.message}`);
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
    }
    // --- End Debug Call ---

    try {
        // Fetch core data in parallel using allSettled
        const [spotResult, expiryResult, analysisResult, newsResult] = await Promise.allSettled([
            fetchNiftyPrice(asset), // Initial fetch (not refresh)
            fetchExpiries(asset),
            fetchAnalysis(asset),
            fetchNews(asset)
        ]);

        let hasCriticalError = false;

        // Process results
        if (spotResult.status === 'rejected') {
            logger.error(`Error fetching spot price: ${spotResult.reason?.message || spotResult.reason}`);
            // Let fetchNiftyPrice handle its own error display
        }
        if (expiryResult.status === 'rejected') {
            logger.error(`Error fetching expiries: ${expiryResult.reason?.message || expiryResult.reason}`);
            hasCriticalError = true; // Can't load chain without expiries
            setElementState(SELECTORS.expiryDropdown, 'error', 'Failed to load expiries');
            setElementState(SELECTORS.optionChainTableBody, 'error', 'Failed to load expiries');
        }
        if (analysisResult.status === 'rejected') {
            logger.error(`Error fetching analysis: ${analysisResult.reason?.message || analysisResult.reason}`);
            // Error display is handled within fetchAnalysis
        }
        if (newsResult.status === 'rejected') {
            logger.error(`Error fetching news: ${newsResult.reason?.message || newsResult.reason}`);
            // Error display is handled within fetchNews
        } // ***** Closing brace was missing here - ADDED *****

        // If initial load was okay (no critical errors), start auto-refresh
        if (!hasCriticalError) {
            startAutoRefresh();
        } else {
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load essential data (expiries) for ${asset}. Check console.`);
        }

    } catch (err) {
        // Catch unexpected errors during orchestration
        logger.error(`Unexpected error fetching initial data for ${asset}:`, err);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load page data for ${asset}.`);
        stopAutoRefresh(); // Stop refresh on major initial load error
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
    setElementState(SELECTORS.assetDropdown, 'loading');
    try {
        const data = await fetchAPI("/get_assets");
        const assets = data?.assets || [];
        populateDropdown(SELECTORS.assetDropdown, assets, "-- Select Asset --");
        setElementState(SELECTORS.assetDropdown, 'content');

        // Default to NIFTY or first asset
        const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
        let defaultAsset = null;
        if (assets.includes("NIFTY")) {
            defaultAsset = "NIFTY";
        } else if (assets.length > 0) {
            defaultAsset = assets[0];
            logger.warn(`"NIFTY" not found, defaulting to first asset: ${defaultAsset}`);
        }

        if (defaultAsset && assetDropdown) {
            assetDropdown.value = defaultAsset;
            logger.info(`Defaulting asset selection to: ${defaultAsset}`);
            await handleAssetChange(); // Trigger data load for default
        } else if (assets.length === 0){
             logger.warn("No assets found in database.");
             setElementState(SELECTORS.assetDropdown, 'error', 'No assets found');
             await handleAssetChange(); // Clear dependent fields
        } else {
             // Assets exist but neither NIFTY nor the first one is defaultable? Unlikely.
             await handleAssetChange(); // Clear dependent fields
        }

    } catch (error) {
        logger.error("Failed to load assets:", error);
        setElementState(SELECTORS.assetDropdown, 'error', `Assets Error: ${error.message}`);
        setElementState(SELECTORS.expiryDropdown, 'error', 'Asset load failed');
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Asset load failed');
        // Show global error as this is critical failure
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load assets: ${error.message}`);
    }
}

/** Fetches stock analysis for the selected asset */
async function fetchAnalysis(asset) {
    if (!asset) return;
    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Fetching analysis...');
    try {
        // Ensure marked.js is loaded
        let attempts = 0;
        while (typeof marked === 'undefined' && attempts < 10) {
            await new Promise(resolve => setTimeout(resolve, 200)); // Wait 200ms
            attempts++;
        }
        if (typeof marked === 'undefined') {
            throw new Error("Markdown parser (marked.js) failed to load in time.");
        }

        const data = await fetchAPI("/get_stock_analysis", {
            method: "POST", body: JSON.stringify({ asset })
        });
        const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
        if (analysisContainer) {
            const rawAnalysis = data?.analysis || "*No analysis content received.*";
            // Basic sanitization (remove script tags) - consider more robust library if needed
            const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
            // Use marked.parse() which should be available now
            analysisContainer.innerHTML = marked.parse(potentiallySanitized);
            setElementState(SELECTORS.analysisResultContainer, 'content');
        } else {
             logger.warn("Analysis container not found in DOM.");
        }
    } catch (error) {
        // Display error within the analysis container
        setElementState(SELECTORS.analysisResultContainer, 'error', `Analysis Error: ${error.message}`);
        // Log the full error for debugging
        logger.error(`Error fetching analysis for ${asset}:`, error);
        if (error.message.includes("Essential stock data not found")) {
             setElementState(SELECTORS.analysisResultContainer, 'content'); // Set state to content
             analysisContainer.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${error.message}</p>`; // Display message inside
        } else {
             // Display other errors locally too
             setElementState(SELECTORS.analysisResultContainer, 'error', `Analysis Error: ${error.message}`);
        }
        // Do not show global error here
        // setElementState(SELECTORS.globalErrorDisplay, 'error', `Analysis Error: ${error.message}`); // REMOVE/COMMENT OUT
        // ***** END CHANGE *****
         // Re-throw error so handleAssetChange knows about the failure (optional)
         // throw error;
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

    // --- Start of Function Log ---
    // console.log(`[FETCH CHAIN START] Asset: ${asset}, Expiry: ${expiry}, Refresh: ${isRefresh}`);

    const optionTable = document.querySelector("#optionChainTable");
    if (!optionTable) {
        logger.error("Option chain table element (#optionChainTable) not found.");
        return;
    }
    const currentTbody = optionTable.querySelector("tbody");
    if (!currentTbody) {
        logger.error("Option chain tbody element not found within #optionChainTable.");
        return;
    }

    if (!asset || !expiry) {
        currentTbody.innerHTML = `<tr><td colspan="7">Select Asset and Expiry</td></tr>`;
        if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
        return;
    }
    if (!isRefresh) {
        setElementState(SELECTORS.optionChainTableBody, 'loading');
        currentTbody.innerHTML = `<tr><td colspan="7" class="loading-text">Loading Chain...</td></tr>`;
    }

    try {
        // --- Fetch Spot Price if Needed ---
        if (currentSpotPrice <= 0 && scrollToATM) {
             logger.info("Spot price unavailable, fetching before option chain for ATM scroll...");
             try {
                 await fetchNiftyPrice(asset);
             } catch (spotError) {
                 logger.warn("Failed to fetch spot price for ATM calculation:", spotError.message);
             }
             if (currentSpotPrice <= 0) {
                 logger.warn("Spot price still unavailable, cannot calculate ATM strike accurately.");
                 scrollToATM = false;
             }
         }

        // --- Fetch Option Chain Data ---
        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);

        // ***** LOG 1: Log the entire data object returned by fetchAPI *****
        console.log('[FETCH CHAIN] Data received from fetchAPI:', JSON.stringify(data, null, 2));
        // ******************************************************************

        const currentChainData = data?.option_chain;

        // ***** LOG 2: Log the extracted option_chain object *****
        console.log('[FETCH CHAIN] Extracted currentChainData:', JSON.stringify(currentChainData, null, 2));
        // *******************************************************

        // --- Handle Empty/Invalid Data ---
        if (!currentChainData || typeof currentChainData !== 'object' || currentChainData === null || Object.keys(currentChainData).length === 0) {
            logger.warn(`[FETCH CHAIN] currentChainData is null, not an object, or empty. Asset: ${asset}, Expiry: ${expiry}`);
            currentTbody.innerHTML = `<tr><td colspan="7">No option chain data available for ${asset} on ${expiry}</td></tr>`;
            if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
            previousOptionChainData = {};
            return;
        }

        // --- Render Table & Handle Highlights ---
        const strikes = Object.keys(currentChainData).map(Number).sort((a, b) => a - b);
        const atmStrike = currentSpotPrice > 0 ? findATMStrike(strikes, currentSpotPrice) : null;

        currentTbody.innerHTML = ''; // Clear existing tbody

        strikes.forEach((strike, strikeIndex) => {
            const optionDataForStrike = currentChainData[strike];
            const optionData = (typeof optionDataForStrike === 'object' && optionDataForStrike !== null)
                                ? optionDataForStrike
                                : { call: null, put: null };

             // ***** LOG 3: Log data for specific strikes within the loop *****
             if (strikeIndex < 2 || strike === atmStrike) {
                 console.log(`[STRIKE LOOP ${strike}] optionData from currentChainData[${strike}]:`, JSON.stringify(optionData));
             }
             // **************************************************************

            const call = optionData.call || {};
            const put = optionData.put || {};

             // ***** LOG 4: Log the derived call/put objects *****
             if (strikeIndex < 2 || strike === atmStrike) {
                 console.log(`[STRIKE LOOP ${strike}] Derived call object:`, JSON.stringify(call));
                 console.log(`[STRIKE LOOP ${strike}] Derived put object:`, JSON.stringify(put));
             }
             // ***************************************************

            const prevOptionData = previousOptionChainData[strike] || { call: {}, put: {} };
            const prevCall = prevOptionData.call || {};
            const prevPut = prevOptionData.put || {};

            const tr = document.createElement("tr");
            tr.dataset.strike = strike;
            if (atmStrike !== null && Math.abs(strike - atmStrike) < 0.01) {
                tr.classList.add("atm-strike");
            }

            const columns = [ // Column definitions remain the same
                { class: 'call clickable price', type: 'CE', key: 'last_price', format: val => formatNumber(val, 2, '-') },
                { class: 'call oi', key: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'call iv', key: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'strike', key: 'strike', isStrike: true, format: val => formatNumber(val, val % 1 === 0 ? 0 : 2) },
                { class: 'put iv', key: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'put oi', key: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'put clickable price', type: 'PE', key: 'last_price', format: val => formatNumber(val, 2, '-') },
            ];
            
            if (strikeIndex < 2 || strike === atmStrike) {
                console.log(`[STRIKE LOOP ${strike}] Derived call object:`, JSON.stringify(call)); // Add stringify
                console.log(`[STRIKE LOOP ${strike}] Derived put object:`, JSON.stringify(put));   // Add stringify
            }

            // ----- Process Columns for the Row -----
            columns.forEach(col => {
                try { // Add try...catch around inner cell processing
                    const td = document.createElement('td');
                    td.className = col.class;

                    let currentValue; // Declare currentValue

                    // ***** USING SIMPLIFIED/DIRECT VALUE ACCESS *****
                    if (col.isStrike) {
                        currentValue = strike;
                    } else if (col.class.includes('call')) {
                        // Directly access the 'call' object defined in the outer scope
                        currentValue = (typeof call === 'object' && call !== null) ? call[col.key] : undefined;
                    } else { // Must be put
                        // Directly access the 'put' object defined in the outer scope
                        currentValue = (typeof put === 'object' && put !== null) ? put[col.key] : undefined;

                    }
                    // *************************************************

                    // ***** LOG 5: The RAW value log *****
                    if (!col.isStrike && (strikeIndex < 1 || strike === atmStrike)) { // Log only first strike & ATM
                        console.log(`[RAW] Strike: ${strike}, Key: ${col.key}, Type: ${col.class.includes('call') ? 'CE' : 'PE'}, Value Received:`, currentValue, `(Type: ${typeof currentValue})`);
                    }
                    // *************************************

                    // Format and set the text content
                    td.textContent = col.format(currentValue);

                    // Add data attributes (using direct call/put access)
                    if (col.type) {
                        td.dataset.type = col.type;
                        let sourceObj = col.class.includes('call') ? call : put; // Use directly defined call/put
                        if(typeof sourceObj === 'object' && sourceObj !== null) {
                            const ivValue = sourceObj['implied_volatility'];
                            const priceValue = sourceObj['last_price'];

                            if (ivValue !== null && ivValue !== undefined && !isNaN(parseFloat(ivValue))) {
                                td.dataset.iv = ivValue;
                            }
                            if (priceValue !== null && priceValue !== undefined && !isNaN(parseFloat(priceValue))) {
                                td.dataset.price = priceValue;
                            } else {
                                td.dataset.price = 0;
                            }
                        } else {
                             td.dataset.price = 0; // Default dataset price if sourceObj missing
                        }
                    }

                    // Highlight check (using direct prevCall/prevPut access)
                    if (isRefresh && !col.isStrike) {
                        let prevDataObject = col.class.includes('call') ? prevCall : prevPut;
                        if(typeof prevDataObject === 'object' && prevDataObject !== null) {
                            let previousValue = prevDataObject[col.key];
                            let changed = false;
                            const currentExists = currentValue !== null && typeof currentValue !== 'undefined';
                            const previousExists = previousValue !== null && typeof previousValue !== 'undefined';
                            if (currentExists && previousExists) {
                                if (typeof currentValue === 'number' && typeof previousValue === 'number') {
                                    changed = Math.abs(currentValue - previousValue) > 0.001;
                                } else {
                                    changed = currentValue !== previousValue;
                                }
                            } else if (currentExists !== previousExists) {
                                changed = true;
                            }
                            if (changed) {
                                highlightElement(td);
                            }
                        }
                    }
                    tr.appendChild(td); // Append the cell to the row
                 } catch (cellError) {
                     // ***** LOG 6: Catch errors within the cell processing *****
                     console.error(`[CELL ERROR] Strike: ${strike}, Column Key: ${col.key}, Error:`, cellError);
                     // Optionally add placeholder content to the cell on error
                     const errorTd = document.createElement('td');
                     errorTd.textContent = 'ERR';
                     errorTd.className = col.class + ' error-message';
                     errorTd.style.backgroundColor = 'pink';
                     tr.appendChild(errorTd); // Append error cell instead
                     // ************************************************************
                 }
            }); // End columns.forEach

            currentTbody.appendChild(tr); // Append the row to the table body

        }); // End strikes.forEach

        if (!isRefresh) {
            setElementState(SELECTORS.optionChainTableBody, 'content');
        }
        previousOptionChainData = currentChainData; // Store data for next comparison

        // Scroll logic (remains the same)
        if (scrollToATM && atmStrike !== null && !isRefresh) {
             setTimeout(() => {
                 const atmRow = currentTbody.querySelector(".atm-strike");
                 if (atmRow) {
                     atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
                     logger.debug(`Scrolled to ATM strike: ${atmStrike}`);
                 } else { logger.warn(`ATM strike row (${atmStrike}) not found for scrolling.`); }
             }, 150);
         }

    } catch (error) { // This catches errors in fetchAPI or outer logic
        logger.error("Error during fetchOptionChain execution (outer try/catch):", error);
        if (currentTbody) {
            currentTbody.innerHTML = `<tr><td colspan="7" class="error-message">Chain Error: ${error.message}</td></tr>`;
        }
        if (!isRefresh) {
            setElementState(SELECTORS.optionChainTableBody, 'error', `Chain Error: ${error.message}`);
        } else {
            logger.warn(`Option Chain refresh error: ${error.message}`);
        }
        previousOptionChainData = {}; // Clear previous data on error
    }
    // --- End of Function Log ---
    // console.log(`[FETCH CHAIN END] Asset: ${asset}, Expiry: ${expiry}`);
}

// ===============================================================
// Event Delegation Handlers
// ===============================================================

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


/** Resets the chart and results UI to initial state */
function resetResultsUI() {
     const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
     if (chartContainer) {
         chartContainer.innerHTML = '<div class="placeholder-text">Add positions and click "Update" to see the payoff chart.</div>';
         setElementState(SELECTORS.payoffChartContainer, 'content');
     }

     // Reset other containers/tables
     setElementState(SELECTORS.taxInfoContainer, 'content');
     document.querySelector(SELECTORS.taxInfoContainer).innerHTML = '<p class="loading-text">Update strategy to calculate charges.</p>'; // Reset placeholder

     setElementState(SELECTORS.greeksTable, 'content');
     const greeksTable = document.querySelector(SELECTORS.greeksTable);
     if (greeksTable) {
         // Reset caption, clear body/foot
          const caption = greeksTable.querySelector('caption'); if (caption) caption.textContent = 'Portfolio Option Greeks';
          const greekBody = greeksTable.querySelector('tbody'); if(greekBody) greekBody.innerHTML = '<tr><td colspan="9" class="loading-text">Update strategy to calculate Greeks.</td></tr>'; // Reset placeholder
          const greekFoot = greeksTable.querySelector('tfoot'); if(greekFoot) greekFoot.innerHTML = "";
     }

     setElementState(SELECTORS.costBreakdownList, 'content');
     document.querySelector(SELECTORS.costBreakdownList).innerHTML = ""; // Clear list
     setElementState(SELECTORS.costBreakdownContainer, 'hidden'); // Hide the details initially
     const detailsElement = document.querySelector(SELECTORS.costBreakdownContainer);
     if (detailsElement) detailsElement.open = false; // Ensure it's closed


     // Reset metrics display using the specific value spans
     displayMetric("N/A", SELECTORS.maxProfitDisplay);
     displayMetric("N/A", SELECTORS.maxLossDisplay);
     displayMetric("N/A", SELECTORS.breakevenDisplay);
     displayMetric("N/A", SELECTORS.rewardToRiskDisplay);
     displayMetric("N/A", SELECTORS.netPremiumDisplay);

     // Reset the labels as well if needed (optional, displayMetric only changes value part)
     const newsContainer = document.querySelector(SELECTORS.newsResultContainer);
     if (newsContainer) {
         setElementState(SELECTORS.newsResultContainer, 'content'); // Set state
         newsContainer.innerHTML = '<p class="loading-text">Select an asset to load news...</p>'; // Reset placeholder
    }
    
}


// ===============================================================
// Payoff Chart & Results Logic
// ===============================================================

/** Fetches calculation results and displays the payoff chart and other metrics */
async function fetchPayoffChart() {
    const asset = document.querySelector(SELECTORS.assetDropdown)?.value;
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    const updateButton = document.querySelector(SELECTORS.updateChartButton);

    if (!asset) { alert("Please select an asset."); return; }
    if (strategyPositions.length === 0) {
         resetResultsUI(); // Clear UI if no positions
         alert("Please add positions to the strategy first.");
         return;
    }
    if (!chartContainer) { logger.error("Payoff chart container element not found."); return; }

    // Set loading states for all results areas
    setElementState(SELECTORS.payoffChartContainer, 'loading', 'Generating results...');
    setElementState(SELECTORS.taxInfoContainer, 'loading', 'Calculating...');
    setElementState(SELECTORS.greeksTable, 'loading', 'Calculating...');
    setElementState(SELECTORS.costBreakdownContainer, 'hidden'); // Hide breakdown during load
    // Set metrics to loading state
    displayMetric("...", SELECTORS.maxProfitDisplay);
    displayMetric("...", SELECTORS.maxLossDisplay);
    displayMetric("...", SELECTORS.breakevenDisplay);
    displayMetric("...", SELECTORS.rewardToRiskDisplay);
    displayMetric("...", SELECTORS.netPremiumDisplay);

    if (updateButton) updateButton.disabled = true;

    // --- Prepare Request Data ---
    // Ensure all required fields (including iv, days_to_expiry) are present
    let dataIsValid = true;
    const requestStrategy = strategyPositions.map((pos, index) => {
        // Recalculate DTE just in case expiry changed relative to today
        const currentDTE = calculateDaysToExpiry(pos.expiry_date);
        if (currentDTE === null) {
             logger.error(`Invalid expiry date found for leg ${index+1} during final validation.`);
             dataIsValid = false;
        }
        // Ensure IV is present (it should be from addPosition, but double check)
        if (pos.iv === null) {
             logger.warn(`Missing IV for leg ${index+1} (${pos.option_type} ${pos.strike_price}). Greeks may be inaccurate.`);
             // Decide if this is critical? For now, allow sending null.
        }

        return {
            option_type: pos.option_type, // CE or PE
            strike_price: String(pos.strike_price), // Send as string
            tr_type: pos.lots >= 0 ? "b" : "s", // Determine buy/sell
            option_price: String(pos.last_price), // Send as string
            expiry_date: pos.expiry_date,
            lots: String(Math.abs(pos.lots)), // Send absolute lots as string
            lot_size: null, // Let backend fetch default lot size based on asset
            // Include IV and DTE needed for Greeks
            iv: pos.iv, // Send stored IV (can be null)
            days_to_expiry: currentDTE, // Send current DTE (can be null if expiry invalid)
        };
    });

    if (!dataIsValid) {
        alert("Invalid expiry date found in one or more legs. Cannot calculate.");
         setElementState(SELECTORS.payoffChartContainer, 'error', 'Invalid expiry date found.');
         if (updateButton) updateButton.disabled = false;
         return;
    }

    const requestData = { asset: asset, strategy: requestStrategy };
    logger.debug("Sending request to /get_payoff_chart:", requestData);

    try {
        const data = await fetchAPI('/get_payoff_chart', {
            method: 'POST', body: JSON.stringify(requestData)
        });
        logger.debug("Received response from /get_payoff_chart:", data);

        // --- Render Chart Image ---
        if (data && data.image_base64) {
            chartContainer.innerHTML = ""; // Clear loading/placeholder
            const img = document.createElement("img");
            img.src = `data:image/png;base64,${data.image_base64}`;
            img.alt = `Option Strategy Payoff Chart for ${asset}`;
            img.className = "payoff-chart-image"; // Add class for styling
            chartContainer.appendChild(img);
            setElementState(SELECTORS.payoffChartContainer, 'content');
            logger.info("Successfully rendered Matplotlib chart image.");
        } else {
             // Handle case where backend succeeded but didn't return image (or data is null)
             logger.error("Payoff chart image_base64 missing in response or data is null.");
             setElementState(SELECTORS.payoffChartContainer, 'error', 'Chart image generation failed or missing.');
        }

        // --- Display Metrics, Breakdown, Taxes, Greeks ---
        // Metrics (handle potential null data)
        const metricsData = data?.metrics?.metrics; // Access nested metrics safely
        if (metricsData) {
            displayMetric(metricsData.max_profit, SELECTORS.maxProfitDisplay, "", "", 2, true);
            displayMetric(metricsData.max_loss, SELECTORS.maxLossDisplay, "", "", 2, true);
            const breakevens = Array.isArray(metricsData.breakeven_points) && metricsData.breakeven_points.length > 0
                ? metricsData.breakeven_points.map(p => formatCurrency(p, 2, 'N/A', '₹')).join(', ')
                : "None";
            displayMetric(breakevens, SELECTORS.breakevenDisplay); // Display as is
            displayMetric(metricsData.reward_to_risk_ratio, SELECTORS.rewardToRiskDisplay); // Format handles ∞ etc.
            // Determine prefix for Net Premium based on sign
            const netPremiumValue = metricsData.net_premium;
            const netPremiumPrefix = (typeof netPremiumValue === 'number' && netPremiumValue >= 0) ? "Net Credit: " : "Net Debit: ";
            displayMetric(Math.abs(netPremiumValue), SELECTORS.netPremiumDisplay, netPremiumPrefix, "", 2, true);
        } else {
             logger.warn("Metrics data missing or invalid in response.");
             displayMetric("N/A", SELECTORS.maxProfitDisplay);
             displayMetric("N/A", SELECTORS.maxLossDisplay);
             displayMetric("N/A", SELECTORS.breakevenDisplay);
             displayMetric("N/A", SELECTORS.rewardToRiskDisplay);
             displayMetric("N/A", SELECTORS.netPremiumDisplay);
        }

        // Cost Breakdown (Premium only)
        const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
        const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
        const costBreakdownData = data?.metrics?.cost_breakdown_per_leg; // From metrics response
        if (breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) {
            breakdownList.innerHTML = ""; // Clear previous
            costBreakdownData.forEach(item => {
                const li = document.createElement("li");
                const absPremium = Math.abs(item.total_premium);
                const premiumEffect = item.effect === 'Paid' ? `(Paid ${formatCurrency(absPremium)})` : `(Received ${formatCurrency(absPremium)})`;
                // Use Buy/Sell from action field, lots/lot_size for quantity
                li.textContent = `${item.action} ${item.lots}x${item.lot_size} ${item.type} @ ${item.strike} ${premiumEffect}`;
                breakdownList.appendChild(li);
            });
             setElementState(SELECTORS.costBreakdownContainer, 'content'); // Use state to make visible
             breakdownContainer.style.display = ""; // Ensure visible
             breakdownContainer.open = false; // Default closed state for details
        } else if (breakdownContainer) {
            setElementState(SELECTORS.costBreakdownContainer, 'hidden'); // Hide if no data
        }

        // Taxes/Charges (handle potential null data)
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
        if (taxContainer) {
            if (data?.charges) {
                renderTaxTable(taxContainer, data.charges);
                setElementState(SELECTORS.taxInfoContainer, 'content');
            } else {
                 logger.warn("Charges data missing in response.");
                 taxContainer.innerHTML = "<p>Charge data unavailable.</p>";
                 setElementState(SELECTORS.taxInfoContainer, 'content'); // Set content state even for placeholder
            }
        }

        // Greeks (handle potential null/undefined data)
        const greeksTable = document.querySelector(SELECTORS.greeksTable);
        if (greeksTable) {
            // Pass the greeks list (which could be null/undefined) directly
            renderGreeksTable(greeksTable, data?.greeks); // Renderer handles null/array cases
            setElementState(SELECTORS.greeksTable, 'content'); // Set content state after rendering
        }


    } catch (error) {
        logger.error("Error fetching or displaying payoff results:", error);
        // Set error states for all results areas
        setElementState(SELECTORS.payoffChartContainer, 'error', `Calculation Error: ${error.message}`);
        displayMetric("Error", SELECTORS.maxProfitDisplay);
        displayMetric("Error", SELECTORS.maxLossDisplay);
        displayMetric("Error", SELECTORS.breakevenDisplay);
        displayMetric("Error", SELECTORS.rewardToRiskDisplay);
        displayMetric("Error", SELECTORS.netPremiumDisplay);
        setElementState(SELECTORS.taxInfoContainer, 'error', 'Calculation Failed');
        setElementState(SELECTORS.greeksTable, 'error', 'Calculation Failed');
        setElementState(SELECTORS.costBreakdownContainer, 'hidden');

    } finally {
         // Always re-enable the button
         if (updateButton) updateButton.disabled = false;
    }
}

// --- Rendering Helpers for Payoff Results ---

function renderTaxTable(containerElement, taxData) {
    // Guard against null/undefined taxData or missing nested properties
    if (!taxData || !taxData.breakdown_per_leg || !taxData.charges_summary || !Array.isArray(taxData.breakdown_per_leg)) {
        containerElement.innerHTML = "<p>Charge calculation data is incomplete or unavailable.</p>";
        logger.warn("renderTaxTable called with invalid or incomplete taxData:", taxData);
        return;
    }

    containerElement.innerHTML = ""; // Clear previous content

    const details = document.createElement('details');
    details.className = "results-details tax-details";
    details.open = false; // Default closed

    const summary = document.createElement('summary');
    summary.innerHTML = `<strong>Estimated Charges Breakdown (Total: ${formatCurrency(taxData.total_estimated_cost, 2)})</strong>`;
    details.appendChild(summary);

    const tableWrapper = document.createElement('div');
    tableWrapper.className = 'table-wrapper thin-scrollbar'; // Add scrollbar class
    details.appendChild(tableWrapper);

    const table = document.createElement("table");
    table.className = "results-table charges-table data-table"; // Add data-table class
    const charges = taxData.charges_summary || {};
    const breakdown = taxData.breakdown_per_leg;

    // Ensure all expected keys exist in breakdown items, providing defaults
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
            <td class="note" title="${t.stt_note || ''}">${((t.stt_note || '').substring(0, 15))}${ (t.stt_note || '').length > 15 ? '...' : ''}</td>
        </tr>`).join('');

    // Ensure all expected keys exist in charges summary
    const total_stt = charges.stt ?? 0;
    const total_stamp = charges.stamp_duty ?? 0;
    const total_sebi = charges.sebi_fee ?? 0;
    const total_txn = charges.txn_charges ?? 0; // Note: backend key is txn_charges
    const total_brokerage = charges.brokerage ?? 0;
    const total_gst = charges.gst ?? 0;
    const overall_total = taxData.total_estimated_cost ?? 0;


    // Colspan = 12 based on header
    table.innerHTML = `
        <thead>
            <tr><th>Act</th><th>Type</th><th>Strike</th><th>Lots</th><th>Premium</th><th>STT</th><th>Stamp</th><th>SEBI</th><th>Txn</th><th>Broker</th><th>GST</th><th title="Securities Transaction Tax Note">STT Note</th></tr>
        </thead>
        <tbody>${tableBody}</tbody>
        <tfoot>
            <tr class="totals-row">
                <td colspan="5" style="text-align: right; font-weight: bold;">Total</td>
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
    // Clear previous table content (headers, body, footer)
    tableElement.innerHTML = '';
    // Add caption back
    const caption = tableElement.createCaption();
    caption.className = "table-caption";
    caption.textContent = "Portfolio Option Greeks";

    // --- Check if the input is a valid array ---
    if (!Array.isArray(greeksList)) {
        logger.warn("Greeks data provided is not an array or is null/undefined. Cannot render table.");
        const tbody = tableElement.createTBody();
        tbody.innerHTML = `<tr><td colspan="9" class="error-message">Greeks data unavailable or invalid format received.</td></tr>`; // Colspan=9
        return;
    }

    const totalLegsProcessed = greeksList.length; // Total legs returned by backend

    // --- Case 1: No legs returned by backend ---
    if (totalLegsProcessed === 0) {
         logger.info("No Greek results returned from backend.");
         const tbody = tableElement.createTBody();
         tbody.innerHTML = `<tr><td colspan="9">No strategy legs were processed for Greeks (e.g., missing IV).</td></tr>`; // Colspan=9
         return;
    }

    // --- Proceed with rendering ---
    caption.textContent = `Portfolio Option Greeks (${totalLegsProcessed} Leg${totalLegsProcessed > 1 ? 's' : ''} Processed)`;

    // --- Create Header ---
    const thead = tableElement.createTHead();
    thead.innerHTML = `
        <tr>
            <th>Action</th>
            <th>Quantity</th>
            <th>Type</th>
            <th>Strike</th>
            <th title="Portfolio Delta: Change in option value per ₹1 change in underlying">Δ Delta</th>
            <th title="Portfolio Gamma: Change in Delta per ₹1 change in underlying">Γ Gamma</th>
            <th title="Portfolio Theta: Change in option value per day passing (Daily Decay)">Θ Theta/Day</th>
            <th title="Portfolio Vega: Change in option value per 1% change in Implied Volatility">Vega</th>
            <th title="Portfolio Rho: Change in option value per 1% change in Interest Rate">Ρ Rho</th>
        </tr>`;

    // --- Create Body & Calculate Totals ---
    const tbody = tableElement.createTBody();
    let hasCalculatedGreeks = false; // Track if we have valid numbers for totals
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    let skippedLegsCount = 0;

    greeksList.forEach(g => {
        const row = tbody.insertRow();
        // Safely access nested data
        const inputData = g?.input_data;
        const gv = g?.calculated_greeks_per_share; // Backend sends PER-SHARE greeks

        if (!inputData || !gv) {
            // Handle potential malformed item in the list
            logger.warn("Malformed item found in Greeks list:", g);
            row.className = 'greeks-skipped';
            const cell = row.insertCell();
            cell.colSpan = 9; // Span all columns
            cell.textContent = `Invalid data for leg index ${g?.leg_index || 'unknown'}`;
            cell.classList.add('error-message');
            skippedLegsCount++;
            return; // Skip this iteration
        }

        const action = (inputData.tr_type || '?').toUpperCase(); // 'B' or 'S'
        // Backend should provide these in input_data if needed, or we get from strategyPositions?
        // Let's assume they are in input_data for consistency:
        const lots = typeof inputData.lots === 'number' ? inputData.lots : parseInt(inputData.lots || '0', 10); // Added 'lots' to input_data expectation
        const lot_size = typeof inputData.lot_size === 'number' ? inputData.lot_size : parseInt(inputData.lot_size || '0', 10); // Added 'lot_size'
        const quantity = lots * lot_size;
        const lotsDisplay = (lots && lot_size) ? `${lots}x${lot_size}=${quantity}` : 'N/A';
        const optType = (inputData.op_type || '?').toUpperCase(); // 'C' or 'P'
        const strike = inputData.strike || '?';

        // Fill cell data for the leg
        row.insertCell().textContent = action;
        row.insertCell().textContent = lotsDisplay;
        row.insertCell().textContent = optType;
        row.insertCell().textContent = formatNumber(strike, 2); // Format strike

        // Format and display PER-SHARE greeks for the leg
        row.insertCell().textContent = formatNumber(gv.delta, 4, '-');
        row.insertCell().textContent = formatNumber(gv.gamma, 4, '-');
        row.insertCell().textContent = formatNumber(gv.theta, 4, '-');
        row.insertCell().textContent = formatNumber(gv.vega, 4, '-');
        row.insertCell().textContent = formatNumber(gv.rho, 4, '-');

        // --- Accumulate PORTFOLIO totals ---
        // Check if all necessary Greek values are valid numbers and quantity is positive
        if (quantity > 0 &&
            typeof gv.delta === 'number' && isFinite(gv.delta) &&
            typeof gv.gamma === 'number' && isFinite(gv.gamma) &&
            typeof gv.theta === 'number' && isFinite(gv.theta) &&
            typeof gv.vega === 'number' && isFinite(gv.vega) &&
            typeof gv.rho === 'number' && isFinite(gv.rho) )
        {
            // Add (per-share greek * quantity) to totals
            totals.delta += gv.delta * quantity;
            totals.gamma += gv.gamma * quantity;
            totals.theta += gv.theta * quantity; // Theta is per day
            totals.vega += gv.vega * quantity;   // Vega is per 1% IV change
            totals.rho += gv.rho * quantity;     // Rho is per 1% rate change
            hasCalculatedGreeks = true; // Mark that we have at least one valid calculation
            row.classList.add('greeks-calculated');
        } else {
            logger.warn(`Greeks values invalid or quantity zero for leg index ${g.leg_index}, skipping totals. Greeks:`, gv, `Quantity: ${quantity}`);
            row.classList.add('greeks-skipped');
            skippedLegsCount++;
             // Add a visual indicator maybe?
             row.style.opacity = '0.6';
             row.title = 'Greeks invalid or quantity zero, excluded from totals.';
        }
    });

    // --- Create Footer with Totals ---
    const tfoot = tableElement.createTFoot();
    const footerRow = tfoot.insertRow();
    footerRow.className = 'totals-row';

    if (hasCalculatedGreeks) {
        const headerCell = footerRow.insertCell();
        headerCell.colSpan = 4; // Span first 4 columns
        headerCell.textContent = 'Total Portfolio Greeks';
        headerCell.style.textAlign = 'right';
        headerCell.style.fontWeight = 'bold';

        // Display calculated portfolio totals
        footerRow.insertCell().textContent = formatNumber(totals.delta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.gamma, 4);
        footerRow.insertCell().textContent = formatNumber(totals.theta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.vega, 4);
        footerRow.insertCell().textContent = formatNumber(totals.rho, 4);
    } else if (totalLegsProcessed > 0) {
         // Data received, but no valid Greeks could be calculated for totals
         const cell = footerRow.insertCell();
         cell.colSpan = 9; // Span all columns
         cell.textContent = `No valid Greeks data found for total calculation (${skippedLegsCount}/${totalLegsProcessed} legs skipped).`;
         cell.style.textAlign = 'center';
    } else {
         // This case should be handled earlier (empty greeksList)
    }
}


// ===============================================================
// Misc Helpers
// ===============================================================

/** Finds the strike closest to the current spot price */
function findATMStrike(strikes = [], spotPrice) {
    if (!Array.isArray(strikes) || strikes.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         logger.warn("Cannot find ATM strike: Invalid strikes array or spot price.", { strikes, spotPrice });
         return null;
    }
    // Ensure strikes are numbers for comparison
    const numericStrikes = strikes.map(Number).filter(n => !isNaN(n) && isFinite(n));
    if(numericStrikes.length === 0) {
        logger.warn("Cannot find ATM strike: No valid numeric strikes found.");
        return null;
    }

    // Use reduce to find the strike with the minimum absolute difference
    const closestStrike = numericStrikes.reduce((prev, curr) =>
        Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev
    );
    logger.debug(`Calculated ATM strike: ${closestStrike} for spot price: ${spotPrice}`);
    return closestStrike;
}