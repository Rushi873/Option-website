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
    statusMessageContainer: '#statusMessage', // General status messages (optional)
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
// Structure for strategyPositions items (to match backend inputs after processing)
// Each item will store details needed for UI AND for backend payload creation
let strategyPositions = []; // Holds objects like: { strike_price: number, expiry_date: string, option_type: 'CE'|'PE', lots: number, tr_type: 'b'|'s', last_price: number, iv: number|null, days_to_expiry: number|null, lot_size: number|null }
let activeAsset = null;
let autoRefreshIntervalId = null; // Timer ID for auto-refresh
let previousOptionChainData = {}; // Store previous chain data for highlighting
let previousSpotPrice = 0; // Store previous spot price for highlighting

// ===============================================================
// Utility Functions (Enhanced & Corrected)
// ===============================================================

/** Safely formats a number or returns a fallback string, handling backend specials */
function formatNumber(value, decimals = 2, fallback = "N/A") {
    if (value === null || typeof value === 'undefined') { return fallback; }
    // Handle specific string representations from backend (like infinity, loss)
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
        if (["∞", "INFINITY"].includes(upperVal)) return "∞";
        if (["-∞", "-INFINITY"].includes(upperVal)) return "-∞";
        if (["N/A", "UNDEFINED", "LOSS", "0 / 0", "∞ / ∞", "LOSS / ∞"].includes(upperVal)) return value; // Pass through specific statuses/ratios
    }
    const num = Number(value);
    if (!isNaN(num) && isFinite(num)) { // Check for finite numbers only
        // Use locale string for commas and specified decimals
        return num.toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }
    // Handle explicit Infinities after converting potential string representations
    if (num === Infinity) return "∞";
    if (num === -Infinity) return "-∞";

    // If it's a string we didn't specifically handle and couldn't convert to a finite number
    if (typeof value === 'string') return value; // Return the original string (might be a label)

    return fallback; // Fallback for other non-numeric types or non-finite numbers
}

/** Safely formats currency, handling backend specials */
function formatCurrency(value, decimals = 2, fallback = "N/A", prefix = "₹") {
     // Handle specific non-numeric strings first
    if (typeof value === 'string') {
        const upperVal = value.toUpperCase();
        // Keep R:R strings as they are, don't prefix currency
         if (["∞", "INFINITY", "-∞", "-INFINITY", "N/A", "UNDEFINED", "LOSS", "0 / 0", "∞ / ∞", "LOSS / ∞"].includes(upperVal)) {
             return value;
         }
    }
    // Try formatting as number, use null fallback to distinguish between valid 0 and error
    const formattedNumberResult = formatNumber(value, decimals, null); // Pass null as fallback

    if (formattedNumberResult !== null && !["∞", "-∞"].includes(formattedNumberResult)) { // Don't prefix infinity
        // toLocaleString usually handles negative signs correctly
        return `${prefix}${formattedNumberResult}`;
    }
    // Return ∞/-∞ as is, or the fallback if formatting failed
    return formattedNumberResult === null ? fallback : formattedNumberResult;
}

/** Helper to display formatted metric/value in a UI element */
function displayMetric(value, targetElementSelector, prefix = '', suffix = '', decimals = 2, isCurrency = false, fallback = "N/A") {
     const element = document.querySelector(targetElementSelector);
     if (!element) {
        logger.warn(`displayMetric: Element not found for selector "${targetElementSelector}"`);
        return;
     }
     const formatFunc = isCurrency ? formatCurrency : formatNumber;
     // Pass the fallback value to the formatting functions
     const formattedValue = formatFunc(value, decimals, fallback, isCurrency ? "₹" : ""); // Let formatCurrency handle prefix

     // Construct final string
     element.textContent = `${prefix}${formattedValue}${suffix}`;
}

/** Sets the loading/error/content/hidden state for an element. */
function setElementState(selectorOrElement, state, message = 'Loading...') {
    const element = (typeof selectorOrElement === 'string') ? document.querySelector(selectorOrElement) : selectorOrElement;
    if (!element) { logger.warn(`setElementState: Element not found for "${selectorOrElement}"`); return; }

    const isSelect = element.tagName === 'SELECT';
    const isButton = element.tagName === 'BUTTON';
    const isTbody = element.tagName === 'TBODY';
    const isTable = element.tagName === 'TABLE';
    const isContainer = element.tagName === 'DIV' || element.tagName === 'SECTION' || element.classList.contains('chart-container') || element.tagName === 'UL' || element.tagName === 'DETAILS';
    const isSpan = element.tagName === 'SPAN';
    const isGlobalError = element.id === SELECTORS.globalErrorDisplay.substring(1);

    // Reset states and styles
    element.classList.remove('loading', 'error', 'loaded', 'hidden');
    if (isSelect || isButton) element.disabled = false;
    element.style.display = ''; // Default display
    if (isGlobalError) { element.style.display = 'none'; } // Default hidden

    // Default Colspan (adjust per table if needed)
    let defaultColspan = 7; // Option Chain default
    if (element.closest(SELECTORS.greeksTable)) defaultColspan = 9;
    if (element.closest('.charges-table')) defaultColspan = 12;

    // Clear specific content based on type only when NOT setting error
    if (state !== 'error' && state !== 'loading') {
        if (isTbody) element.innerHTML = ''; // Clear table body
        else if (isContainer && !element.closest(SELECTORS.metricsList)) { // Avoid clearing metrics spans
            // Clear containers, but be careful with specific ones like metrics list
            // element.innerHTML = '';
        }
    }

    switch (state) {
        case 'loading':
            element.classList.add('loading');
            if (isSelect) element.innerHTML = `<option>${message}</option>`;
            else if (isTbody) element.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`;
            else if (isTable) {
                const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="loading-text">${message}</td></tr>`;
                const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = ''; // Clear footer
            }
            else if (isContainer) element.innerHTML = `<div class="loading-text" style="padding: 20px; text-align: center;">${message}</div>`;
            else if (isSpan) element.textContent = '...'; // Loading indicator for spans
            else if (!isButton && !isGlobalError) element.textContent = message;
            if (isSelect || isButton) element.disabled = true;
            if (isGlobalError) { element.textContent = message; element.style.display = 'block'; }
            break;
        case 'error':
            element.classList.add('error');
            const displayMessage = `${message}`; // Use message directly
            if (isSelect) { element.innerHTML = `<option>${displayMessage}</option>`; element.disabled = true; }
            else if (isTbody) { element.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`; }
            else if (isTable) {
                 const tbody = element.querySelector('tbody'); if (tbody) tbody.innerHTML = `<tr><td colspan="${defaultColspan}" class="error-message">${displayMessage}</td></tr>`;
                 const tfoot = element.querySelector('tfoot'); if (tfoot) tfoot.innerHTML = ''; // Clear footer
            }
            else if (isContainer) { element.innerHTML = `<p class="error-message" style="text-align: center; padding: 20px;">${displayMessage}</p>`; }
            else if (isSpan) { element.textContent = 'Error'; element.classList.add('error-message'); } // Concise error for spans
            else { element.textContent = displayMessage; element.classList.add('error-message'); } // Other text elements
            if (isGlobalError) { element.style.display = 'block'; element.textContent = displayMessage; } // Ensure global error is visible and has message
            break;
        case 'content':
            element.classList.add('loaded');
            // Calling function will set the actual content
            if (isGlobalError) element.style.display = 'none'; // Hide global error when other content is loaded
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
        placeholderOption.disabled = true;
        selectElement.appendChild(placeholderOption);
    }

    items.forEach(item => {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });

    // Try to restore previous selection, or set the default, or leave placeholder selected
    let valueSet = false;
    if (items.includes(currentValue)) {
        selectElement.value = currentValue;
        valueSet = true;
    } else if (defaultSelection !== null && items.includes(String(defaultSelection))) {
        selectElement.value = String(defaultSelection);
        valueSet = true;
    }

    if (!valueSet && placeholder) {
        selectElement.value = ""; // Ensure placeholder is selected if nothing else matches
    }

    selectElement.disabled = false;
}

/** Fetches data from the API with enhanced error handling. */
async function fetchAPI(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;
    const defaultHeaders = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    options.headers = { ...defaultHeaders, ...options.headers };
    const method = options.method || 'GET';
    const requestBody = options.body ? JSON.parse(options.body) : ''; // Log parsed body if exists
    logger.debug(`fetchAPI Request: ${method} ${url}`, requestBody || '(No Body)');

    try {
        const response = await fetch(url, options);
        let responseData = null;
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
             try {
                 responseData = await response.json();
             } catch (jsonError) {
                  logger.error(`API Error (${method} ${url} - ${response.status}): Failed to parse JSON response. Body: ${await response.text().catch(() => '[Could not read body]')}`, jsonError);
                  throw new Error(`Invalid JSON response from server (Status: ${response.status})`);
             }
        } else if (response.status !== 204) { // Handle non-JSON, non-empty responses
             const textResponse = await response.text().catch(() => '[Could not read body]');
             logger.warn(`Received non-JSON response from ${method} ${url} (Status: ${response.status}). Body: ${textResponse.substring(0, 100)}...`);
             // Decide if text response should be returned or treated as error
             // For now, we assume JSON is expected for data endpoints
             if (!response.ok) { // Treat non-ok text responses as errors
                  throw new Error(textResponse || `HTTP error ${response.status}`);
             }
             // If response.ok, maybe return text? Or null? Let's assume null for now.
             responseData = null;
        }

        logger.debug(`fetchAPI Response Status: ${response.status} for ${method} ${url}`);

        if (!response.ok) {
            const errorMessage = responseData?.detail // FastAPI default
                              || responseData?.message // Common alternative
                              || responseData?.error // Another common alternative
                              || response.statusText // Standard HTTP status text
                              || `HTTP error ${response.status}`;
            logger.error(`API Error (${method} ${url} - ${response.status}): ${errorMessage}`, responseData);
            // Don't automatically set global error here, let the caller decide based on context
            throw new Error(errorMessage); // Throw error with specific message
        }

        // Clear global error ONLY if fetch was successful
        setElementState(SELECTORS.globalErrorDisplay, 'hidden');
        logger.debug(`fetchAPI Response Data:`, responseData);
        return responseData; // Return parsed data (or null)

    } catch (error) {
        // Catch network errors (e.g., DNS, connection refused) or errors thrown above
        logger.error(`Fetch/Network Error or API Error (${method} ${url}):`, error);
        // Set global error for these types of failures
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Network/API Error: ${error.message || 'Could not connect or invalid response'}`);
        throw error; // Re-throw for specific UI handling if needed
    }
}

/** Applies a temporary highlight effect to an element */
function highlightElement(element) {
    if (!element) return;
    element.classList.remove('value-changed'); // Remove previous highlight instantly
    void element.offsetWidth; // Trigger reflow
    element.classList.add('value-changed'); // Add the class to start animation
    setTimeout(() => {
        element.classList.remove('value-changed');
    }, HIGHLIGHT_DURATION_MS);
}


// ===============================================================
// Initialization & Event Listeners
// ===============================================================

document.addEventListener("DOMContentLoaded", () => {
    logger.info("DOM Ready. Initializing...");
    // Ensure dependent functions are defined before calling initializePage/setupEventListeners
    if (typeof loadMarkdownParser === 'function' &&
        typeof initializePage === 'function' &&
        typeof setupEventListeners === 'function') {
            loadMarkdownParser();
            initializePage();
            setupEventListeners();
    } else {
        console.error("CRITICAL ERROR: Core initialization functions are not defined. Script cannot run.");
        alert("A critical error occurred loading the page script. Please refresh.");
    }
});

async function initializePage() {
    logger.info("Initializing page: Setting initial states...");
    // Ensure resetResultsUI is defined
    if (typeof resetResultsUI === 'function') {
        resetResultsUI();
    } else {
        logger.error("initializePage: resetResultsUI function not defined!"); return;
    }
    // Set initial placeholder states for elements not covered by resetResultsUI
    setElementState(SELECTORS.expiryDropdown, 'content');
    document.querySelector(SELECTORS.expiryDropdown).innerHTML = '<option value="">-- Select Asset --</option>';
    setElementState(SELECTORS.optionChainTableBody, 'content');
    document.querySelector(SELECTORS.optionChainTableBody).innerHTML = '<tr><td colspan="7">Select Asset and Expiry</td></tr>';
    setElementState(SELECTORS.spotPriceDisplay, 'content');
    document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
    setElementState(SELECTORS.analysisResultContainer, 'content');
    document.querySelector(SELECTORS.analysisResultContainer).innerHTML = '<p class="placeholder-text">Select an asset to load analysis...</p>';
    setElementState(SELECTORS.newsResultContainer, 'content');
    document.querySelector(SELECTORS.newsResultContainer).innerHTML = '<p class="placeholder-text">Select an asset to load news...</p>';
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Ensure hidden

    try {
        // Ensure loadAssets is defined
        if (typeof loadAssets !== 'function') throw new Error("loadAssets function not defined!");
        const defaultAsset = await loadAssets();

        if (defaultAsset) {
            logger.info(`Default asset determined: ${defaultAsset}. Fetching initial data...`);
            // Ensure handleAssetChange is defined
            if (typeof handleAssetChange !== 'function') throw new Error("handleAssetChange function not defined!");
            await handleAssetChange(); // Await the full process
        } else {
            logger.warn("No default asset set or assets failed to load. Waiting for user selection.");
        }

    } catch (error) {
        logger.error("Page Initialization failed:", error);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Initialization Error: ${error.message}`);
        setElementState(SELECTORS.assetDropdown, 'error', 'Failed');
        setElementState(SELECTORS.expiryDropdown, 'error', 'Failed');
        setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error');
    }
    logger.info("Initialization sequence complete.");
}


function setupEventListeners() {
    logger.info("Setting up event listeners...");
    // Ensure event handlers are defined before assigning them
    if (typeof handleAssetChange !== 'function') { logger.error("handleAssetChange not defined for listener!"); return; }
    if (typeof handleExpiryChange !== 'function') { logger.error("handleExpiryChange not defined for listener!"); return; }
    if (typeof fetchPayoffChart !== 'function') { logger.error("fetchPayoffChart not defined for listener!"); return; }
    if (typeof clearAllPositions !== 'function') { logger.error("clearAllPositions not defined for listener!"); return; }
    if (typeof handleStrategyTableChange !== 'function') { logger.error("handleStrategyTableChange not defined for listener!"); return; }
    if (typeof handleStrategyTableClick !== 'function') { logger.error("handleStrategyTableClick not defined for listener!"); return; }
    if (typeof handleOptionChainClick !== 'function') { logger.error("handleOptionChainClick not defined for listener!"); return; }


    document.querySelector(SELECTORS.assetDropdown)?.addEventListener("change", handleAssetChange);
    document.querySelector(SELECTORS.expiryDropdown)?.addEventListener("change", handleExpiryChange);
    document.querySelector(SELECTORS.updateChartButton)?.addEventListener("click", fetchPayoffChart);
    document.querySelector(SELECTORS.clearPositionsButton)?.addEventListener("click", clearAllPositions);

    // Event delegation for strategy table interaction
    const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (strategyTableBody) {
        strategyTableBody.addEventListener('input', handleStrategyTableChange);
        strategyTableBody.addEventListener('click', handleStrategyTableClick);
    } else { logger.warn("Strategy table body not found for event listeners."); }

    // Event delegation for option chain table clicks
     const optionChainTableBody = document.querySelector(SELECTORS.optionChainTableBody);
     if (optionChainTableBody) {
         optionChainTableBody.addEventListener('click', handleOptionChainClick);
     } else { logger.warn("Option chain table body not found for event listeners."); }
}

/** Loads assets, populates dropdown, sets default, and returns the default asset value. */
async function loadAssets() {
    logger.info("Loading assets...");
    setElementState(SELECTORS.assetDropdown, 'loading');
    let defaultAsset = null; // Variable to store the default asset

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
        logger.error("Failed to load assets:", error);
        populateDropdown(SELECTORS.assetDropdown, [], "-- Error Loading --");
        setElementState(SELECTORS.assetDropdown, 'error', `Error`);
        // Let initializePage handle the global error display
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
        strategyTableBody.addEventListener('input', handleStrategyTableChange); // Use 'input' for lots
        strategyTableBody.addEventListener('click', handleStrategyTableClick); // For buttons
    }

    // Event delegation for option chain table clicks
     const optionChainTableBody = document.querySelector(SELECTORS.optionChainTableBody);
     if (optionChainTableBody) {
         optionChainTableBody.addEventListener('click', handleOptionChainClick);
     }
}

function loadMarkdownParser() {
    // Check if marked is already loaded
    if (typeof marked !== 'undefined') {
        logger.info("Markdown parser (marked.js) already available.");
        return;
    }
    try {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
        script.async = true;
        script.onload = () => logger.info("Markdown parser (marked.js) loaded dynamically.");
        script.onerror = () => logger.error("Failed to load Markdown parser (marked.js). Analysis rendering may fail.");
        document.head.appendChild(script);
    } catch (e) {
         logger.error("Error creating script tag for marked.js", e);
    }
}

// ===============================================================
// Auto-Refresh Logic (Corrected)
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

    logger.debug(`Auto-refreshing live data (Spot & Chain) for ${activeAsset}...`);

    // Fetch ONLY dynamic data concurrently
    const results = await Promise.allSettled([
        fetchNiftyPrice(activeAsset, true), // Pass true for refresh call (handles spot display)
        fetchOptionChain(false, true)       // No scroll, is refresh call (handles chain display)
    ]);

    // Log errors from settled promises if any
    results.forEach((result, index) => {
        if (result.status === 'rejected') {
            const source = index === 0 ? 'Spot price' : 'Option chain';
             // Log as warning, as refresh might fail temporarily
             logger.warn(`Auto-refresh: ${source} fetch failed: ${result.reason?.message || result.reason}`);
             // Optionally display a subtle error indicator for the failing component
             if (index === 0) setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Err'); // Brief error
             // Don't set table to full error on refresh failure, just log
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
    logger.info(`Starting auto-refresh every ${REFRESH_INTERVAL_MS / 1000}s for ${activeAsset}`);
    // Initialize previous values before starting
    previousSpotPrice = currentSpotPrice; // Store current before first refresh
    // previousOptionChainData is updated within fetchOptionChain
    autoRefreshIntervalId = setInterval(refreshLiveData, REFRESH_INTERVAL_MS);
}

function stopAutoRefresh() {
    if (autoRefreshIntervalId) {
        clearInterval(autoRefreshIntervalId);
        autoRefreshIntervalId = null;
        logger.info("Auto-refresh stopped.");
    }
}

// ===============================================================
// Event Handlers & Data Fetching Logic (Corrected)
// ===============================================================

/** Handles asset dropdown change - Refactored for clarity */
async function handleAssetChange() {
    const assetDropdown = document.querySelector(SELECTORS.assetDropdown);
    const selectedAsset = assetDropdown?.value;

    // Prevent running if asset is empty or hasn't actually changed
    if (!selectedAsset) {
        logger.info("Asset selection cleared.");
        activeAsset = null; // Clear active asset
        stopAutoRefresh();
        resetPageToInitialState(); // Reset UI completely
        return;
    }
    if (selectedAsset === activeAsset) {
        logger.debug(`handleAssetChange skipped: Asset unchanged (${selectedAsset}).`);
        return;
    }

    logger.info(`Asset changed to: ${selectedAsset}. Fetching all related data...`);
    activeAsset = selectedAsset; // Update global state *now*
    stopAutoRefresh(); // Stop refresh for the old asset

    // Clear previous interaction data
    previousOptionChainData = {};
    previousSpotPrice = 0;
    currentSpotPrice = 0;

    // Reset results UI AND strategy input table
    resetResultsUI();

    // Set loading states for ALL sections being fetched
    setLoadingStateForAssetChange();

    // Optional Debug Call to backend
    sendDebugAssetSelection(activeAsset);

    try {
        // Fetch Spot, Expiry, News, and Analysis concurrently
        const [spotResult, expiryResult, analysisResult, newsResult] = await Promise.allSettled([
            fetchNiftyPrice(activeAsset), // Initial fetch for this asset
            fetchExpiries(activeAsset),   // Fetches expiries AND triggers chain load if needed
            fetchAnalysis(activeAsset),
            fetchNews(activeAsset)
        ]);

        let hasCriticalError = false;

        // Process critical results (Spot/Expiry) - errors are logged within fetch functions
        if (spotResult.status === 'rejected') {
            hasCriticalError = true;
            // Error state already set by fetchNiftyPrice
        }
        if (expiryResult.status === 'rejected') {
            hasCriticalError = true;
            // Error states set by fetchExpiries
        }

        // Log non-critical failures (News/Analysis) - errors handled internally
        if (analysisResult.status === 'rejected') { logger.error(`Analysis fetch failed during asset change: ${analysisResult.reason?.message || analysisResult.reason}`); }
        if (newsResult.status === 'rejected') { logger.error(`News fetch failed during asset change: ${newsResult.reason?.message || newsResult.reason}`); }

        // Start auto-refresh ONLY if critical data loaded successfully
        if (!hasCriticalError) {
            logger.info(`Essential data loaded for ${activeAsset}. Starting auto-refresh.`);
            startAutoRefresh(); // Start refresh for the new asset
        } else {
             logger.error(`Failed to load essential data (spot/expiries) for ${activeAsset}. Auto-refresh NOT started.`);
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load essential market data for ${activeAsset}. Check asset validity and network.`);
             // Option chain is likely already showing an error from fetchExpiries failure
        }

    } catch (err) {
        // Catch unexpected errors during orchestration
        logger.error(`Unexpected error during handleAssetChange for ${activeAsset}:`, err);
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load data for ${activeAsset}. ${err.message}`);
        stopAutoRefresh(); // Stop refresh on major load error
        setElementState(SELECTORS.expiryDropdown, 'error', 'Failed');
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Failed');
        setElementState(SELECTORS.spotPriceDisplay, 'error', 'Spot: Error');
        setElementState(SELECTORS.analysisResultContainer, 'error', 'Failed');
        setElementState(SELECTORS.newsResultContainer, 'error', 'Failed');
    }
}

/** Helper to reset UI when asset is deselected */
function resetPageToInitialState() {
    populateDropdown(SELECTORS.expiryDropdown, [], "-- Select Asset First --");
    setElementState(SELECTORS.optionChainTableBody, 'content');
    document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Asset</td></tr>`;
    setElementState(SELECTORS.spotPriceDisplay, 'content');
    document.querySelector(SELECTORS.spotPriceDisplay).textContent = 'Spot Price: -';
    resetResultsUI(); // Clears strategy and all results including news/analysis placeholders
    setElementState(SELECTORS.globalErrorDisplay, 'hidden');
}

/** Helper to set loading states during asset change */
function setLoadingStateForAssetChange() {
    setElementState(SELECTORS.expiryDropdown, 'loading');
    setElementState(SELECTORS.optionChainTableBody, 'loading');
    setElementState(SELECTORS.analysisResultContainer, 'loading');
    setElementState(SELECTORS.newsResultContainer, 'loading');
    setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot: ...'); // Concise loading
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear previous global error
    // Reset calculation outputs as well
    resetCalculationOutputsUI();
}

/** Helper function to send debug asset selection */
async function sendDebugAssetSelection(asset) {
    try {
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST', body: JSON.stringify({ asset: asset })
        });
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        logger.error("Failed to send debug asset selection:", debugErr.message);
        // Non-critical error, don't need to alert user
    }
}

/** Handles expiry dropdown change */
async function handleExpiryChange() {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    previousOptionChainData = {}; // Clear previous chain data on expiry change
    if (!expiry) {
        setElementState(SELECTORS.optionChainTableBody, 'content');
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Expiry</td></tr>`;
        return;
    }
    logger.info(`Expiry changed to: ${expiry}. Fetching option chain...`);
    await fetchOptionChain(true); // Fetch new chain and scroll to ATM
}

/** Fetches and populates expiry dates, triggers option chain load */
async function fetchExpiries(asset) {
    if (!asset) return; // Should not happen if called from handleAssetChange
    setElementState(SELECTORS.expiryDropdown, 'loading');
    try {
        const data = await fetchAPI(`/expiry_dates?asset=${encodeURIComponent(asset)}`);
        const expiries = data?.expiry_dates || [];
        populateDropdown(SELECTORS.expiryDropdown, expiries, "-- Select Expiry --", expiries[0]); // Select first expiry by default
        setElementState(SELECTORS.expiryDropdown, 'content');

        const selectedExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
        if (selectedExpiry) {
            // Option chain will be loaded because the dropdown 'change' event fires
            // OR because we explicitly call it if there was only one expiry set as default
            if (expiries.length > 0 && selectedExpiry === expiries[0]) {
                 logger.info(`Default expiry ${selectedExpiry} selected. Fetching chain...`);
                 await fetchOptionChain(true); // Manually trigger for the selected default
            }
        } else {
             logger.warn(`No expiry dates found for asset: ${asset}`);
             setElementState(SELECTORS.expiryDropdown, 'error', 'No Expiries'); // Indicate no expiries found
             setElementState(SELECTORS.optionChainTableBody, 'content'); // Set chain to content but show message
             document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">No expiry dates found for ${asset}</td></tr>`;
             throw new Error("No expiry dates found."); // Throw error to signal issue to caller
        }

    } catch (error) {
        logger.error(`Error fetching expiries for ${asset}: ${error.message}`);
        populateDropdown(SELECTORS.expiryDropdown, [], "-- Error Loading Expiries --");
        setElementState(SELECTORS.expiryDropdown, 'error', `Error`);
        setElementState(SELECTORS.optionChainTableBody, 'error', `Failed to load expiry dates for ${asset}.`);
        throw error; // Re-throw so handleAssetChange knows about the failure
    }
}

/** Fetches and displays the spot price */
async function fetchNiftyPrice(asset, isRefresh = false) {
    if (!asset) return;
    const priceElement = document.querySelector(SELECTORS.spotPriceDisplay);

    // Only show main loading text on initial load
    if (!isRefresh) {
        setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot: ...');
    }

    try {
        const data = await fetchAPI(`/get_spot_price?asset=${encodeURIComponent(asset)}`);
        const newSpotPrice = data?.spot_price;

        if (newSpotPrice === null || typeof newSpotPrice === 'undefined' || isNaN(parseFloat(newSpotPrice))) {
             throw new Error("Spot price not available or invalid from API.");
        }
        const validSpotPrice = parseFloat(newSpotPrice);

        // Store previous price *before* updating current
        const previousValue = currentSpotPrice;
        currentSpotPrice = validSpotPrice; // Update global state

        if (priceElement) {
            priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}`;
            setElementState(SELECTORS.spotPriceDisplay, 'content'); // Ensure state is content after update

            // Highlight only on refresh if value changed significantly
            if (isRefresh && Math.abs(currentSpotPrice - previousValue) > 0.001 && previousValue !== 0) {
                 logger.debug(`Spot price changed: ${previousValue.toFixed(2)} -> ${currentSpotPrice.toFixed(2)}`);
                 highlightElement(priceElement);
            }
        }
    } catch (error) {
         logger.error(`Error fetching spot price for ${asset}:`, error.message);
         currentSpotPrice = 0; // Reset spot price on error
         if (!isRefresh) {
             setElementState(SELECTORS.spotPriceDisplay, 'error', `Spot: Error`);
             // Re-throw only on initial load failure so handleAssetChange knows
             throw error;
         } else {
             // For refresh errors, just log and maybe show subtle error on display
             logger.warn(`Spot Price refresh Error (${asset}):`, error.message);
             if (priceElement) priceElement.classList.add('error-message'); // Add error class, but keep last value?
         }
    }
}

/** Fetches and displays the option chain, optionally highlights changes */
async function fetchOptionChain(scrollToATM = false, isRefresh = false) {
    const asset = activeAsset; // Use global activeAsset
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    const currentTbody = document.querySelector(SELECTORS.optionChainTableBody);

    if (!currentTbody) { logger.error("Option chain tbody element not found."); return; }

    // --- Initial Checks & Loading State ---
    if (!asset || !expiry) {
        currentTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">Select Asset and Expiry</td></tr>`;
        if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
        previousOptionChainData = {}; // Clear previous data if selection invalid
        return;
    }
    if (!isRefresh) {
        setElementState(SELECTORS.optionChainTableBody, 'loading', 'Loading Chain...');
    }

    try {
        // --- Ensure Spot Price Available for ATM ---
        if (currentSpotPrice <= 0 && scrollToATM && !isRefresh) {
            logger.info("Spot price needed for ATM scroll, attempting fetch...");
            try { await fetchNiftyPrice(asset); } catch (spotError) { /* Handled by fetchNiftyPrice */ }
            if (currentSpotPrice <= 0) { logger.warn("Spot price unavailable, cannot calculate ATM strike."); scrollToATM = false; }
        }

        // --- Fetch Option Chain Data ---
        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);
        const currentChainData = data?.option_chain;

        // --- Handle Empty/Invalid Data ---
        if (!currentChainData || typeof currentChainData !== 'object' || Object.keys(currentChainData).length === 0) {
            logger.warn(`No option chain data available for ${asset} on ${expiry}.`);
            currentTbody.innerHTML = `<tr><td colspan="7" class="placeholder-text">No option chain data found for ${asset} on ${expiry}</td></tr>`;
            if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
            previousOptionChainData = {}; // Clear previous data
            return;
        }

        // --- Render Table & Handle Highlights ---
        const strikeStringKeys = Object.keys(currentChainData).sort((a, b) => Number(a) - Number(b));
        const atmStrikeObjectKey = currentSpotPrice > 0 ? findATMStrikeAsStringKey(strikeStringKeys, currentSpotPrice) : null;

        const newTbody = document.createElement('tbody'); // Build in memory

        strikeStringKeys.forEach((strikeStringKey) => {
            const optionDataForStrike = currentChainData[strikeStringKey];
            const optionData = (typeof optionDataForStrike === 'object' && optionDataForStrike !== null)
                                ? optionDataForStrike : { call: null, put: null };
            const call = optionData.call || {};
            const put = optionData.put || {};
            const strikeNumericValue = Number(strikeStringKey);
            const prevOptionData = previousOptionChainData[strikeStringKey] || { call: {}, put: {} };
            const prevCall = prevOptionData.call || {};
            const prevPut = prevOptionData.put || {};

            const tr = newTbody.insertRow(); // Add row to the new tbody
            tr.dataset.strike = strikeNumericValue; // Store numeric strike for ATM calc/scroll
             tr.dataset.strikeKey = strikeStringKey; // Store original key if needed later

            if (atmStrikeObjectKey !== null && strikeStringKey === atmStrikeObjectKey) {
                tr.classList.add("atm-strike");
            }

            const columns = [
                { class: 'call clickable price', type: 'CE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') },
                { class: 'call oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'call iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'strike', isStrike: true, format: val => formatNumber(val, val % 1 === 0 ? 0 : 2) }, // Format strike based on decimal
                { class: 'put iv', dataKey: 'implied_volatility', format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'put oi', dataKey: 'open_interest', format: val => formatNumber(val, 0, '-') },
                { class: 'put clickable price', type: 'PE', dataKey: 'last_price', format: val => formatNumber(val, 2, '-') },
            ];

            columns.forEach(col => {
                try {
                    const td = tr.insertCell();
                    td.className = col.class;
                    let currentValue;
                    let sourceObj;
                    let prevDataObject;

                    if (col.isStrike) {
                        currentValue = strikeNumericValue;
                        sourceObj = null; // Strike doesn't have a source object
                        prevDataObject = null; // Strike doesn't change
                    } else {
                        sourceObj = col.class.includes('call') ? call : put;
                        prevDataObject = col.class.includes('call') ? prevCall : prevPut;
                        currentValue = (typeof sourceObj === 'object' && sourceObj !== null) ? sourceObj[col.dataKey] : undefined;
                    }

                    td.textContent = col.format(currentValue);

                    // Add data attributes for adding positions
                    if (col.type && typeof sourceObj === 'object' && sourceObj !== null) {
                        td.dataset.type = col.type;
                        // Use ?? for safe fallback to 0 or null
                        td.dataset.price = String(sourceObj['last_price'] ?? 0);
                        td.dataset.iv = String(sourceObj['implied_volatility'] ?? ''); // Store IV as string or empty
                    }

                    // Highlighting Logic
                    if (isRefresh && !col.isStrike && typeof prevDataObject === 'object' && prevDataObject !== null) {
                        let previousValue = prevDataObject[col.dataKey];
                        let changed = false;
                        const currentExists = currentValue !== null && typeof currentValue !== 'undefined';
                        const previousExists = previousValue !== null && typeof previousValue !== 'undefined';

                        if (currentExists && previousExists) {
                            if (typeof currentValue === 'number' && typeof previousValue === 'number') {
                                changed = Math.abs(currentValue - previousValue) > 0.001; // Tolerance for float comparison
                            } else {
                                changed = currentValue !== previousValue; // String comparison
                            }
                        } else if (currentExists !== previousExists) { // Value appeared or disappeared
                            changed = true;
                        }

                        if (changed) {
                            highlightElement(td);
                        }
                    } else if (isRefresh && !col.isStrike && currentValue !== undefined && currentValue !== null && !prevDataObject) {
                         highlightElement(td); // Highlight if new data appeared where there was none before
                    }


                } catch (cellError) {
                    logger.error(`Error rendering cell for Strike: ${strikeStringKey}, Column Key: ${col.dataKey}`, cellError);
                    const errorTd = tr.insertCell();
                    errorTd.textContent = 'ERR'; errorTd.className = col.class + ' error-message';
                }
            });
        }); // End strikeStringKeys.forEach

        // --- Replace Table Body & Update State ---
        currentTbody.parentNode.replaceChild(newTbody, currentTbody); // Replace old tbody with new one
        if (!isRefresh) {
            setElementState(SELECTORS.optionChainTableBody, 'content'); // Use selector for state setting
        }
        previousOptionChainData = currentChainData; // Update previous data cache

        // --- Scroll to ATM ---
        // Use the corrected logic with refined ATM row finding
        if (scrollToATM && atmStrikeObjectKey !== null && !isRefresh) {
            // Pass the new tbody context to the scroll function
             triggerATMScroll(newTbody, atmStrikeObjectKey);
        }

    } catch (error) { // Outer catch
        logger.error(`Error during fetchOptionChain execution for ${activeAsset}/${expiry}:`, error);
        if (currentTbody) { // Check currentTbody again
            currentTbody.innerHTML = `<tr><td colspan="7" class="error-message">Chain Error: ${error.message}</td></tr>`;
        }
        if (!isRefresh) { setElementState(SELECTORS.optionChainTableBody, 'error', `Chain Error`); }
        else { logger.warn(`Option Chain refresh failed: ${error.message}`); }
        previousOptionChainData = {};
    }
}

/** Helper function to trigger the ATM scroll logic */
function triggerATMScroll(tbodyElement, atmKeyToUse) {
    setTimeout(() => {
        try {
            const numericATMStrike = Number(atmKeyToUse);
            logger.debug(`Scroll Timeout: Finding ATM row data-strike="${numericATMStrike}". Tbody has ${tbodyElement.rows.length} rows.`);

            if (isNaN(numericATMStrike)) {
                logger.warn(`Invalid ATM key passed to scroll timeout: ${atmKeyToUse}`);
                return;
            }

            // --- Refined ATM Row Finding ---
            let atmRow = tbodyElement.querySelector(`tr[data-strike="${numericATMStrike}"]`); // Try exact numeric match first

            if (!atmRow) { // Try original string key if numeric failed (e.g., "22500.0")
                logger.debug(`Numeric match failed, trying original key: ${atmKeyToUse}`);
                atmRow = tbodyElement.querySelector(`tr[data-strike-key="${atmKeyToUse}"]`); // Use data-strike-key
            }

             // If still not found, try closest match (less reliable, use with caution)
            if (!atmRow) {
                logger.debug(`String/Numeric key match failed, checking all rows for closest match`);
                const allRows = tbodyElement.querySelectorAll('tr[data-strike]');
                let closestRow = null;
                let closestDiff = Infinity;

                allRows.forEach(row => {
                    const rowStrike = Number(row.dataset.strike);
                    if (!isNaN(rowStrike)) {
                        const diff = Math.abs(rowStrike - numericATMStrike);
                        if (diff < closestDiff) {
                            closestDiff = diff;
                            closestRow = row;
                        }
                    }
                });
                 // Only use closest match if it's very close (adjust tolerance if needed)
                 if (closestRow && closestDiff < 0.01) {
                     logger.debug(`Using closest match row with difference ${closestDiff}`);
                     atmRow = closestRow;
                 }
            }
            // --- End Refined Finding ---


            if (atmRow) {
                atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
                logger.debug(`Scrolled to ATM strike row: ${atmRow.dataset.strike}`);
                // Optional highlight
                atmRow.classList.add("highlight-atm");
                setTimeout(() => atmRow.classList.remove("highlight-atm"), 2000);
            } else {
                logger.warn(`ATM strike row for key (${atmKeyToUse} / ${numericATMStrike}) not found for scrolling.`);
            }
        } catch (e) {
            logger.error("Error inside scroll timeout:", e);
        }
    }, 250); // Delay slightly for rendering
}


/** Finds the nearest strike key (string) to the spot price */
function findATMStrikeAsStringKey(strikeStringKeys = [], spotPrice) {
    if (!Array.isArray(strikeStringKeys) || strikeStringKeys.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         logger.warn("Cannot find ATM strike key: Invalid input.", { numKeys: strikeStringKeys?.length, spotPrice });
         return null;
    }

    let closestKey = null;
    let minDiff = Infinity;

    for (const key of strikeStringKeys) {
        const numericStrike = Number(key);
        if (!isNaN(numericStrike)) {
            const diff = Math.abs(numericStrike - spotPrice);
            if (diff < minDiff) {
                minDiff = diff;
                closestKey = key; // Store the original string key
            }
        } else {
             logger.warn(`Skipping non-numeric strike key '${key}' during ATM calculation.`);
        }
    }
    logger.debug(`Calculated ATM strike key: ${closestKey} (Min diff: ${minDiff.toFixed(4)}) for spot price: ${spotPrice.toFixed(4)}`);
    return closestKey;
}


/** Fetches stock analysis for the selected asset */
async function fetchAnalysis(asset) {
    const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
    if (!analysisContainer) return;
    if (!asset) {
         analysisContainer.innerHTML = '<p class="placeholder-text">Select an asset to load analysis.</p>';
         setElementState(SELECTORS.analysisResultContainer, 'content');
         return;
    }

    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Fetching analysis...');
    logger.debug(`Fetching analysis for ${asset}...`);

    try {
        // Ensure marked.js is loaded
        if (typeof marked === 'undefined') {
            logger.warn("Waiting for marked.js...");
            await new Promise(resolve => setTimeout(resolve, 500)); // Wait a bit longer
            if (typeof marked === 'undefined') {
                throw new Error("Markdown parser (marked.js) failed to load.");
            }
        }

        const data = await fetchAPI("/get_stock_analysis", {
            method: "POST", body: JSON.stringify({ asset })
        });
        logger.debug(`Received analysis data for ${asset}`);

        const rawAnalysis = data?.analysis || "*Analysis generation failed or returned empty.*";
        // Basic script tag removal (Consider DOMPurify for full sanitization if needed)
        const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
        analysisContainer.innerHTML = marked.parse(potentiallySanitized);
        setElementState(SELECTORS.analysisResultContainer, 'content');
        logger.info(`Successfully rendered analysis for ${asset}`);

    } catch (error) {
        logger.error(`Error fetching or rendering analysis for ${asset}:`, error);
        let displayMessage = `Analysis Error: ${error.message}`;

        // Check for specific backend error messages
        if (error.message && error.message.includes("Essential stock data not found")) {
            displayMessage = `Analysis unavailable: ${error.message}`; // More specific message
        } else if (error.message && error.message.includes("Analysis blocked by content filter")) {
             displayMessage = `Analysis blocked due to content restrictions.`;
        } else if (error.message && (error.message.includes("Analysis generation failed") || error.message.includes("Analysis feature not configured"))) {
             displayMessage = `Analysis Error: ${error.message}`; // Show specific generation failure
        }

        // Display error within the analysis container using setElementState
        setElementState(SELECTORS.analysisResultContainer, 'error', displayMessage);
        // Do not set global error for this specific failure
    }
}

/** Fetches and renders news for the selected asset */
async function fetchNews(asset) {
    if (!asset) return;
    const newsContainer = document.querySelector(SELECTORS.newsResultContainer);
    if (!newsContainer) return;

    setElementState(SELECTORS.newsResultContainer, 'loading', 'Fetching news...');

    try {
        const data = await fetchAPI(`/get_news?asset=${encodeURIComponent(asset)}`);
        const newsItems = data?.news;

        if (Array.isArray(newsItems)) {
            renderNews(newsContainer, newsItems); // Render the fetched items
            setElementState(SELECTORS.newsResultContainer, 'content');
        } else {
            logger.error("Invalid news data format received:", data);
            throw new Error("Invalid news data format from server.");
        }
    } catch (error) {
        logger.error(`Error fetching or rendering news for ${asset}:`, error);
        setElementState(SELECTORS.newsResultContainer, 'error', `News Error: ${error.message}`);
        // Non-critical, don't throw to handleAssetChange
    }
}

/** Renders the news items into the specified container */
function renderNews(containerElement, newsData) {
    containerElement.innerHTML = ""; // Clear previous content

    if (!newsData || newsData.length === 0) {
        containerElement.innerHTML = '<p class="placeholder-text">No recent news found.</p>';
        return;
    }

    // Handle specific error/empty messages returned within the array by the backend
    const firstItemHeadline = newsData[0]?.headline?.toLowerCase() || "";
    if (newsData.length === 1 && (
        firstItemHeadline.includes("error fetching news") ||
        firstItemHeadline.includes("no recent news found") ||
        firstItemHeadline.includes("no relevant news found") ||
        firstItemHeadline.includes("timeout fetching news") ||
        firstItemHeadline.includes("no news data available")
        )) {
        const messageClass = firstItemHeadline.includes("error") || firstItemHeadline.includes("timeout") ? "error-message" : "placeholder-text";
        containerElement.innerHTML = `<p class="${messageClass}">${newsData[0].headline}</p>`;
        return;
    }

    const ul = document.createElement("ul");
    ul.className = "news-list";

    newsData.forEach(item => {
        const li = document.createElement("li");
        li.className = "news-item";

        const headline = document.createElement("div");
        headline.className = "news-headline";
        const link = document.createElement("a");
        link.href = item.link || "#";
        link.textContent = item.headline || "No Title";
        link.target = "_blank";
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

// ===============================================================
// Event Delegation Handlers (Corrected)
// ===============================================================

/** Handles clicks within the option chain table body */
function handleOptionChainClick(event) {
    const targetCell = event.target.closest('td.clickable'); // Target only clickable cells
    if (!targetCell) return; // Ignore clicks elsewhere

    const row = targetCell.closest('tr');
    // Ensure row and necessary data attributes exist
    if (!row || !row.dataset.strike || !targetCell.dataset.type || !targetCell.dataset.price) {
        logger.warn("Could not get required data from clicked option cell.", { rowDataset: row?.dataset, cellDataset: targetCell?.dataset });
        alert("Could not retrieve necessary option details (strike, type, price).");
        return;
    }

    const strike = parseFloat(row.dataset.strike);
    const type = targetCell.dataset.type; // 'CE' or 'PE'
    // Safely parse price, default to 0 if invalid/missing (though check above should prevent)
    const price = parseFloat(targetCell.dataset.price ?? '0');
    // Safely parse IV, default to null if invalid/missing/empty string
    const ivString = targetCell.dataset.iv;
    const iv = (ivString && !isNaN(parseFloat(ivString))) ? parseFloat(ivString) : null;

    if (!isNaN(strike) && type && !isNaN(price)) {
         addPosition(strike, type, price, iv); // Pass potentially null IV
    } else {
        // This case should be rare due to checks above
        logger.error('Data parsing failed AFTER initial check in handleOptionChainClick', { strike, type, price, iv });
        alert('An error occurred retrieving option details.');
    }
}

/** Handles clicks within the strategy table body (remove/toggle buttons) */
function handleStrategyTableClick(event) {
     const removeButton = event.target.closest('button.remove-btn');
     if (removeButton?.dataset.index) {
         const index = parseInt(removeButton.dataset.index, 10);
         if (!isNaN(index)) { removePosition(index); }
         return;
     }

     const toggleButton = event.target.closest('button.toggle-buy-sell');
     if (toggleButton?.dataset.index) {
          const index = parseInt(toggleButton.dataset.index, 10);
         if (!isNaN(index)) { toggleBuySell(index); }
         return;
     }
}

/** Handles input changes within the strategy table body (for lots input) */
function handleStrategyTableChange(event) {
    if (event.target.matches('input[type="number"].lots-input') && event.target.dataset.index) {
        const index = parseInt(event.target.dataset.index, 10);
        if (!isNaN(index)) {
            updateLots(index, event.target.value); // Pass the raw value for validation
        }
    }
}

// ===============================================================
// Strategy Management UI Logic (Corrected)
// ===============================================================

/** Adds a position (called by handleOptionChainClick) */
function addPosition(strike, type, price, iv) {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) { alert("Please select an expiry date first."); return; }

    // Ensure price is non-negative
    const lastPrice = (typeof price === 'number' && !isNaN(price) && price >= 0) ? price : 0;
    // Ensure IV is positive, otherwise null
    const impliedVol = (typeof iv === 'number' && !isNaN(iv) && iv > 0) ? iv : null;
    const dte = calculateDaysToExpiry(expiry);

    if (impliedVol === null) {
        logger.warn(`Adding position ${type} ${strike} @ ${expiry} without valid IV (${iv}). Greeks calculation may fail or be inaccurate for this leg.`);
        // Consider a non-blocking notification instead of alert
        // showNotification(`Warning: IV missing for ${type} ${strike}. Greeks may be inaccurate.`);
    }
     if (dte === null) {
        logger.error(`Could not calculate Days to Expiry for ${expiry}. Cannot add position.`);
        alert(`Error: Invalid expiry date ${expiry} provided.`);
        return; // Critical error, stop
    }

    // TODO: Fetch Lot Size from backend? Or assume a default?
    // For now, we'll add it as null and expect the backend/prepare_strategy_data to handle it.
    const lotSize = null; // Placeholder - Ideally fetch this when asset changes

    const newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type, // 'CE' or 'PE'
        lots: 1,           // Default to 1 lot (BUY)
        tr_type: 'b',      // Default to buy ('b') - This might be redundant if lots dictates it
        last_price: lastPrice,
        iv: impliedVol,    // Store fetched or null IV
        days_to_expiry: dte, // Store calculated DTE
        lot_size: lotSize  // Store lot size (currently null)
    };
    strategyPositions.push(newPosition);
    updateStrategyTable(); // Update UI
    logger.info("Added position:", newPosition);
    // Automatically update chart/calculations after adding a leg?
    // fetchPayoffChart(); // Uncomment if desired
}

/** Helper to calculate days to expiry from YYYY-MM-DD string */
function calculateDaysToExpiry(expiryDateStr) {
    try {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(expiryDateStr)) {
             throw new Error("Invalid date format. Expected YYYY-MM-DD.");
        }
        // Compare dates at UTC midnight
        const expiryDate = new Date(expiryDateStr + 'T00:00:00Z');
        const today = new Date();
        const todayUTC = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()));

        if (isNaN(expiryDate.getTime()) || isNaN(todayUTC.getTime())) {
            throw new Error("Could not parse dates.");
        }
        // Add a small buffer to handle potential floating point issues with exact midnight comparisons
        const diffTime = expiryDate.getTime() - todayUTC.getTime() + 1000; // +1 second buffer

        // Calculate days, rounding up. DTE=0 for today's expiry.
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        return Math.max(0, diffDays); // Return 0 if expiry is past

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
        tableBody.innerHTML = '<tr><td colspan="7" class="placeholder-text">No positions added. Click option prices in the chain to add.</td></tr>';
        return;
    }

    strategyPositions.forEach((pos, index) => {
        // Derive type/class from lots
        const isLong = pos.lots >= 0;
        pos.tr_type = isLong ? 'b' : 's'; // Ensure internal state matches lots
        const positionType = isLong ? "BUY" : "SELL";
        const positionClass = isLong ? "long-position" : "short-position";
        const buttonClass = isLong ? "button-buy" : "button-sell";

        const row = document.createElement("tr");
        row.className = positionClass;
        row.dataset.index = index;

        // Ensure number formatting handles potential non-numeric strikes temporarily
        const formattedStrike = formatNumber(pos.strike_price, pos.strike_price % 1 === 0 ? 0 : 2, 'Invalid');

        row.innerHTML = `
            <td>${pos.option_type || 'N/A'}</td>
            <td>${formattedStrike}</td>
            <td>${pos.expiry_date || 'N/A'}</td>
            <td>
                <input type="number" value="${pos.lots}" data-index="${index}" min="-100" max="100" step="1" class="lots-input number-input-small" aria-label="Lots for position ${index+1}">
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
function updateLots(index, rawValue) {
    if (index < 0 || index >= strategyPositions.length) return;

    // Trim whitespace from raw value
    const trimmedValue = String(rawValue).trim();
    const inputElement = document.querySelector(`${SELECTORS.strategyTableBody} input.lots-input[data-index="${index}"]`);

    // Allow negative sign, but validate the rest is numeric
    if (!/^-?\d+$/.test(trimmedValue) || trimmedValue === '-') {
         logger.warn(`Invalid lots input for index ${index}: "${rawValue}"`);
         if (inputElement) inputElement.value = strategyPositions[index].lots; // Revert UI
         // Optionally show a brief validation message near the input
         return;
    }

    const newLots = parseInt(trimmedValue, 10);

    // Check range after parsing (optional, but good practice)
    const minLots = -100; const maxLots = 100;
    if (newLots < minLots || newLots > maxLots) {
         logger.warn(`Lots input for index ${index} (${newLots}) out of range (${minLots} to ${maxLots}).`);
         alert(`Lots must be between ${minLots} and ${maxLots}.`);
         if (inputElement) inputElement.value = strategyPositions[index].lots; // Revert UI
         return;
    }


    if (newLots === 0) {
        logger.info(`Lots set to 0 for index ${index}, removing position.`);
        removePosition(index); // Remove position if lots become zero
    } else {
        const previousLots = strategyPositions[index].lots;
        strategyPositions[index].lots = newLots;
        strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's'; // Update buy/sell type

        // Update UI elements for the specific row
        const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
        const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);

        if (row && toggleButton) {
             const isNowLong = newLots >= 0;
             const positionType = isNowLong ? "BUY" : "SELL";
             const buttonClass = isNowLong ? "button-buy" : "button-sell";
             row.className = isNowLong ? "long-position" : "short-position";
             toggleButton.textContent = positionType;
             toggleButton.className = `toggle-buy-sell ${buttonClass}`; // Replace classes
        } else {
            logger.warn(`Could not find row/button elements for index ${index} during lot update, doing full table refresh.`);
             updateStrategyTable(); // Fallback
        }
        logger.info(`Updated lots for index ${index} from ${previousLots} to ${newLots}`);
        // Note: Calculation update requires clicking "Update" button
    }
}

/** Toggles a position between Buy and Sell */
function toggleBuySell(index) {
    if (index < 0 || index >= strategyPositions.length) return;

    const previousLots = strategyPositions[index].lots;
    let newLots = -previousLots; // Flip the sign

    // If flipping results in 0, default to Buy 1
     if (newLots === 0) { newLots = 1; }

    strategyPositions[index].lots = newLots;
    strategyPositions[index].tr_type = newLots >= 0 ? 'b' : 's'; // Update type

    logger.info(`Toggled Buy/Sell for index ${index}. Prev lots: ${previousLots}, New lots: ${newLots}`);

    // Update the specific row UI
    const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
    const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);
    const lotsInput = row?.querySelector(`input.lots-input[data-index="${index}"]`);

    if (row && toggleButton && lotsInput) {
        const isLong = newLots >= 0;
        const positionType = isLong ? "BUY" : "SELL";
        const buttonClass = isLong ? "button-buy" : "button-sell";
        row.className = isLong ? "long-position" : "short-position";
        toggleButton.textContent = positionType;
        toggleButton.className = `toggle-buy-sell ${buttonClass}`;
        lotsInput.value = newLots; // Update number input
    } else {
        logger.warn(`Could not find row/button/input elements for index ${index} during toggle, doing full table refresh.`);
        updateStrategyTable(); // Fallback
    }
    // Note: Calculation update requires clicking "Update" button
}

/** Removes a position from the strategy */
function removePosition(index) {
    if (index < 0 || index >= strategyPositions.length) return;
    const removedPos = strategyPositions.splice(index, 1);
    logger.info("Removed position at index", index, removedPos[0]);

    // Re-render the table - indices of subsequent items change
    updateStrategyTable();

    // Update calculations if the strategy is not empty AND results were previously shown
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    // Check if chart has actual Plotly content (might need refinement)
    const hasChartContent = chartContainer?.querySelector('.plotly') !== null;

    if (strategyPositions.length > 0 && hasChartContent) {
        logger.info("Remaining positions exist, updating calculations...");
        fetchPayoffChart(); // Update results
    } else if (strategyPositions.length === 0) {
        logger.info("Strategy is now empty, resetting calculation outputs.");
        resetCalculationOutputsUI(); // Clear only outputs if strategy becomes empty
    }
}

/** Clears all positions and resets calculation outputs */
function clearAllPositions() {
    if (strategyPositions.length === 0) return;
    if (confirm("Are you sure you want to clear all strategy legs?")) {
        logger.info("Clearing all positions...");
        strategyPositions = [];
        updateStrategyTable(); // Clear table UI
        resetCalculationOutputsUI(); // Reset chart, metrics, tax, greeks etc.
        logger.info("Strategy cleared.");
    }
}

/** Resets ONLY the calculation output areas (chart, metrics, tax, greeks, warnings) */
function resetCalculationOutputsUI() {
    const logger = window.logger || console; // Ensure logger is accessible
    logger.debug("Resetting calculation output UI elements...");

    // Reset Payoff Chart
    const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
    if (chartContainer) {
        try { if (typeof Plotly !== 'undefined' && chartContainer.id) Plotly.purge(chartContainer.id); } catch (e) { /* ignore */ }
        chartContainer.innerHTML = '<div class="placeholder-text">Add legs and click "Update Strategy"</div>';
        setElementState(chartContainer, 'content');
    } else { logger.warn("Payoff chart container not found during output reset."); }

    // Reset Tax Container
    const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
    if (taxContainer) {
        taxContainer.innerHTML = '<p class="placeholder-text">Update strategy to calculate charges.</p>';
        setElementState(taxContainer, 'content');
    } else { logger.warn("Tax info container not found during output reset."); }

    // Reset Greeks Table & Section
    const greeksTable = document.querySelector(SELECTORS.greeksTable);
    const greeksSection = document.querySelector(SELECTORS.greeksTableContainer);
    if (greeksTable) {
        const caption = greeksTable.querySelector('caption'); if (caption) caption.textContent = 'Portfolio Option Greeks';
        const greekBody = greeksTable.querySelector('tbody'); if (greekBody) greekBody.innerHTML = `<tr><td colspan="9" class="placeholder-text">Update strategy to calculate Greeks.</td></tr>`;
        const greekFoot = greeksTable.querySelector('tfoot'); if (greekFoot) greekFoot.innerHTML = "";
        setElementState(greeksTable, 'content');
    } else { logger.warn("Greeks table not found during output reset."); }
    if (greeksSection) { setElementState(greeksSection, 'content'); } else { logger.warn("Greeks section not found during output reset."); }


    // Reset Greeks Analysis Section
    const greeksAnalysisSection = document.querySelector(SELECTORS.greeksAnalysisSection);
    const greeksAnalysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
    if (greeksAnalysisSection) { setElementState(greeksAnalysisSection, 'hidden'); } else { logger.warn("Greeks Analysis section not found during output reset."); }
    if (greeksAnalysisContainer) { greeksAnalysisContainer.innerHTML = ''; setElementState(greeksAnalysisContainer, 'content'); } else { logger.warn("Greeks Analysis result container not found during output reset."); }

   // Reset Metrics Display using displayMetric helper
   displayMetric('N/A', SELECTORS.maxProfitDisplay);
   displayMetric('N/A', SELECTORS.maxLossDisplay);
   displayMetric('N/A', SELECTORS.breakevenDisplay);
   displayMetric('N/A', SELECTORS.rewardToRiskDisplay);
   displayMetric('N/A', SELECTORS.netPremiumDisplay, '', '', 2, true); // Format as currency
   setElementState(SELECTORS.metricsList, 'content');


    // Reset Cost Breakdown
    const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
    const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
    if (breakdownList) { breakdownList.innerHTML = ""; setElementState(breakdownList, 'content'); } else { logger.warn("Cost breakdown list not found during output reset."); }
    if (breakdownContainer) { breakdownContainer.open = false; setElementState(breakdownContainer, 'hidden'); breakdownContainer.style.display = 'none'; } else { logger.warn("Cost breakdown container not found during output reset."); }


    // Reset Warning Container
     const warningContainer = document.querySelector(SELECTORS.warningContainer);
     if (warningContainer) {
         warningContainer.textContent = '';
         setElementState(warningContainer, 'hidden'); // Hide it
     } else { logger.warn("Warning container not found during output reset."); }

    logger.debug("Calculation output UI elements reset complete.");
}


/** Resets the entire results area AND the strategy input table */
function resetResultsUI() {
    const logger = window.logger || console; // Ensure logger is accessible
    logger.info("Resetting results UI elements AND clearing strategy input...");

    // --- Clear Strategy Data and Table ---
    // Ensure strategyPositions is defined (it should be in global state)
    if (typeof strategyPositions !== 'undefined' && Array.isArray(strategyPositions)) {
        strategyPositions = []; // Clear the data array
        logger.debug("Strategy positions data array cleared.");
        // Ensure updateStrategyTable is defined before calling
        if (typeof updateStrategyTable === 'function') {
            updateStrategyTable(); // Update the #strategyTable tbody
            logger.debug("Strategy input table UI updated.");
        } else {
             logger.warn("updateStrategyTable function not found at time of resetResultsUI call - cannot clear strategy table UI visually.");
             // Manual clear as fallback
             const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
             if (strategyTableBody) { strategyTableBody.innerHTML = `<tr><td colspan="7" class="placeholder-text">No positions added. Click option prices in the chain to add.</td></tr>`; }
        }
    } else { logger.warn("Cannot clear strategy positions: 'strategyPositions' array not found or not an array."); }

    // --- Call the specific output reset function ---
    // Ensure resetCalculationOutputsUI is defined
    if (typeof resetCalculationOutputsUI === 'function') {
        resetCalculationOutputsUI();
    } else {
        logger.error("resetCalculationOutputsUI function not defined when resetResultsUI was called.");
        // Add fallback logic here if needed, or ensure ordering is correct.
    }


    // --- Reset News and Stock Analysis Placeholders (If desired on full reset) ---
    // This is usually done during asset change, but uncomment if you want clearAllPositions to also clear these
    /*
    const newsContainer = document.querySelector(SELECTORS.newsResultContainer);
    if (newsContainer) { newsContainer.innerHTML = '<p class="placeholder-text">Select an asset to load news...</p>'; setElementState(newsContainer, 'content'); }
    const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
    if (analysisContainer) { analysisContainer.innerHTML = '<p class="placeholder-text">Select an asset to load analysis...</p>'; setElementState(analysisContainer, 'content'); }
    */

    // --- Reset Status Message ---
     const messageContainer = document.querySelector(SELECTORS.statusMessageContainer);
     if (messageContainer) { messageContainer.textContent = ''; setElementState(messageContainer, 'hidden'); }

     logger.info("Full UI reset (Strategy input AND Calculation Outputs) complete.");
}
// ===============================================================
// Payoff Chart & Results Logic (Corrected)
// ===============================================================

/** Fetches payoff chart data, metrics, taxes, greeks and triggers rendering */
async function fetchPayoffChart() {
    const logger = window.logger || console; // Ensure logger is accessible
    const updateButton = document.querySelector(SELECTORS.updateChartButton);

    logger.info("--- [fetchPayoffChart] START ---");

    // 1. Get Asset and Strategy Legs
    const asset = activeAsset;
    if (!asset) {
        alert("Please select an asset first.");
        logger.error("[fetchPayoffChart] Aborted: No active asset.");
        return;
    }

    logger.debug("[fetchPayoffChart] Gathering strategy legs...");
    // Ensure gatherStrategyLegsFromTable is defined
    if (typeof gatherStrategyLegsFromTable !== 'function') {
        logger.error("gatherStrategyLegsFromTable function is not defined!");
        alert("Internal Error: Cannot gather strategy data.");
        return;
    }
    const currentStrategyLegs = gatherStrategyLegsFromTable(); // Use the VALIDATED gather function

    if (!currentStrategyLegs || currentStrategyLegs.length === 0) {
        logger.warn("[fetchPayoffChart] Aborted: No valid strategy legs found.");
        // Ensure resetCalculationOutputsUI is defined
        if (typeof resetCalculationOutputsUI === 'function') {
            resetCalculationOutputsUI();
        } else {
            logger.error("resetCalculationOutputsUI function not defined!");
        }
        alert("Please add strategy legs from the option chain first.");
        return;
    }
    logger.debug(`[fetchPayoffChart] Found ${currentStrategyLegs.length} valid legs.`);

    // 2. Reset OUTPUT UI & Set Loading States
    logger.debug("[fetchPayoffChart] Resetting output UI sections...");
    // Ensure resetCalculationOutputsUI is defined
     if (typeof resetCalculationOutputsUI === 'function') {
        resetCalculationOutputsUI();
    } else {
        logger.error("resetCalculationOutputsUI function not defined!");
        // Add fallback UI clearing if necessary
    }
    logger.debug("[fetchPayoffChart] Setting loading states...");
    // Ensure setElementState is defined
    if (typeof setElementState !== 'function') {
        logger.error("setElementState function not defined!");
        return; // Cannot proceed without setting states
    }
    setElementState(SELECTORS.payoffChartContainer, 'loading', 'Calculating...');
    setElementState(SELECTORS.taxInfoContainer, 'loading');
    setElementState(SELECTORS.greeksTableContainer, 'loading');
    setElementState(SELECTORS.greeksTable, 'loading');
    setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
    setElementState(SELECTORS.metricsList, 'loading');
    // Ensure displayMetric is defined
    if (typeof displayMetric === 'function') {
        displayMetric('...', SELECTORS.maxProfitDisplay);
        displayMetric('...', SELECTORS.maxLossDisplay);
        displayMetric('...', SELECTORS.breakevenDisplay);
        displayMetric('...', SELECTORS.rewardToRiskDisplay);
        displayMetric('...', SELECTORS.netPremiumDisplay);
    } else { logger.error("displayMetric function not defined!"); }
    setElementState(SELECTORS.costBreakdownContainer, 'hidden');
    setElementState(SELECTORS.warningContainer, 'hidden');

    if (updateButton) updateButton.disabled = true;

    // 3. Prepare Request Data
    const requestData = { asset: asset, strategy: currentStrategyLegs };
    logger.debug("[fetchPayoffChart] Final requestData for backend:", JSON.parse(JSON.stringify(requestData)));

    // 4. Fetch API and Handle Response
    try {
        // Ensure fetchAPI is defined
        if (typeof fetchAPI !== 'function') throw new Error("fetchAPI function not defined!");

        const data = await fetchAPI('/get_payoff_chart', {
            method: 'POST', body: JSON.stringify(requestData)
        });
        logger.debug("[fetchPayoffChart] Received response data:", JSON.parse(JSON.stringify(data)));

        // 5. Validate Backend Response
        if (!data || typeof data !== 'object') {
            throw new Error("Invalid response format received from server.");
        }
        if (!data.success) {
             const errorMessage = data.message || "Calculation failed on the server. Check input legs.";
             logger.error(`[fetchPayoffChart] Backend reported failure: ${errorMessage}`, data);
             throw new Error(errorMessage);
        }
        logger.debug("[fetchPayoffChart] Backend response indicates success=true.");

        // 6. Render Results
        logger.debug("[fetchPayoffChart] Rendering results...");

        // Render Metrics & Warnings (ensure displayMetric is defined)
        const metricsContainerData = data.metrics;
         if (typeof displayMetric === 'function' && metricsContainerData && metricsContainerData.metrics) {
            const metrics = metricsContainerData.metrics;
            displayMetric(metrics.max_profit, SELECTORS.maxProfitDisplay);
            displayMetric(metrics.max_loss, SELECTORS.maxLossDisplay);
            const bePoints = Array.isArray(metrics.breakeven_points) ? metrics.breakeven_points.join(' / ') : metrics.breakeven_points;
            displayMetric(bePoints || 'N/A', SELECTORS.breakevenDisplay);
            displayMetric(metrics.reward_to_risk_ratio, SELECTORS.rewardToRiskDisplay);
            displayMetric(metrics.net_premium, SELECTORS.netPremiumDisplay, '', '', 2, true);
            setElementState(SELECTORS.metricsList, 'content');

            const warnings = metrics.warnings;
            const warningElement = document.querySelector(SELECTORS.warningContainer);
            if (warningElement && Array.isArray(warnings) && warnings.length > 0) {
                warningElement.innerHTML = `<strong>Calculation Warnings:</strong><ul>${warnings.map(w => `<li>${w}</li>`).join('')}</ul>`;
                setElementState(warningElement, 'content'); warningElement.style.display = '';
            } else if (warningElement) { setElementState(warningElement, 'hidden'); }
         } else { /* Handle missing metrics - error state set below in catch or here */
            logger.error("[fetchPayoffChart] Metrics data missing or displayMetric not defined.");
            setElementState(SELECTORS.metricsList, 'error');
            // Set individual metrics to Error
            if(typeof displayMetric === 'function') {
                displayMetric('Error', SELECTORS.maxProfitDisplay); /* ... and others ... */
            }
         }

        // Render Cost Breakdown (ensure renderCostBreakdown defined)
        const costBreakdownData = metricsContainerData?.cost_breakdown_per_leg;
        const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
        const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
        if (typeof renderCostBreakdown === 'function' && breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) {
            renderCostBreakdown(breakdownList, costBreakdownData);
            setElementState(breakdownContainer, 'content'); breakdownContainer.style.display = ''; breakdownContainer.open = false;
        } else if (breakdownContainer) { setElementState(breakdownContainer, 'hidden'); }
        else { logger.error("renderCostBreakdown function not defined!"); }

        // Render Tax Table (ensure renderTaxTable defined)
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
        if (typeof renderTaxTable === 'function' && taxContainer) {
            if (data.charges) { renderTaxTable(taxContainer, data.charges); setElementState(taxContainer, 'content'); }
            else { taxContainer.innerHTML = "<p class='placeholder-text'>Charge data unavailable.</p>"; setElementState(taxContainer, 'content'); }
        } else { logger.error("renderTaxTable function not defined!"); }


        // Render Chart (ensure renderPayoffChart defined)
        const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
        const chartDataKey = "chart_figure_json";
        if (typeof renderPayoffChart === 'function' && chartContainer) {
            if (data[chartDataKey]) { renderPayoffChart(chartContainer, data[chartDataKey]); }
            else { setElementState(chartContainer, 'error', 'Chart could not be generated.'); }
        } else { logger.error("renderPayoffChart function not defined!"); }


        // Render Greeks Table & Trigger Analysis (ensure renderGreeksTable & fetchAndDisplayGreeksAnalysis defined)
        const greeksTableElement = document.querySelector(SELECTORS.greeksTable);
        const greeksSectionElement = document.querySelector(SELECTORS.greeksTableContainer);
        if (typeof renderGreeksTable === 'function' && typeof fetchAndDisplayGreeksAnalysis === 'function' && greeksTableElement && greeksSectionElement) {
             if (data.greeks && Array.isArray(data.greeks)) {
                const calculatedTotals = renderGreeksTable(greeksTableElement, data.greeks);
                setElementState(greeksSectionElement, 'content');
                if (calculatedTotals) {
                    const hasMeaningfulGreeks = Object.values(calculatedTotals).some(v => v !== null && Math.abs(v) > 1e-9);
                    if (hasMeaningfulGreeks) { fetchAndDisplayGreeksAnalysis(asset, calculatedTotals); }
                    else { /* Show placeholder for no greeks */
                        const analysisSection = document.querySelector(SELECTORS.greeksAnalysisSection); const analysisContainer = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
                        if (analysisSection && analysisContainer) { analysisContainer.innerHTML = '<p class="placeholder-text">No net option exposure to analyze Greeks.</p>'; setElementState(analysisSection, 'content'); setElementState(analysisContainer, 'content'); }
                    }
                } else { /* Handle null totals */ const greeksAS = document.querySelector(SELECTORS.greeksAnalysisSection); if(greeksAS) setElementState(greeksAS, 'hidden'); }
             } else { /* handle missing greeks data */ greeksSectionElement.innerHTML = '<h3 class="section-subheader">Options Greeks</h3><p class="placeholder-text">Greeks data not available.</p>'; setElementState(greeksSectionElement, 'content'); const greeksAS = document.querySelector(SELECTORS.greeksAnalysisSection); if(greeksAS) setElementState(greeksAS, 'hidden'); }
        } else { logger.error("renderGreeksTable or fetchAndDisplayGreeksAnalysis function not defined!"); }

        logger.info("[fetchPayoffChart] Successfully processed and rendered results.");

    } catch (error) { // Catch errors from fetchAPI or rendering logic
        logger.error(`[fetchPayoffChart] Error during calculation/rendering: ${error.message}`, error);
        // Ensure resetCalculationOutputsUI is defined
        if (typeof resetCalculationOutputsUI === 'function') resetCalculationOutputsUI();
        let errorMsg = `Calculation Error: ${error.message || 'Failed to process results.'}`;
        // Ensure setElementState is defined
        if (typeof setElementState === 'function') {
            setElementState(SELECTORS.payoffChartContainer, 'error', errorMsg);
            setElementState(SELECTORS.taxInfoContainer, 'error', 'Error');
            setElementState(SELECTORS.greeksTableContainer, 'error', 'Error');
            setElementState(SELECTORS.greeksTable, 'error');
            setElementState(SELECTORS.metricsList, 'error');
            // Ensure displayMetric defined before setting error states
            if (typeof displayMetric === 'function') { /* Set metrics spans to Error */ }
            setElementState(SELECTORS.greeksAnalysisSection, 'hidden');
            setElementState(SELECTORS.costBreakdownContainer, 'hidden');
            setElementState(SELECTORS.warningContainer, 'hidden');
            setElementState(SELECTORS.globalErrorDisplay, 'error', errorMsg); // Show global error
        } else {
             alert("Critical Error: Cannot update UI state.");
        }

    } finally {
        if (updateButton) updateButton.disabled = false; // Always re-enable button
        logger.info("--- [fetchPayoffChart] END ---");
    }
}

/**
 * Gathers VALID strategy leg data from the global strategyPositions array.
 * Formats the data specifically for the '/get_payoff_chart' backend endpoint.
 */
 function gatherStrategyLegsFromTable() { // Name is legacy, now reads state array
      const logger = window.logger || console;
      logger.debug("--- [gatherStrategyLegs] START ---");

      if (!Array.isArray(strategyPositions)) {
          logger.error("[gatherStrategyLegs] Aborted: strategyPositions is not an array or not defined.");
          return [];
      }
      if (strategyPositions.length === 0) {
           logger.warn("[gatherStrategyLegs] Aborted: strategyPositions array is empty.");
           return [];
      }

      logger.debug("[gatherStrategyLegs] Source strategyPositions:", JSON.parse(JSON.stringify(strategyPositions)));

      const validLegs = [];
      let invalidLegCount = 0;

      strategyPositions.forEach((pos, index) => {
           logger.debug(`[gatherStrategyLegs] Processing index ${index}, raw pos object:`, JSON.parse(JSON.stringify(pos)));

           let legIsValid = true;
           let validationError = null;

           // --- Detailed Validation ---
           if (!pos || typeof pos !== 'object') { validationError = "Position data is not an object."; legIsValid = false; }
           else {
                // Option Type ('CE' or 'PE')
                if (!pos.option_type || (pos.option_type !== 'CE' && pos.option_type !== 'PE')) { validationError = `Invalid option_type: ${pos.option_type}`; legIsValid = false; }
                // Strike Price (Positive Number)
                else if (pos.strike_price === undefined || pos.strike_price === null || isNaN(parseFloat(pos.strike_price)) || parseFloat(pos.strike_price) <= 0) { validationError = `Invalid strike_price: ${pos.strike_price}`; legIsValid = false; }
                // Expiry Date (YYYY-MM-DD Format)
                else if (!pos.expiry_date || !/^\d{4}-\d{2}-\d{2}$/.test(pos.expiry_date)) { validationError = `Invalid expiry_date: ${pos.expiry_date}`; legIsValid = false; }
                // Lots (Non-zero Integer) - Using Number.isInteger for stricter check
                else if (pos.lots === undefined || pos.lots === null || !Number.isInteger(pos.lots) || pos.lots === 0) { validationError = `Invalid lots: ${pos.lots}`; legIsValid = false; }
                // Last Price (Non-negative Number)
                else if (pos.last_price === undefined || pos.last_price === null || isNaN(parseFloat(pos.last_price)) || parseFloat(pos.last_price) < 0) { validationError = `Invalid last_price: ${pos.last_price}`; legIsValid = false; }
                // Days To Expiry (Non-negative Integer - should be pre-calculated) - Stricter check
                else if (pos.days_to_expiry === undefined || pos.days_to_expiry === null || !Number.isInteger(pos.days_to_expiry) || pos.days_to_expiry < 0) { validationError = `Invalid/missing days_to_expiry: ${pos.days_to_expiry}`; legIsValid = false; }
                // Implied Volatility (IV - Can be null, but if present, must be non-negative number)
                else if (pos.iv !== null && pos.iv !== undefined && (isNaN(parseFloat(pos.iv)) || parseFloat(pos.iv) < 0)) { validationError = `Invalid iv: ${pos.iv}`; legIsValid = false; }
                // Lot Size (Optional - Can be null, but if present, must be positive integer) - Stricter check
                else if (pos.lot_size !== null && pos.lot_size !== undefined && (!Number.isInteger(pos.lot_size) || pos.lot_size <= 0)) { validationError = `Invalid lot_size: ${pos.lot_size}`; legIsValid = false; }
           }

           // --- If Valid, Format for Backend ---
           if (legIsValid) {
                const lotsInt = parseInt(pos.lots); // Already validated as integer
                const formattedLeg = {
                     // Keys and types MUST match Python Pydantic model 'StrategyLegInput'
                     op_type: pos.option_type === 'CE' ? 'c' : 'p', // 'c' or 'p'
                     strike: String(pos.strike_price),             // String
                     tr_type: lotsInt >= 0 ? 'b' : 's',            // 'b' or 's'
                     op_pr: String(pos.last_price),                // String
                     lot: String(Math.abs(lotsInt)),               // String (absolute value)
                     // Send lot_size as string if present and valid, otherwise null
                     lot_size: (pos.lot_size && Number.isInteger(pos.lot_size) && pos.lot_size > 0) ? String(pos.lot_size) : null, // String or null
                     // Send IV if it's a valid number, otherwise null
                     iv: (pos.iv !== null && pos.iv !== undefined && !isNaN(parseFloat(pos.iv))) ? parseFloat(pos.iv) : null, // Number or null
                     days_to_expiry: pos.days_to_expiry,           // Number (already validated as integer)
                     expiry_date: pos.expiry_date,                 // String (YYYY-MM-DD)
                };
                logger.debug(`[gatherStrategyLegs] Pushing valid formatted leg ${index}:`, formattedLeg);
                validLegs.push(formattedLeg);
           } else {
                logger.error(`[gatherStrategyLegs] Skipping invalid position data at index ${index}. Reason: ${validationError || 'Unknown validation failure'}. Raw Data:`, JSON.parse(JSON.stringify(pos)));
                invalidLegCount++;
           }
      }); // End forEach

      // --- User Feedback on Invalid Legs ---
      if (invalidLegCount > 0 && validLegs.length === 0) {
           alert(`Error: ${invalidLegCount} invalid strategy leg(s) found and NO valid legs remaining. Cannot calculate. Please check console logs for details and correct the strategy.`);
      } else if (invalidLegCount > 0) {
           alert(`Warning: ${invalidLegCount} invalid strategy leg(s) were ignored during calculation. Results are based on the remaining ${validLegs.length} valid leg(s). Check console logs for details.`);
      }


      logger.debug(`[gatherStrategyLegs] Returning ${validLegs.length} valid formatted legs (ignored ${invalidLegCount}).`);
      logger.debug("--- [gatherStrategyLegs] END ---");
      return validLegs;
 }


// --- Rendering Helpers for Payoff Results ---

function renderTaxTable(containerElement, taxData) {
    // Assume logger, formatCurrency, formatNumber are defined globally or imported
    const logger = window.logger || console; // Use console if no specific logger

    if (!taxData || !taxData.charges_summary || !taxData.breakdown_per_leg || !Array.isArray(taxData.breakdown_per_leg)) {
        containerElement.innerHTML = '<p class="error-message">Charge calculation data is incomplete or unavailable.</p>';
        logger.warn("renderTaxTable called with invalid or incomplete taxData:", taxData);
        setElementState(containerElement, 'content'); // Show container with error message
        return;
    }

    containerElement.innerHTML = ""; // Clear previous content

    const details = document.createElement('details');
    details.className = "results-details tax-details";
    details.open = false; // Default closed

    const summary = document.createElement('summary');
    // Use formatCurrency for the total in the summary
    summary.innerHTML = `<strong>Estimated Charges Breakdown (Total: ${formatCurrency(taxData.total_estimated_cost, 2)})</strong>`;
    details.appendChild(summary);

    const tableWrapper = document.createElement('div');
    tableWrapper.className = 'table-wrapper thin-scrollbar'; // Add scrollbar class
    details.appendChild(tableWrapper);

    const table = document.createElement("table");
    table.className = "results-table charges-table data-table";
    const charges = taxData.charges_summary || {};
    const breakdown = taxData.breakdown_per_leg;

    // Generate Table Body with Mapped Values
    const tableBody = breakdown.map(t => {
        // Safely map Transaction Type ('B'/'S' to 'BUY'/'SELL')
        const actionDisplay = (t.transaction_type || '').toUpperCase() === 'B' ? 'BUY' : (t.transaction_type || '').toUpperCase() === 'S' ? 'SELL' : '?';
        // Safely map Option Type ('CE'/'PE')
        const typeDisplay = (t.option_type || '').toUpperCase(); // Backend sends CE/PE directly now

        // Use nullish coalescing (??) for safer defaults in formatting
        return `
        <tr>
            <td>${actionDisplay}</td>
            <td>${typeDisplay}</td>
            <td>${formatNumber(t.strike, 2, '-')}</td>
            <td>${formatNumber(t.lots, 0, '-')}</td>
            <td>${formatNumber(t.premium_per_share, 2, '-')}</td>
            <td>${formatNumber(t.stt ?? 0, 2)}</td>
            <td>${formatNumber(t.stamp_duty ?? 0, 2)}</td>
            <td>${formatNumber(t.sebi_fee ?? 0, 4)}</td>
            <td>${formatNumber(t.txn_charge ?? 0, 4)}</td>
            <td>${formatNumber(t.brokerage ?? 0, 2)}</td>
            <td>${formatNumber(t.gst ?? 0, 2)}</td>
            <td class="note" title="${t.stt_note || ''}">${((t.stt_note || '').substring(0, 15))}${ (t.stt_note || '').length > 15 ? '...' : ''}</td>
        </tr>`;
    }).join('');

    // Prepare Footer Totals (using nullish coalescing)
    const total_stt = charges.stt ?? 0;
    const total_stamp = charges.stamp_duty ?? 0;
    const total_sebi = charges.sebi_fee ?? 0;
    const total_txn = charges.txn_charges ?? 0; // Key from backend
    const total_brokerage = charges.brokerage ?? 0;
    const total_gst = charges.gst ?? 0;
    const overall_total = taxData.total_estimated_cost ?? 0;

    // Assemble Table HTML
    table.innerHTML = `
        <thead>
            <tr>
                <th>Act</th><th>Type</th><th>Strike</th><th>Lots</th><th>Premium</th>
                <th>STT</th><th>Stamp</th><th>SEBI</th><th>Txn</th><th>Broker</th><th>GST</th>
                <th title="Securities Transaction Tax Note">STT Note</th>
            </tr>
        </thead>
        <tbody>${tableBody}</tbody>
        <tfoot>
            <tr class="totals-row">
                <td colspan="5">Total Estimated Charges</td>
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


/** Renders the Greeks table and calculates/returns portfolio totals */
function renderGreeksTable(tableElement, greeksList) {
    const logger = window.logger || console;
    tableElement.innerHTML = ''; // Clear previous

    if (!tableElement || !(tableElement instanceof HTMLTableElement)) {
        logger.error("renderGreeksTable: Invalid tableElement provided.");
        return null;
    }

    const caption = tableElement.createCaption();
    caption.className = "table-caption";

    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    let hasCalculatedGreeks = false;
    let skippedLegsCount = 0;
    let processedLegsCount = 0;

    if (!Array.isArray(greeksList)) {
        caption.textContent = "Error: Invalid Greeks data received.";
        setElementState(tableElement, 'error');
        return null;
    }

    const totalLegsInput = greeksList.length;
    if (totalLegsInput === 0) {
        caption.textContent = "Portfolio Option Greeks (No legs)";
        const tbody = tableElement.createTBody();
        tbody.innerHTML = `<tr><td colspan="9" class="placeholder-text">No option legs in the strategy.</td></tr>`;
        setElementState(tableElement, 'content');
        return totals; // Return zeroed totals
    }

    // Create Header
    const thead = tableElement.createTHead();
    thead.innerHTML = `
        <tr>
            <th>Action</th><th>Lots</th><th>Type</th><th>Strike</th>
            <th title="Delta per Share">Δ Delta</th><th title="Gamma per Share">Γ Gamma</th>
            <th title="Theta per Share (per Day)">Θ Theta</th><th title="Vega per Share (per 1% IV)">Vega</th>
            <th title="Rho per Share (per 1% Rate)">Ρ Rho</th>
        </tr>`;

    // Populate Body
    const tbody = tableElement.createTBody();
    greeksList.forEach((g, index) => {
        const row = tbody.insertRow();
        const inputData = g?.input_data;
        // Use PER SHARE greeks for row display
        const gv_per_share = g?.calculated_greeks_per_share;
        // Use PER LOT greeks for easier totals calculation if available, fallback to per_share * lot_size
        const gv_per_lot = g?.calculated_greeks_per_lot;

        if (!inputData || !gv_per_share) {
             logger.warn(`renderGreeksTable: Malformed data for leg index ${index}. Skipping.`);
             skippedLegsCount++;
             row.innerHTML = `<td colspan="9" class="skipped-leg">Leg ${index + 1}: Invalid data</td>`;
             return;
        }

        // Extract details
        const actionDisplay = (inputData.tr_type === 'b') ? 'BUY' : (inputData.tr_type === 's' ? 'SELL' : '?');
        const typeDisplay = (inputData.op_type === 'c') ? 'CE' : (inputData.op_type === 'p' ? 'PE' : '?');
        const lots = parseInt(inputData.lots || '0', 10); // Read 'lots' key for quantity multiplier
        const lotSize = parseInt(inputData.lot_size || '0', 10); // Read 'lot_size' for display/fallback calc
        const strike = inputData.strike;
        const lotsDisplay = (lots !== 0) ? `${lots}` : 'N/A'; // Use state 'lots'

        // Fill Cells (Displaying Per-Share Greeks)
        row.insertCell().textContent = actionDisplay;
        row.insertCell().textContent = lotsDisplay; // Display number of lots
        row.insertCell().textContent = typeDisplay;
        row.insertCell().textContent = formatNumber(strike, 2);
        row.insertCell().textContent = formatNumber(gv_per_share.delta, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.gamma, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.theta, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.vega, 4, '-');
        row.insertCell().textContent = formatNumber(gv_per_share.rho, 4, '-');

        // Accumulate PORTFOLIO Totals
        // Use per-lot greeks if available and valid, otherwise calculate from per-share
        let legDelta = 0, legGamma = 0, legTheta = 0, legVega = 0, legRho = 0;
        let isValidForTotal = false;

        if (gv_per_lot && lots !== 0) { // Use per-lot if available
            if (typeof gv_per_lot.delta === 'number' && isFinite(gv_per_lot.delta)) { legDelta = gv_per_lot.delta * lots; isValidForTotal = true; }
            if (typeof gv_per_lot.gamma === 'number' && isFinite(gv_per_lot.gamma)) { legGamma = gv_per_lot.gamma * lots; }
            if (typeof gv_per_lot.theta === 'number' && isFinite(gv_per_lot.theta)) { legTheta = gv_per_lot.theta * lots; }
            if (typeof gv_per_lot.vega === 'number' && isFinite(gv_per_lot.vega)) { legVega = gv_per_lot.vega * lots; }
            if (typeof gv_per_lot.rho === 'number' && isFinite(gv_per_lot.rho)) { legRho = gv_per_lot.rho * lots; }
            if (!isValidForTotal) logger.warn(`renderGreeksTable: Leg ${index + 1} had per-lot data but delta was invalid. Totals might be inaccurate.`);
        } else if (gv_per_share && lots !== 0 && lotSize > 0) { // Fallback to per-share * quantity
            const quantity = lots * lotSize;
             if (typeof gv_per_share.delta === 'number' && isFinite(gv_per_share.delta)) { legDelta = gv_per_share.delta * quantity; isValidForTotal = true; }
             if (typeof gv_per_share.gamma === 'number' && isFinite(gv_per_share.gamma)) { legGamma = gv_per_share.gamma * quantity; }
             if (typeof gv_per_share.theta === 'number' && isFinite(gv_per_share.theta)) { legTheta = gv_per_share.theta * quantity; }
             if (typeof gv_per_share.vega === 'number' && isFinite(gv_per_share.vega)) { legVega = gv_per_share.vega * quantity; }
             if (typeof gv_per_share.rho === 'number' && isFinite(gv_per_share.rho)) { legRho = gv_per_share.rho * quantity; }
             if (!isValidForTotal) logger.warn(`renderGreeksTable: Leg ${index + 1} used per-share data but delta was invalid. Totals might be inaccurate.`);
        }


        if (isValidForTotal) {
            totals.delta += legDelta;
            totals.gamma += legGamma;
            totals.theta += legTheta;
            totals.vega += legVega;
            totals.rho += legRho;
            hasCalculatedGreeks = true;
            processedLegsCount++;
            row.classList.add('greeks-calculated');
        } else {
            logger.warn(`renderGreeksTable: Skipping leg index ${index} from total calculation due to invalid data.`);
            skippedLegsCount++;
            row.classList.add('greeks-skipped');
            row.style.opacity = '0.6';
            row.style.fontStyle = 'italic';
        }
    }); // End forEach

    caption.textContent = `Portfolio Option Greeks (${processedLegsCount} Processed, ${skippedLegsCount} Skipped)`;

    // Create Footer with Totals
    const tfoot = tableElement.createTFoot();
    const footerRow = tfoot.insertRow();
    footerRow.className = 'totals-row';

    if (hasCalculatedGreeks) {
        const headerCell = footerRow.insertCell();
        headerCell.colSpan = 4;
        headerCell.textContent = 'Total Portfolio Exposure';
        headerCell.style.textAlign = 'right'; headerCell.style.fontWeight = 'bold';
        footerRow.insertCell().textContent = formatNumber(totals.delta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.gamma, 4);
        footerRow.insertCell().textContent = formatNumber(totals.theta, 4);
        footerRow.insertCell().textContent = formatNumber(totals.vega, 4);
        footerRow.insertCell().textContent = formatNumber(totals.rho, 4);
        setElementState(tableElement, 'content');
    } else if (totalLegsInput > 0) {
        const cell = footerRow.insertCell(); cell.colSpan = 9;
        cell.textContent = 'Could not calculate portfolio totals (invalid leg data).';
        cell.style.textAlign = 'center'; cell.style.fontStyle = 'italic';
        setElementState(tableElement, 'content');
    }
    // If totalLegsInput was 0, state handled earlier

    // Return rounded totals for analysis
    const finalTotals = {
        delta: roundToPrecision(totals.delta, 4),
        gamma: roundToPrecision(totals.gamma, 4),
        theta: roundToPrecision(totals.theta, 4),
        vega: roundToPrecision(totals.vega, 4),
        rho: roundToPrecision(totals.rho, 4)
    };

    logger.info(`renderGreeksTable: Rendered ${processedLegsCount} valid legs. Totals: ${JSON.stringify(finalTotals)}`);
    return finalTotals;
}


/** Fetches and displays the LLM analysis of portfolio Greeks */
async function fetchAndDisplayGreeksAnalysis(asset, portfolioGreeksData) {
    const container = document.querySelector(SELECTORS.greeksAnalysisResultContainer);
    const section = document.querySelector(SELECTORS.greeksAnalysisSection);

    if (!container || !section) { logger.error("Greeks analysis container/section not found."); return; }
    if (!asset || !portfolioGreeksData || typeof portfolioGreeksData !== 'object') {
        logger.warn("Greeks analysis skipped: Missing asset or valid Greeks data.");
        setElementState(section, 'hidden'); return;
    }

    // Check if all greek values are effectively zero or null
    const allZeroOrNull = Object.values(portfolioGreeksData).every(v => v === null || Math.abs(v) < 1e-9);
    if (allZeroOrNull) {
        logger.info("Greeks analysis skipped: All portfolio Greeks are zero or N/A.");
         container.innerHTML = '<p class="placeholder-text">No net option exposure to analyze Greeks.</p>';
         setElementState(section, 'content'); // Show section
         setElementState(container, 'content'); // Show container with message
         return;
    }

    logger.info(`Fetching Greeks analysis for ${asset}...`);
    setElementState(section, 'content'); // Ensure section is visible
    setElementState(container, 'loading', 'Fetching Greeks analysis...');

    try {
        if (typeof marked === 'undefined') { // Ensure markdown parser loaded
             await new Promise(resolve => setTimeout(resolve, 200)); // Wait briefly
             if (typeof marked === 'undefined') throw new Error("Markdown parser failed to load.");
        }

        const requestBody = {
            asset_symbol: asset,
            portfolio_greeks: portfolioGreeksData // Send the calculated totals
        };

        const data = await fetchAPI("/get_greeks_analysis", {
            method: "POST", body: JSON.stringify(requestBody)
        });

        const rawAnalysis = data?.greeks_analysis || "*Greeks analysis generation failed or returned empty.*";
        const potentiallySanitized = rawAnalysis.replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, '');
        container.innerHTML = marked.parse(potentiallySanitized);
        setElementState(container, 'content');
        logger.info(`Successfully rendered Greeks analysis for ${asset}`);

    } catch (error) {
        logger.error(`Error fetching or rendering Greeks analysis for ${asset}:`, error);
        // Display error within the Greeks analysis container
        setElementState(container, 'error', `Greeks Analysis Error: ${error.message}`);
        // Keep section visible but show error in container
        setElementState(section, 'content');
    }
}

/** Renders the Plotly chart */
async function renderPayoffChart(containerElement, figureJsonString) {
    logger.debug("Attempting to render Plotly chart...");

    if (!containerElement) {
        logger.error("renderPayoffChart: Target container element not found.");
        // Don't set state here, let caller handle it
        return;
    }
    if (typeof Plotly === 'undefined') {
        logger.error("renderPayoffChart: Plotly.js library is not loaded.");
        setElementState(containerElement, 'error', 'Charting library failed to load.');
        return;
    }
     if (!figureJsonString || typeof figureJsonString !== 'string') {
         logger.error("renderPayoffChart: Invalid or missing figure JSON string.");
         setElementState(containerElement, 'error', 'Invalid chart data received.');
         return;
     }

    try {
        const figure = JSON.parse(figureJsonString);

        // Apply Layout Defaults/Overrides
        figure.layout = figure.layout || {};
        figure.layout.height = 450; // Consistent height
        figure.layout.autosize = true;
        figure.layout.margin = { l: 60, r: 30, t: 30, b: 50 }; // Adjusted margins
        figure.layout.template = 'plotly_white';
        figure.layout.showlegend = false;
        figure.layout.hovermode = 'x unified';
        figure.layout.font = { family: 'Arial, sans-serif', size: 12 }; // Consistent font

        // Axis Styling
        figure.layout.yaxis = figure.layout.yaxis || {};
        figure.layout.yaxis.title = { text: 'Profit / Loss (₹)', standoff: 10 };
        figure.layout.yaxis.automargin = true;
        figure.layout.yaxis.gridcolor = 'rgba(220, 220, 220, 0.7)';
        figure.layout.yaxis.zeroline = true; // Show zero line clearly
        figure.layout.yaxis.zerolinecolor = 'rgba(0, 0, 0, 0.5)';
        figure.layout.yaxis.zerolinewidth = 1;
        figure.layout.yaxis.tickprefix = "₹";
        figure.layout.yaxis.tickformat = ',.0f';

        figure.layout.xaxis = figure.layout.xaxis || {};
        figure.layout.xaxis.title = { text: 'Underlying Spot Price', standoff: 10 };
        figure.layout.xaxis.automargin = true;
        figure.layout.xaxis.gridcolor = 'rgba(220, 220, 220, 0.7)';
        figure.layout.xaxis.zeroline = false;
        figure.layout.xaxis.tickformat = ',.0f';

        // Plotly configuration
        const plotConfig = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'lasso2d', 'select2d', 'toImage']
        };

        // Ensure container is ready
        containerElement.style.display = '';
        containerElement.innerHTML = ''; // Clear placeholder

        // Use Plotly.react for potentially better updates/performance
        await Plotly.react(containerElement.id, figure.data, figure.layout, plotConfig);

        setElementState(containerElement, 'content');
        logger.info("Successfully rendered Plotly chart using Plotly.react.");

    } catch (renderError) {
        logger.error("Error during Plotly chart processing:", renderError);
        setElementState(containerElement, 'error', `Chart Display Error: ${renderError.message}`);
    }
}

// ===============================================================
// Misc Helpers (Corrected)
// ===============================================================

/** Helper to round numbers safely */
function roundToPrecision(num, precision) {
    if (typeof num !== 'number' || !isFinite(num)) {
        return null; // Return null for non-finite numbers
    }
    const factor = Math.pow(10, precision);
    return Math.round(num * factor) / factor;
}