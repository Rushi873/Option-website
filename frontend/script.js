// ===============================================================
// Configuration & Constants
// ===============================================================
// const API_BASE = "http://localhost:8000"; // For Local Hosting
const API_BASE = "https://option-strategy-website.onrender.com"; // For Production
const REFRESH_INTERVAL_MS = 3000; // Auto-refresh interval (3 seconds)
const HIGHLIGHT_DURATION_MS = 1500; // How long highlights last

const SELECTORS = {
    assetDropdown: "#asset",
    expiryDropdown: "#expiry",
    spotPriceDisplay: "#niftyPrice", // Corrected ID (was niftyPrice, likely should be spotPrice)
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
    greeksTableBody: "#greeksTable tbody", // Added for direct access if needed
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
// Global State
// ===============================================================
let currentSpotPrice = 0;
let strategyPositions = [];
let activeAsset = null;
let autoRefreshIntervalId = null; // Timer ID for auto-refresh
let previousOptionChainData = {}; // Store previous chain data for highlighting
let previousSpotPrice = 0; // Store previous spot price for highlighting


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

    element.classList.remove('loading', 'error', 'loaded', 'hidden'); // Ensure hidden is removed unless setting hidden
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
             element.classList.add('hidden'); // Use class for hidden state
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
    // Don't hide global errors here, let the caller manage based on context
    // setElementState(SELECTORS.globalErrorDisplay, 'hidden');

    try {
        const response = await fetch(url, options);
        let responseData = null;
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
             responseData = await response.json(); // Attempt to parse JSON
        } else if (response.status !== 204) { // Handle non-JSON, non-empty responses if necessary
             logger.warn(`Received non-JSON response from ${method} ${url} (Status: ${response.status})`);
        }

        logger.debug(`fetchAPI Response Status: ${response.status} for ${method} ${url}`);

        if (!response.ok) {
            // Try to get detail, fallback to statusText, then generic message
            const errorMessage = responseData?.detail || responseData?.message || response.statusText || `HTTP error ${response.status}`;
            logger.error(`API Error (${method} ${url} - ${response.status}): ${errorMessage}`, responseData);
            throw new Error(errorMessage); // Throw standardized error
        }

         logger.debug(`fetchAPI Response Data:`, responseData);
        return responseData; // Return parsed data (or null for 204)

    } catch (error) {
        logger.error(`Network/Fetch Error (${method} ${url}):`, error);
        // Display global error only on Network/Fetch errors, not backend 4xx/5xx handled above
        if (!error.message.startsWith('HTTP error') && !error.message.includes('Stock data not found')) {
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Network Error: ${error.message || 'Failed to fetch'}`);
        }
        throw error; // Re-throw for specific UI handling if needed
    }
}

/** Applies a temporary highlight effect to an element */
function highlightElement(element) {
    if (!element) return;
    element.classList.add('value-changed');
    setTimeout(() => {
        element?.classList.remove('value-changed'); // Add optional chaining just in case
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

    // Add listener for strategy table interaction (event delegation)
    const strategyTableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (strategyTableBody) {
        strategyTableBody.addEventListener('change', handleStrategyTableChange);
        strategyTableBody.addEventListener('click', handleStrategyTableClick);
    }
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
    // previousOptionChainData needs to be populated by fetchOptionChain initially
    autoRefreshIntervalId = setInterval(refreshLiveData, REFRESH_INTERVAL_MS);
    // Optionally trigger one immediate refresh on start?
    // refreshLiveData();
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
    logger.debug(`Auto-refreshing data for ${activeAsset}...`);
    // Fetch data concurrently, but don't stop interval on individual errors
    // Use Promise.allSettled to avoid stopping if one fails
    const results = await Promise.allSettled([
        fetchNiftyPrice(activeAsset, true), // Pass true to indicate it's a refresh call
        fetchOptionChain(false, true) // No scroll, but is refresh call
    ]);

    // Log errors from settled promises if any
    if (results[0].status === 'rejected') {
        logger.warn(`Auto-refresh: Spot price fetch failed: ${results[0].reason?.message || results[0].reason}`);
    }
     if (results[1].status === 'rejected') {
        logger.warn(`Auto-refresh: Option chain fetch failed: ${results[1].reason?.message || results[1].reason}`);
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
    stopAutoRefresh(); // Stop refresh when changing asset

    // Clear previous data used for highlighting
    previousOptionChainData = {};
    previousSpotPrice = 0;
    currentSpotPrice = 0; // Reset current spot price

    if (!asset) {
        // Reset dependent UI if no asset selected
        populateDropdown(SELECTORS.expiryDropdown, [], "-- Select Asset First --");
        setElementState(SELECTORS.optionChainTableBody, 'content'); // Set state
        document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">Select an Asset</td></tr>`;
        setElementState(SELECTORS.analysisResultContainer, 'content'); // Set state
        document.querySelector(SELECTORS.analysisResultContainer).innerHTML = '';
        setElementState(SELECTORS.spotPriceDisplay, 'content'); // Set state
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
    setElementState(SELECTORS.globalErrorDisplay, 'hidden'); // Clear global error on new asset load

    // --- Call Debug Endpoint for Flawed Background Task (Backend Side) ---
    // This remains backend-specific, frontend doesn't directly control the backend thread
    // but changing the asset here *should* make the backend thread target the new asset.
    // The POST request sent previously is just for debugging/forcing selection if needed,
    // maybe not essential for normal operation if backend reads `selected_asset`.
    // Let's keep the debug POST for now as per your original code's intent.
    try {
        await fetchAPI('/debug/set_selected_asset', {
             method: 'POST', body: JSON.stringify({ asset: asset })
        });
        logger.warn(`Sent debug request to set backend selected_asset to ${asset}`);
    } catch (debugErr) {
        logger.error("Failed to send debug asset selection:", debugErr.message);
        // Don't block UI, just show a temporary global error
        setElementState(SELECTORS.globalErrorDisplay, 'error', `Debug Sync Failed: ${debugErr.message}`);
        // Hide it after a few seconds
        setTimeout(() => setElementState(SELECTORS.globalErrorDisplay, 'hidden'), 5000);
    }
    // --- End Debug Call ---

    try {
        // Fetch core data in parallel
        const [spotResult, expiryResult, analysisResult] = await Promise.allSettled([
            fetchNiftyPrice(asset), // Initial fetch
            fetchExpiries(asset),
            fetchAnalysis(asset)
        ]);

        let hasCriticalError = false;
        if (spotResult.status === 'rejected') {
            logger.error(`Error fetching spot price: ${spotResult.reason?.message || spotResult.reason}`);
            // Spot price failure might be acceptable for some views, but not ATM calc
        }
        if (expiryResult.status === 'rejected') {
            logger.error(`Error fetching expiries: ${expiryResult.reason?.message || expiryResult.reason}`);
            hasCriticalError = true; // Can't load chain without expiries
        }
        if (analysisResult.status === 'rejected') {
            logger.error(`Error fetching analysis: ${analysisResult.reason?.message || analysisResult.reason}`);
            // Analysis failure is less critical, page can still function
        }

        // If expiries loaded successfully, option chain fetch was triggered within fetchExpiries
        // If expiries failed, we need to show an error state for the chain
        if (hasCriticalError) {
             setElementState(SELECTORS.optionChainTableBody, 'error', 'Failed to load expiries');
             setElementState(SELECTORS.expiryDropdown, 'error', 'Failed to load expiries');
             // Optionally show a global error
             setElementState(SELECTORS.globalErrorDisplay, 'error', `Failed to load essential data for ${asset}.`);
        } else {
            // Start auto-refresh ONLY if initial load was mostly successful
            startAutoRefresh();
        }

    } catch (err) {
        // Should be caught by Promise.allSettled, but for safety
        logger.error(`Error fetching initial data for ${asset}:`, err);
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
        } else {
            logger.warn("No assets found. Cannot set default.");
            await handleAssetChange(); // Call handler to clear dependent fields
        }

    } catch (error) {
        setElementState(SELECTORS.assetDropdown, 'error', `Assets Error: ${error.message}`);
        setElementState(SELECTORS.expiryDropdown, 'error', 'Asset load failed');
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Asset load failed');
    }
}

/** Fetches stock analysis for the selected asset */
async function fetchAnalysis(asset) {
    if (!asset) return;
    setElementState(SELECTORS.analysisResultContainer, 'loading', 'Fetching analysis...');
    try {
        // Wait for marked.js
        for (let i=0; i<5 && typeof marked === 'undefined'; i++) {
            await new Promise(resolve => setTimeout(resolve, 300));
        }
        if (typeof marked === 'undefined') throw new Error("Markdown parser failed to load.");

        const data = await fetchAPI("/get_stock_analysis", {
            method: "POST", body: JSON.stringify({ asset })
        });
        const analysisContainer = document.querySelector(SELECTORS.analysisResultContainer);
        if (analysisContainer) {
            const rawAnalysis = data?.analysis || "*No analysis content received.*";
            // Basic sanitization (replace potential script tags)
            const potentiallySanitized = rawAnalysis.replace(/<script.*?>.*?<\/script>/gi, '');
            analysisContainer.innerHTML = marked.parse(potentiallySanitized);
            setElementState(SELECTORS.analysisResultContainer, 'content');
        }
    } catch (error) {
        // Use the error message from fetchAPI (which includes backend detail)
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
        populateDropdown(SELECTORS.expiryDropdown, expiries, "-- Select Expiry --", true); // Select first expiry
        setElementState(SELECTORS.expiryDropdown, 'content');

        const selectedExpiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
        if (selectedExpiry) {
            await handleExpiryChange(); // Load chain for the selected expiry
        } else {
            setElementState(SELECTORS.optionChainTableBody, 'content');
            document.querySelector(SELECTORS.optionChainTableBody).innerHTML = `<tr><td colspan="7">No expiry dates found for ${asset}</td></tr>`;
        }

    } catch (error) {
        setElementState(SELECTORS.expiryDropdown, 'error', `Expiries Error: ${error.message}`);
        setElementState(SELECTORS.optionChainTableBody, 'error', 'Could not load chain (expiry fetch failed)');
        throw error; // Re-throw so handleAssetChange knows about the failure
    }
}


/** Fetches and displays the spot price, optionally highlights changes */
async function fetchNiftyPrice(asset, isRefresh = false) {
    if (!asset) return;
    const priceElement = document.querySelector(SELECTORS.spotPriceDisplay);

    // Don't show loading state on silent refresh
    if (!isRefresh) {
        setElementState(SELECTORS.spotPriceDisplay, 'loading', 'Spot Price: ...');
    }

    try {
        const data = await fetchAPI(`/get_spot_price?asset=${encodeURIComponent(asset)}`);
        const newSpotPrice = data?.spot_price ?? 0;

        // Store previous price before updating global state if it's the first fetch or manual call
         if (!isRefresh || previousSpotPrice === 0) {
            previousSpotPrice = currentSpotPrice;
         }

        currentSpotPrice = newSpotPrice; // Update global state

        if (priceElement) {
            priceElement.textContent = `Spot Price: ${formatCurrency(currentSpotPrice, 2, 'N/A')}`;
             if (!isRefresh) { // Set content state only on manual/initial load
                 setElementState(SELECTORS.spotPriceDisplay, 'content');
             }

            // Highlight if the price changed during a refresh cycle
            if (isRefresh && currentSpotPrice !== previousSpotPrice && previousSpotPrice !== 0) {
                 logger.debug(`Spot price changed: ${previousSpotPrice} -> ${currentSpotPrice}`);
                 highlightElement(priceElement);
                 previousSpotPrice = currentSpotPrice; // Update previous price after highlighting
            } else if (isRefresh) {
                 // If it's a refresh but the price didn't change, ensure previous is updated
                 previousSpotPrice = currentSpotPrice;
            }
        }
    } catch (error) {
         if (!isRefresh) { // Show error state only on manual/initial load
             currentSpotPrice = 0; // Reset on error
             previousSpotPrice = 0;
             setElementState(SELECTORS.spotPriceDisplay, 'error', `Spot Price Error`);
         } else {
             // Log error during refresh but don't change UI state drastically
             logger.warn(`Spot Price refresh Error (${asset}):`, error.message);
         }
         // Optionally: Stop refresh if spot price fails repeatedly?
    }
}

/** Fetches and displays the option chain, optionally highlights changes */
async function fetchOptionChain(scrollToATM = false, isRefresh = false) {
    const asset = document.querySelector(SELECTORS.assetDropdown)?.value;
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    const tableBody = document.querySelector(SELECTORS.optionChainTableBody);

    if (!asset || !expiry) {
         if(tableBody) tableBody.innerHTML = `<tr><td colspan="7">Select Asset and Expiry</td></tr>`;
         if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
        return;
    }
    if (!isRefresh) {
        setElementState(SELECTORS.optionChainTableBody, 'loading');
    }

    try {
        // Ensure spot price is available for ATM calculation if scrolling
        if(currentSpotPrice <= 0 && scrollToATM) {
            logger.info("Spot price is 0, fetching before option chain for ATM scroll...");
            await fetchNiftyPrice(asset); // Wait for spot price fetch
            if(currentSpotPrice <= 0) {
                logger.warn("Spot price still unavailable, cannot calculate ATM strike accurately.");
                scrollToATM = false; // Disable ATM scrolling
            }
        }

        const data = await fetchAPI(`/get_option_chain?asset=${encodeURIComponent(asset)}&expiry=${encodeURIComponent(expiry)}`);
        const currentChainData = data?.option_chain; // Keep the fetched data structure

        if (!currentChainData || Object.keys(currentChainData).length === 0) {
            if(tableBody) tableBody.innerHTML = `<tr><td colspan="7">No option chain data available for ${asset} on ${expiry}</td></tr>`;
             if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
             previousOptionChainData = {}; // Clear previous data
            return;
        }

        // --- Render Table & Handle Highlights ---
        const strikes = Object.keys(currentChainData).map(Number).sort((a, b) => a - b);
        const atmStrike = currentSpotPrice > 0 ? findATMStrike(strikes, currentSpotPrice) : null;
        const newTbody = document.createElement("tbody");

        strikes.forEach(strike => {
            const optionData = currentChainData[strike];
            const prevOptionData = previousOptionChainData[strike] || { call: {}, put: {} }; // Get previous data for this strike
            const tr = document.createElement("tr");
            tr.dataset.strike = strike;

            if (atmStrike !== null && Math.abs(strike - atmStrike) < 0.01) {
                tr.classList.add("atm-strike");
            }

            const call = optionData.call || {};
            const put = optionData.put || {};
            const prevCall = prevOptionData.call || {};
            const prevPut = prevOptionData.put || {};

            // Define columns and their properties
            const columns = [
                // Calls
                { class: 'call clickable price', type: 'CE', key: 'last_price', prevData: prevCall, currentData: call, format: val => formatNumber(val, 2, '-') },
                { class: 'call oi', key: 'open_interest', prevData: prevCall, currentData: call, format: val => formatNumber(val, 0, '-') },
                { class: 'call iv', key: 'implied_volatility', prevData: prevCall, currentData: call, format: val => `${formatNumber(val, 2, '-')} %` },
                // Strike
                { class: 'strike', key: 'strike', isStrike: true, format: val => val },
                // Puts
                { class: 'put iv', key: 'implied_volatility', prevData: prevPut, currentData: put, format: val => `${formatNumber(val, 2, '-')} %` },
                { class: 'put oi', key: 'open_interest', prevData: prevPut, currentData: put, format: val => formatNumber(val, 0, '-') },
                { class: 'put clickable price', type: 'PE', key: 'last_price', prevData: prevPut, currentData: put, format: val => formatNumber(val, 2, '-') },
            ];

            columns.forEach(col => {
                const td = document.createElement('td');
                td.className = col.class;
                let currentValue = col.isStrike ? strike : col.currentData[col.key];
                td.textContent = col.format(currentValue);

                // Add data attributes for clickable cells
                if (col.type) td.dataset.type = col.type;
                if (col.key === 'last_price') td.dataset.price = currentValue || 0;

                // Highlight check for refresh
                if (isRefresh && !col.isStrike) {
                     let previousValue = col.prevData[col.key];
                     // Treat null/undefined as equivalent for change detection? Or different? Let's treat as different.
                     // Use a tolerance for float comparison if needed, e.g., for IV/Price
                     const tolerance = 0.001;
                     let changed = false;
                     if (typeof currentValue === 'number' && typeof previousValue === 'number') {
                         changed = Math.abs(currentValue - previousValue) > tolerance;
                     } else {
                         changed = currentValue !== previousValue; // Strict comparison for non-numbers or null/undefined
                     }

                    if (changed && typeof previousValue !== 'undefined') { // Only highlight if previous data existed
                        logger.debug(`Change detected: Strike ${strike}, Key ${col.key}, Prev: ${previousValue}, Curr: ${currentValue}`);
                        highlightElement(td);
                    }
                }
                tr.appendChild(td);
            });

            newTbody.appendChild(tr);
        });

        if (tableBody) {
             tableBody.parentNode.replaceChild(newTbody, tableBody);
             if (!isRefresh) setElementState(SELECTORS.optionChainTableBody, 'content');
        }

        // Add event listener *after* replacing
        newTbody.addEventListener('click', handleOptionChainClick);

        // Store current data for next refresh comparison AFTER rendering
        previousOptionChainData = currentChainData;

        // Scroll after rendering
        if (scrollToATM && atmStrike !== null && !isRefresh) { // Only scroll on initial load/manual change
            setTimeout(() => {
                const atmRow = newTbody.querySelector(".atm-strike");
                if (atmRow) {
                    atmRow.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
                } else { logger.warn("ATM strike row not found for scrolling."); }
            }, 100);
        }

    } catch (error) {
        if (!isRefresh) {
             setElementState(SELECTORS.optionChainTableBody, 'error', `Chain Error: ${error.message}`);
        } else {
             logger.warn(`Option Chain refresh error: ${error.message}`);
        }
        previousOptionChainData = {}; // Clear previous data on error
        // Optionally stop refresh on repeated errors?
    }
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
    const type = targetCell.dataset.type;
    const price = parseFloat(targetCell.dataset.price); // Price from the cell dataset

    if (!isNaN(strike) && type && !isNaN(price)) {
         addPosition(strike, type, price);
    } else {
        logger.warn('Could not add position - invalid data from clicked cell', { strike, type, price });
        alert('Could not retrieve option details. Please try again.');
    }
}

/** Handles clicks within the strategy table body (remove/toggle buttons) */
function handleStrategyTableClick(event) {
     // Check for Remove button click
     const removeButton = event.target.closest('button.remove-btn');
     if (removeButton?.dataset.index) {
         const index = parseInt(removeButton.dataset.index, 10);
         if (!isNaN(index)) { removePosition(index); }
         return; // Stop further processing
     }

     // Check for Toggle Buy/Sell button click
     const toggleButton = event.target.closest('button.toggle-buy-sell');
     if (toggleButton?.dataset.index) {
          const index = parseInt(toggleButton.dataset.index, 10);
         if (!isNaN(index)) { toggleBuySell(index); }
         return; // Stop further processing
     }
}

/** Handles changes within the strategy table body (for lots input) */
function handleStrategyTableChange(event) {
    const lotsInput = event.target.closest('input.lots-input');
     if (lotsInput?.dataset.index) { // Optional chaining
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
function addPosition(strike, type, price) {
    const expiry = document.querySelector(SELECTORS.expiryDropdown)?.value;
    if (!expiry) { alert("Please select an expiry date first."); return; }

    const lastPrice = (typeof price === 'number' && !isNaN(price)) ? price : 0;

    const newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type, // 'CE' or 'PE'
        lots: 1,           // Default to 1 lot (Buy)
        last_price: lastPrice,
        // ADDED: Store IV and days_to_expiry if available, needed for Greeks
        // These need to be fetched from the table or calculated
        iv: getOptionValueFromTable(strike, type, '.iv'), // Helper needed
        days_to_expiry: calculateDaysToExpiry(expiry), // Helper needed
    };
    strategyPositions.push(newPosition);
    updateStrategyTable();
    logger.info("Added position:", newPosition);
    // Don't auto-update chart on add, let user click Update
}

/** Helper to get a specific value (like IV) from the option chain table */
function getOptionValueFromTable(strike, type, cellSelectorSuffix) {
    const tableBody = document.querySelector(SELECTORS.optionChainTableBody);
    if (!tableBody) return null;
    const row = tableBody.querySelector(`tr[data-strike="${strike}"]`);
    if (!row) return null;
    const cellSelector = `td.${type.toLowerCase()}${cellSelectorSuffix}`; // e.g., td.ce.iv
    const cell = row.querySelector(cellSelector);
    if (!cell) return null;
    // Extract numeric value, handling percentages etc.
    const text = cell.textContent;
    if (text && text !== '-') {
        const num = parseFloat(text.replace('%', '').trim());
        return isNaN(num) ? null : num;
    }
    return null;
}

/** Helper to calculate days to expiry */
function calculateDaysToExpiry(expiryDateStr) {
    try {
        const expiryDate = new Date(expiryDateStr + 'T00:00:00'); // Treat as local date start
        const today = new Date();
        today.setHours(0, 0, 0, 0); // Set today to start of day for accurate diff
        const diffTime = expiryDate - today;
        if (diffTime < 0) return 0; // Expiry passed
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        return diffDays;
    } catch (e) {
        logger.error("Error calculating days to expiry for", expiryDateStr, e);
        return null; // Indicate error
    }
}

/** Updates the strategy table in the UI */
function updateStrategyTable() {
    const tableBody = document.querySelector(SELECTORS.strategyTableBody);
    if (!tableBody) return;
    tableBody.innerHTML = "";

    if (strategyPositions.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7">No positions added. Click option prices in the chain to add.</td></tr>';
        resetResultsUI();
        return;
    }

    strategyPositions.forEach((pos, index) => {
        const isLong = pos.lots >= 0;
        const positionType = isLong ? "BUY" : "SELL";
        const positionClass = isLong ? "long-position" : "short-position";
        const buttonClass = isLong ? "button-buy" : "button-sell";

        const row = document.createElement("tr");
        row.className = positionClass;
        row.dataset.index = index;

        row.innerHTML = `
            <td>${pos.option_type}</td>
            <td>${pos.strike_price}</td>
            <td>${pos.expiry_date}</td>
            <td>
                <input type="number" value="${pos.lots}" data-index="${index}" min="-100" max="100" step="1" class="lots-input" aria-label="Lots for position ${index+1}">
            </td>
            <td>
                <button class="toggle-buy-sell ${buttonClass}" data-index="${index}" title="Click to switch between Buy and Sell">${positionType}</button>
            </td>
            <td>${formatCurrency(pos.last_price, 2)}</td>
            <td>
                <button class="remove-btn" data-index="${index}" aria-label="Remove position ${index+1}">×</button>
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
        removePosition(index); // Remove position if lots are zero
    } else {
        const previousLots = strategyPositions[index].lots;
        strategyPositions[index].lots = newLots;

        // Update UI elements for the specific row
        const row = document.querySelector(`${SELECTORS.strategyTableBody} tr[data-index="${index}"]`);
        const toggleButton = row?.querySelector(`button.toggle-buy-sell[data-index="${index}"]`);

        if (toggleButton && row) {
             const isNowLong = newLots >= 0;
             const positionType = isNowLong ? "BUY" : "SELL";
             const buttonClass = isNowLong ? "button-buy" : "button-sell";

            toggleButton.textContent = positionType;
            toggleButton.classList.remove("button-buy", "button-sell");
            toggleButton.classList.add(buttonClass);
            row.className = isNowLong ? "long-position" : "short-position";
        } else {
             updateStrategyTable(); // Fallback if elements not found
        }
        logger.info(`Updated lots for index ${index} from ${previousLots} to ${newLots}`);
    }
    // Don't auto-update chart on lot change, let user click Update
}


/** Toggles a position between Buy and Sell (called by delegation) */
function toggleBuySell(index) {
    if (index < 0 || index >= strategyPositions.length) return;

    const previousLots = strategyPositions[index].lots;
    strategyPositions[index].lots *= -1; // Flip the sign

    // Handle case where flipping results in 0 (e.g., was 0 initially) -> default to Buy 1
     if (strategyPositions[index].lots === 0) {
         strategyPositions[index].lots = 1;
         logger.info(`Lots were 0 for index ${index}, toggling to 1 (BUY).`);
     }

    logger.info(`Toggled Buy/Sell for index ${index}. Prev lots: ${previousLots}, New lots: ${strategyPositions[index].lots}`);

    // Update the specific row UI
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
        row.className = isLong ? "long-position" : "short-position";
        lotsInput.value = strategyPositions[index].lots; // Update number input
    } else {
        updateStrategyTable(); // Fallback
    }
    // Don't auto-update chart on toggle, let user click Update
}


/** Removes a position from the strategy (called by delegation) */
function removePosition(index) {
     if (index < 0 || index >= strategyPositions.length) return;
    strategyPositions.splice(index, 1);
    updateStrategyTable();
    logger.info("Removed position at index", index);
    // Update results only if there are remaining positions
    if (strategyPositions.length > 0) {
        fetchPayoffChart(); // Update results after removal
    } else {
        resetResultsUI(); // Clear results if strategy is now empty
    }
}

/** Clears all positions and resets UI */
function clearAllPositions() {
    logger.info("Clearing all positions...");
    strategyPositions = [];
    updateStrategyTable();
    resetResultsUI();
    stopAutoRefresh(); // Stop refresh when clearing strategy
    logger.info("Strategy cleared.");
}

/** Resets the chart and results UI to initial state */
function resetResultsUI() {
     const chartContainer = document.querySelector(SELECTORS.payoffChartContainer);
     if (chartContainer) {
         // No Plotly purge needed as we use image now
         chartContainer.innerHTML = '<div class="placeholder-text">Add positions and click "Update" to see the payoff chart.</div>';
         setElementState(SELECTORS.payoffChartContainer, 'content');
     }

     // Reset other containers/tables
     setElementState(SELECTORS.taxInfoContainer, 'content');
     document.querySelector(SELECTORS.taxInfoContainer).innerHTML = "";
     setElementState(SELECTORS.greeksTable, 'content');
     const greeksTable = document.querySelector(SELECTORS.greeksTable);
     if (greeksTable) {
         const greekBody = greeksTable.querySelector('tbody'); if(greekBody) greekBody.innerHTML = "";
         const greekFoot = greeksTable.querySelector('tfoot'); if(greekFoot) greekFoot.innerHTML = "";
     }
     setElementState(SELECTORS.costBreakdownList, 'content');
     document.querySelector(SELECTORS.costBreakdownList).innerHTML = "";
     setElementState(SELECTORS.costBreakdownContainer, 'hidden'); // Hide the details initially

     // Reset metrics display
     displayMetric("N/A", SELECTORS.maxProfitDisplay, "Max Profit: ");
     displayMetric("N/A", SELECTORS.maxLossDisplay, "Max Loss: ");
     displayMetric("N/A", SELECTORS.breakevenDisplay, "Breakeven: ");
     displayMetric("N/A", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
     displayMetric("N/A", SELECTORS.netPremiumDisplay, "Net Premium: ");
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
    if (strategyPositions.length === 0) { resetResultsUI(); alert("Please add positions to the strategy first."); return; }
    if (!chartContainer) { logger.error("Payoff chart container element not found."); return; }

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

    // Prepare request data - *Ensure required fields for Greeks are included*
    const requestData = {
        asset: asset,
        strategy: strategyPositions.map(pos => {
            // Get IV and calculate Days to Expiry for Greeks calculation
            const iv = getOptionValueFromTable(pos.strike_price, pos.option_type, '.iv');
            const days_to_expiry = calculateDaysToExpiry(pos.expiry_date);

            // Log warning if IV or days are missing, as Greeks calc will fail for this leg
            if (iv === null) logger.warn(`Missing IV for leg: ${pos.option_type} ${pos.strike_price} @ ${pos.expiry_date}`);
            if (days_to_expiry === null) logger.warn(`Could not calculate days_to_expiry for leg: ${pos.expiry_date}`);

            return {
                // Fields for Payoff/Tax/Metrics
                option_type: pos.option_type,
                strike_price: String(pos.strike_price),
                option_price: String(pos.last_price),
                expiry_date: pos.expiry_date,
                lots: String(Math.abs(pos.lots)), // Send absolute lots
                tr_type: pos.lots >= 0 ? "b" : "s", // Determine buy/sell from sign
                // Additional fields potentially needed by backend for Greeks
                iv: iv, // Send fetched IV (can be null)
                days_to_expiry: days_to_expiry, // Send calculated days (can be null)
                op_type: pos.option_type === 'CE' ? 'c' : 'p', // Send 'c' or 'p'
                lot: Math.abs(pos.lots), // Send absolute lots again if needed separately
                // lot_size: Might need lot size if backend doesn't fetch it reliably? (Assuming backend handles it)
            };
        })
    };
    logger.debug("Sending request to /get_payoff_chart:", requestData);

    try {
        const data = await fetchAPI('/get_payoff_chart', {
            method: 'POST', body: JSON.stringify(requestData)
        });
        logger.debug("Received response from /get_payoff_chart:", data);

        // Validate response structure
        if (!data || data.success === false || !data.image_base64 || !data.metrics || !data.charges || data.greeks === undefined) {
            const errorDetail = data?.detail || data?.message || "Server response missing expected data or indicated failure.";
            logger.error("Payoff response validation failed.", errorDetail, data);
            throw new Error(errorDetail);
        }

        // Render Chart Image
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
             setElementState(SELECTORS.payoffChartContainer, 'error', 'Chart image missing');
             logger.error("Payoff chart image_base64 missing in successful response.");
        }

        // --- Display Metrics, Breakdown, Taxes, Greeks ---
        // Metrics
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
             logger.warn("Metrics data missing or invalid.");
             displayMetric("N/A", SELECTORS.maxProfitDisplay, "Max Profit: ");
             displayMetric("N/A", SELECTORS.maxLossDisplay, "Max Loss: ");
             displayMetric("N/A", SELECTORS.breakevenDisplay, "Breakeven: ");
             displayMetric("N/A", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
             displayMetric("N/A", SELECTORS.netPremiumDisplay, "Net Premium: ");
        }

        // Cost Breakdown
        const breakdownList = document.querySelector(SELECTORS.costBreakdownList);
        const breakdownContainer = document.querySelector(SELECTORS.costBreakdownContainer);
        const costBreakdownData = data.metrics?.cost_breakdown_per_leg;
        if (breakdownList && breakdownContainer && Array.isArray(costBreakdownData) && costBreakdownData.length > 0) {
            breakdownList.innerHTML = "";
            costBreakdownData.forEach(item => {
                const li = document.createElement("li");
                const premiumEffect = item.effect === 'Paid' ? `(Paid ${formatCurrency(Math.abs(item.total_premium))})` : `(Received ${formatCurrency(item.total_premium)})`;
                li.textContent = `${item.action} ${item.quantity} x ${item.type} @ ${item.strike} ${premiumEffect}`;
                breakdownList.appendChild(li);
            });
             setElementState(SELECTORS.costBreakdownContainer, 'content'); // Use state to show
             breakdownContainer.style.display = ""; // Ensure visible
             breakdownContainer.open = false; // Default closed
        } else if (breakdownContainer) {
            setElementState(SELECTORS.costBreakdownContainer, 'hidden');
        }

        // Taxes/Charges
        const taxContainer = document.querySelector(SELECTORS.taxInfoContainer);
        if (taxContainer && data.charges) {
            renderTaxTable(taxContainer, data.charges);
            setElementState(SELECTORS.taxInfoContainer, 'content');
        } else if (taxContainer) {
             taxContainer.innerHTML = "<p>Charge data unavailable.</p>";
             setElementState(SELECTORS.taxInfoContainer, 'content');
        }

        // Greeks (using the corrected renderer)
        const greeksTable = document.querySelector(SELECTORS.greeksTable);
        // data.greeks should be the list directly from the backend now
        if (greeksTable && Array.isArray(data.greeks)) {
             renderGreeksTable(greeksTable, data.greeks); // Pass the list
             setElementState(SELECTORS.greeksTable, 'content');
        } else if (greeksTable) {
            logger.warn("Greeks data received from backend is not an array:", data.greeks);
            renderGreeksTable(greeksTable, null); // Let renderer handle invalid input
            setElementState(SELECTORS.greeksTable, 'error', 'Invalid Greeks Format');
        }


    } catch (error) {
        logger.error("Error fetching or displaying payoff results:", error);
        setElementState(SELECTORS.payoffChartContainer, 'error', `Calculation Error: ${error.message}`);
        displayMetric("Error", SELECTORS.maxProfitDisplay, "Max Profit: ");
        displayMetric("Error", SELECTORS.maxLossDisplay, "Max Loss: ");
        displayMetric("Error", SELECTORS.breakevenDisplay, "Breakeven: ");
        displayMetric("Error", SELECTORS.rewardToRiskDisplay, "Reward:Risk: ");
        displayMetric("Error", SELECTORS.netPremiumDisplay, "Net Premium: ");
        setElementState(SELECTORS.taxInfoContainer, 'error', 'Calculation Failed');
        setElementState(SELECTORS.greeksTable, 'error', 'Calculation Failed');
        setElementState(SELECTORS.costBreakdownContainer, 'hidden');

    } finally {
         if (updateButton) updateButton.disabled = false;
    }
}

// --- Rendering Helpers for Payoff Results ---

function renderTaxTable(containerElement, taxData) {
    if (!taxData || !taxData.breakdown_per_leg || !taxData.charges_summary) {
        containerElement.innerHTML = "<p>Charge calculation data not available.</p>";
        return;
    }
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

    // Colspan = 12
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

/**
 * Renders the Greeks table.
 * @param {HTMLElement} tableElement - The table element to render into.
 * @param {Array | null | undefined} greeksList - The list of Greek results per leg directly from the backend.
 */
function renderGreeksTable(tableElement, greeksList) {
    // --- Check if the input is a valid array ---
    if (!Array.isArray(greeksList)) {
        logger.warn("Greeks data provided is not an array or is null/undefined.");
        tableElement.innerHTML = `
            <caption class="table-caption">Portfolio Option Greeks</caption>
            <thead><tr><th>Info</th></tr></thead>
            <tbody><tr><td class="error-message">Greeks data unavailable or invalid format received.</td></tr></tbody>`;
        return;
    }

    const totalLegsProcessed = greeksList.length; // Total legs returned by backend

    // --- Case 1: No legs returned by backend ---
    if (totalLegsProcessed === 0) {
         logger.info("No Greek results returned from backend.");
         tableElement.innerHTML = `
            <caption class="table-caption">Portfolio Option Greeks</caption>
            <thead><tr><th>Info</th></tr></thead>
            <tbody><tr><td>No strategy legs were processed for Greeks.</td></tr></tbody>`;
         return;
    }

    // --- Proceed with rendering ---
    let hasCalculatedGreeks = false; // Track if we can calculate totals
    const totals = { delta: 0, gamma: 0, theta: 0, vega: 0, rho: 0 };
    // Note: The backend scale factor was 100 in the example, adjust if different
    const backend_scaling_factor = 100.0;

    const tableBodyContent = greeksList.map(g => {
        // Each 'g' should be the object like: { leg_index: N, input_data: {...}, calculated_greeks: {...} }
        if (!g || !g.input_data || !g.calculated_greeks) {
            // Handle potential malformed item in the list (shouldn't happen ideally)
            logger.warn("Malformed item found in Greeks list:", g);
            return `<tr class="greeks-skipped"><td colspan="9" class="error-message">Invalid data for leg</td></tr>`;
        }

        const leg = g.input_data;
        const gv = g.calculated_greeks;
        const action = (leg.tr_type || '?').toUpperCase();
        // Determine quantity from input_data (assuming lots/lot_size are there)
        // Need to ensure input_data has these fields when backend prepares it.
        const lots = typeof leg.lot === 'number' ? leg.lot : parseInt(leg.lot || '0', 10);
        const lot_size = typeof leg.lot_size === 'number' ? leg.lot_size : parseInt(leg.lot_size || '0', 10);
        const quantity = lots * lot_size;
        const lotsDisplay = `${lots}x${lot_size}`;


        // Accumulate totals using PER-SHARE greeks * quantity
        if (quantity > 0 && typeof gv.delta === 'number' && typeof gv.gamma === 'number' && typeof gv.theta === 'number' && typeof gv.vega === 'number' && typeof gv.rho === 'number') {
            totals.delta += (gv.delta / backend_scaling_factor) * quantity;
            totals.gamma += (gv.gamma / backend_scaling_factor) * quantity;
            totals.theta += (gv.theta / backend_scaling_factor) * quantity; // Theta is per day
            totals.vega += (gv.vega / backend_scaling_factor) * quantity;   // Vega is per 1% IV change
            totals.rho += (gv.rho / backend_scaling_factor) * quantity;     // Rho is per 1% rate change
            hasCalculatedGreeks = true;
        } else {
             logger.warn(`Greeks values invalid or quantity invalid for leg index ${g.leg_index}, skipping totals. Greeks:`, gv, `Quantity: ${quantity}`);
        }

        // Render the row for this leg
        return `<tr class="greeks-calculated">
                    <td>${action}</td>
                    <td>${lotsDisplay}</td>
                    <td>${(leg.op_type || '?').toUpperCase()}</td>
                    <td>${leg.strike || '?'}</td>
                    <td>${formatNumber(gv.delta, 2, '-')}</td>
                    <td>${formatNumber(gv.gamma, 4, '-')}</td>
                    <td>${formatNumber(gv.theta, 2, '-')}</td>
                    <td>${formatNumber(gv.vega, 2, '-')}</td>
                    <td>${formatNumber(gv.rho, 2, '-')}</td>
                </tr>`;

    }).join('');

    // Final table structure
    tableElement.className = "results-table greeks-table";
    // Note: We don't have 'skipped_count' directly anymore unless backend provides it separately.
    // Caption reflects the number of legs *successfully* processed shown in the table.
    const captionText = `Portfolio Option Greeks (${totalLegsProcessed} Leg${totalLegsProcessed > 1 ? 's' : ''} Processed)`;
    tableElement.innerHTML = `
        <caption class="table-caption">${captionText}</caption>
        <thead>
            <tr>
                <th>Action</th>
                <th>Quantity</th>
                <th>Type</th>
                <th>Strike</th>
                <th title="Scaled Delta per share (x${backend_scaling_factor})">Δ Delta</th>
                <th title="Scaled Gamma per share (x${backend_scaling_factor})">Γ Gamma</th>
                <th title="Scaled Theta per share (x${backend_scaling_factor}, Daily)">Θ Theta/Day</th>
                <th title="Scaled Vega per share (x${backend_scaling_factor}, per 1% IV)">Vega</th>
                <th title="Scaled Rho per share (x${backend_scaling_factor}, per 1% Rate)">Ρ Rho</th>
            </tr>
        </thead>
        <tbody>${tableBodyContent}</tbody>
        <tfoot>
            ${hasCalculatedGreeks ?
                `<tr class="totals-row">
                    <td colspan="4">Total Portfolio Greeks</td>
                    <td>${formatNumber(totals.delta, 4)}</td>
                    <td>${formatNumber(totals.gamma, 4)}</td>
                    <td>${formatNumber(totals.theta, 4)}</td>
                    <td>${formatNumber(totals.vega, 4)}</td>
                    <td>${formatNumber(totals.rho, 4)}</td>
                </tr>`
                :
                `<tr class="totals-row">
                    <td colspan="9">No Greeks calculated for totals.</td>
                </tr>`
            }
        </tfoot>`;
}


// ===============================================================
// Misc Helpers
// ===============================================================

/** Finds the strike closest to the current spot price */
function findATMStrike(strikes = [], spotPrice) {
    if (!Array.isArray(strikes) || strikes.length === 0 || typeof spotPrice !== 'number' || spotPrice <= 0) {
         return null;
    }
    const numericStrikes = strikes.map(Number).filter(n => !isNaN(n));
    if(numericStrikes.length === 0) return null;

    return numericStrikes.reduce((prev, curr) =>
        Math.abs(curr - spotPrice) < Math.abs(prev - spotPrice) ? curr : prev
    );
}