const API_BASE = "http://127.0.0.1:8000";
let niftyPrice = 0;
let strategyPositions = [];
let lastExpiry = null;
let lastScrollTop = 0; 

// ✅ Save & Restore Scroll Position
window.addEventListener("scroll", () => { lastScrollTop = window.scrollY; });
window.addEventListener("load", () => { window.scrollTo(0, lastScrollTop); });

document.addEventListener("DOMContentLoaded", async () => {
    await fetchNiftyPrice();
    await loadAssets();
    startAutoRefresh();
});

// ✅ Fetch Available Assets
async function loadAssets() {
    try {
        let response = await fetch(`${API_BASE}/get_assets`);
        let data = await response.json();
        let assetDropdown = document.getElementById("asset");
        assetDropdown.innerHTML = "";

        data.assets.forEach(asset => {
            let option = document.createElement("option");
            option.value = asset;
            option.textContent = asset;
            assetDropdown.appendChild(option);
        });

        assetDropdown.addEventListener("change", async () => {
            await updateSelectedAsset();
            await fetchNiftyPrice();
            await fetchExpiries();
        });

        await updateSelectedAsset();
        await fetchExpiries();
    } catch (error) {
        console.error("Error loading assets:", error);
    }
}


// ✅ Include a Markdown parser (marked.js)
document.addEventListener("DOMContentLoaded", function () {
    if (!window.marked) {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
        script.onload = () => console.log("Markdown parser loaded.");
        document.head.appendChild(script);
    }
});

// ✅ Trigger analysis when an asset is selected
document.getElementById("asset").addEventListener("change", fetchAnalysis);

// ✅ Fetch Stock Analysis from Backend and Render as Markdown
async function fetchAnalysis() {
    const asset = document.getElementById("asset").value;
    if (!asset) return;

    const analysisContainer = document.getElementById("analysisResult");
    analysisContainer.innerText = "Fetching analysis...";

    try {
        const response = await fetch(`${API_BASE}/get_stock_analysis`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ asset })
        });

        if (!response.ok) throw new Error("Failed to fetch stock analysis.");

        const data = await response.json();
        
        // ✅ Convert Markdown to HTML
        const markdownText = data.analysis;
        analysisContainer.innerHTML = marked.parse(markdownText);

    } catch (error) {
        console.error("Error fetching stock analysis:", error);
        analysisContainer.innerText = "Error fetching data.";
    }
}



// ✅ Update Selected Asset in Backend
async function updateSelectedAsset() {
    let asset = document.getElementById("asset").value;
    await fetch(`${API_BASE}/update_selected_asset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ asset: asset })
    });
}

// ✅ Fetch Expiry Dates for Selected Asset
async function fetchExpiries() {
    let asset = document.getElementById("asset").value;
    let expiryDropdown = document.getElementById("expiry");
    expiryDropdown.innerHTML = "<option>Loading...</option>";

    try {
        let response = await fetch(`${API_BASE}/expiry_dates?asset=${asset}`);
        let data = await response.json();
        expiryDropdown.innerHTML = "";

        data.expiry_dates.forEach(expiry => {
            let option = document.createElement("option");
            option.value = expiry;
            option.textContent = expiry;
            expiryDropdown.appendChild(option);
        });

        expiryDropdown.removeEventListener("change", fetchOptionChain);
        expiryDropdown.addEventListener("change", fetchOptionChain);

        if (lastExpiry !== data.expiry_dates[0]) {
            lastExpiry = data.expiry_dates[0];
            await fetchOptionChain(true);
        } else {
            await fetchOptionChain(false);
        }
    } catch (error) {
        console.error("Error fetching expiry dates:", error);
    }
}

// ✅ Fetch NIFTY Spot Price
async function fetchNiftyPrice() {
    try {
        let asset = document.getElementById("asset").value || "NIFTY";
        let response = await fetch(`${API_BASE}/get_spot_price?asset=${asset}`);
        let data = await response.json();
        niftyPrice = data.spot_price;
        document.getElementById("niftyPrice").textContent = `Spot Price: ${niftyPrice}`;
    } catch (error) {
        console.error("Error fetching spot price:", error);
    }
}

// ✅ Fetch & Display Option Chain Data
async function fetchOptionChain(scrollToATM = false) {
    let asset = document.getElementById("asset").value;
    let expiry = document.getElementById("expiry").value;
    let tableBody = document.querySelector("#optionChainTable tbody");

    if (!expiry) return;

    try {
        let response = await fetch(`${API_BASE}/get_option_chain?asset=${asset}&expiry=${expiry}`);
        if (!response.ok) throw new Error("Failed to fetch option chain data.");
        let data = await response.json();

        if (!data.option_chain || Object.keys(data.option_chain).length === 0) {
            tableBody.innerHTML = "<tr><td colspan='7'>No data available</td></tr>";
            return;
        }

        let atmStrike = findATMStrike(Object.keys(data.option_chain));
        let newTbody = document.createElement("tbody");

        Object.keys(data.option_chain).forEach(strike => {
            let optionData = data.option_chain[strike];
            let tr = document.createElement("tr");

            tr.className = strike == atmStrike ? "atm-strike" : "";
            tr.innerHTML = `
                <td class="call" onclick="addPosition(${strike}, 'CE')">${formatValue(optionData.call?.last_price)}</td>
                <td class="call">${formatValue(optionData.call?.open_interest)}</td>
                <td class="call">${formatValue(optionData.call?.implied_volatility)}</td>
                <td class="strike">${strike}</td>
                <td class="put">${formatValue(optionData.put?.implied_volatility)}</td>
                <td class="put">${formatValue(optionData.put?.open_interest)}</td>
                <td class="put" onclick="addPosition(${strike}, 'PE')">${formatValue(optionData.put?.last_price)}</td>
            `;

            newTbody.appendChild(tr);
        });

        tableBody.parentNode.replaceChild(newTbody, tableBody);

        if (scrollToATM) {
            document.querySelector(".atm-strike").scrollIntoView({ behavior: "smooth", block: "center" });
        }

    } catch (error) {
        console.error("Error fetching option chain:", error);
        tableBody.innerHTML = "<tr><td colspan='7'>Error loading data</td></tr>";
    }
}

// ✅ Add Position to Strategy
function addPosition(strike, type) {
    let expiry = document.getElementById("expiry").value;
    let newPosition = { strike_price: strike, expiry_date: expiry, option_type: type, lots: 1 };
    strategyPositions.push(newPosition);
    updateStrategyTable();
}

// ✅ Update Strategy Table
function updateStrategyTable() {
    let tableBody = document.querySelector("#strategyTable tbody");
    tableBody.innerHTML = "";

    strategyPositions.forEach((pos, index) => {
        let row = document.createElement("tr");
        row.innerHTML = `
            <td>${pos.strike_price}</td>
            <td>${pos.option_type}</td>
            <td>${pos.expiry_date}</td>
            <td>${pos.lots}</td>
            <td><button onclick="removePosition(${index})">Remove</button></td>
        `;
        tableBody.appendChild(row);
    });
}

// ✅ Remove Position
function removePosition(index) {
    strategyPositions.splice(index, 1);
    updateStrategyTable();
}

// ✅ Format Numbers
function formatValue(value) {
    return value !== undefined ? parseFloat(value).toFixed(2) : "-";
}

// ✅ Find ATM Strike
function findATMStrike(strikes) {
    return strikes.reduce((prev, curr) => Math.abs(curr - niftyPrice) < Math.abs(prev - niftyPrice) ? curr : prev);
}

// ✅ Auto Refresh Logic
function startAutoRefresh() {
    setInterval(() => {
        fetchNiftyPrice();
        let asset = document.getElementById("asset").value;
        let expiry = document.getElementById("expiry").value;
        if (expiry) fetchOptionChain();
    }, 3000);
}