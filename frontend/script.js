const API_BASE = "https://option-strategy-website.onrender.com";
// const API_BASE = "http://localhost:8000";
let niftyPrice = 0;
let strategyPositions = [];
let lastExpiry = null;
let lastScrollTop = 0; 
let payoffChartInstance = null;  // ✅ Store chart instance to avoid multiple popups


// ✅ Save Scroll Position in localStorage
window.addEventListener("scroll", () => {
    localStorage.setItem("scrollPosition", window.scrollY);
});

// ✅ Restore Scroll Position on Page Load
window.addEventListener("load", () => {
    let savedScrollTop = localStorage.getItem("scrollPosition");
    if (savedScrollTop !== null) {
        window.scrollTo(0, parseInt(savedScrollTop));
    }
});

document.addEventListener("DOMContentLoaded", async () => {
    await fetchNiftyPrice();
    await loadAssets();
    startAutoRefresh();

    // Attach event listener for updating chart
    document.getElementById("updateChartBtn").addEventListener("click", fetchPayoffChart);
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
                <td class="call">${parseInt(optionData.call?.open_interest || 0)}</td>
                <td class="call">${formatValue(optionData.call?.implied_volatility)}</td>
                <td class="strike">${strike}</td>
                <td class="put">${formatValue(optionData.put?.implied_volatility)}</td>
                <td class="put">${parseInt(optionData.put?.open_interest || 0)}</td>
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





// ✅ Add Position to Strategy (Including Premium)
function addPosition(strike, type) {
    let expiry = document.getElementById("expiry").value;

    // Fetch the last traded price (LTP) from the option chain
    let optionRow = [...document.querySelectorAll("#optionChainTable tbody tr")]
        .find(row => row.children[3].textContent == strike); // Find row with matching strike

    if (!optionRow) {
        alert("Option data not found.");
        return;
    }

    let lastPrice = type === "CE" 
        ? parseFloat(optionRow.children[0].textContent) || 0  // Call Last Price
        : parseFloat(optionRow.children[6].textContent) || 0; // Put Last Price

    let newPosition = {
        strike_price: strike,
        expiry_date: expiry,
        option_type: type,
        lots: 1,
        last_price: lastPrice,  // Fetched from table
    };

    strategyPositions.push(newPosition);
    updateStrategyTable();
}

// ✅ Update Strategy Table (Now Editable Lots)
function updateStrategyTable() {
    let tableBody = document.querySelector("#strategyTable tbody");
    tableBody.innerHTML = "";

    strategyPositions.forEach((pos, index) => {
        const isLong = pos.lots >= 0;
        const positionType = isLong ? "L" : "S";
        const positionClass = isLong ? "long-label" : "short-label";

        let row = document.createElement("tr");
        row.innerHTML = `
            <td>${pos.strike_price}</td>
            <td>${pos.option_type}</td>
            <td>${pos.expiry_date}</td>
            <td>
                <input type="number" value="${pos.lots}" 
                    onchange="updateLots(${index}, this.value)" 
                    style="width: 50px; text-align: center;">
                <span class="${positionClass}">${positionType}</span>
            </td>
            <td>${pos.last_price.toFixed(2)}</td> 
            <td><button onclick="removePosition(${index})">Remove</button></td>
        `;
        tableBody.appendChild(row);
    });
}


// ✅ Function to Update Lots
function updateLots(index, value) {
    let newLots = parseInt(value);
    if (!isNaN(newLots)) {
        strategyPositions[index].lots = newLots;
        updateStrategyTable();  // Refresh the table to reflect L/S labels
    } else {
        alert("Please enter a valid number for lots.");
    }
}




// ✅ Remove Position
function removePosition(index) {
    strategyPositions.splice(index, 1);
    updateStrategyTable();
    fetchPayoffChart(); // 🔁 Update chart after removal
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
        if (document.getElementById("autoRefresh").checked) {
            fetchNiftyPrice();
            let expiry = document.getElementById("expiry").value;
            if (expiry) fetchOptionChain();
        }
    }, parseInt(document.getElementById("refreshInterval").value) * 1000);
}

// Clear position
function clearAllPositions() {
    // ✅ Clear frontend array (this is critical)
    strategyPositions.length = 0; // Ensures array is fully reset in memory

    // ✅ Clear strategy table
    const strategyTableBody = document.querySelector("#strategyTable tbody");
    strategyTableBody.innerHTML = "";

    // ✅ Clear chart canvas
    const chartCanvas = document.getElementById("payoffChart");
    const ctx = chartCanvas.getContext("2d");
    ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);

    // ✅ Reset metrics
    document.getElementById("maxProfit").innerText = "Max Profit: ";
    document.getElementById("maxLoss").innerText = "Max Loss: ";
    document.getElementById("breakeven").innerText = "Breakeven Points: ";
    document.getElementById("rewardToRisk").innerText = "Reward:Risk Ratio: ";
    document.getElementById("totalOptionPrice").innerText = "Total Option Cost: ";
    document.getElementById("costBreakdownList").innerHTML = "";

    // ✅ Clear backend strategy positions via FastAPI
    fetch(`${API_BASE}/clear_strategy`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        console.log("Backend strategy cleared:", data);
        fetchPayoffChart(); // force update chart with now-empty positions
    })
    .catch(error => {
        console.error("Error clearing backend strategy:", error);
    });
}








// ✅ Fetch Payoff Chart Data
// ✅ Fetch Payoff Chart Data from FastAPI
async function fetchPayoffChart() {
    if (strategyPositions.length === 0) {
        alert("Add positions to see the payoff chart.");
        return;
    }

    const chartContainer = document.getElementById("payoffChartContainer");
    chartContainer.innerHTML = '<div style="text-align: center; padding: 20px;">Loading chart...</div>';

    let asset = document.getElementById("asset").value;

    let requestData = {
        asset: asset,
        strategy: strategyPositions.map(pos => ({
            option_type: pos.option_type,
            strike_price: pos.strike_price.toString(),
            option_price: pos.last_price.toString(),
            expiry_date: pos.expiry_date,
            lots: Math.abs(pos.lots), 
            tr_type: pos.lots >= 0 ? "b" : "s"  // buy if positive, sell if negative
        }))
    };

    try {
        const response = await fetch(`${API_BASE}/get_payoff_chart`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to fetch payoff data: ${errorText}`);
        }

        const data = await response.json();

        // ✅ Payoff Chart
        chartContainer.innerHTML = "";
        const imgElement = document.createElement("img");
        imgElement.src = `data:image/png;base64,${data.image}`;
        imgElement.alt = "Payoff Chart";
        imgElement.style.width = "100%";
        imgElement.style.objectFit = "contain";
        chartContainer.appendChild(imgElement);

        // ✅ Strategy Metrics
        if (data.metrics) {
            const maxProfitDisplay = typeof data.metrics.max_profit === "string"
                ? data.metrics.max_profit
                : `₹${data.metrics.max_profit.toFixed(2)}`;
            const maxLossDisplay = typeof data.metrics.max_loss === "string"
                ? data.metrics.max_loss
                : `₹${data.metrics.max_loss.toFixed(2)}`;
            const breakevens = data.metrics.breakeven_points.length
                ? data.metrics.breakeven_points.map(p => `₹${p.toFixed(2)}`).join(', ')
                : "None";
            const rewardToRisk = data.metrics.reward_to_risk_ratio;
            const optionCost = data.metrics.total_option_price != null
                ? `₹${data.metrics.total_option_price.toFixed(2)}`
                : "N/A";
        
            document.getElementById("maxProfit").textContent = `Max Profit: ${maxProfitDisplay}`;
            document.getElementById("maxLoss").textContent = `Max Loss: ${maxLossDisplay}`;
            document.getElementById("breakeven").textContent = `Breakeven Points: ${breakevens}`;
            document.getElementById("rewardToRisk").textContent = `Reward:Risk Ratio: ${rewardToRisk}`;
            document.getElementById("totalOptionPrice").textContent = `Total Option Cost: ${optionCost}`;
        }
        // ✅ Option Cost Breakdown
        const breakdownList = document.getElementById("costBreakdownList");
        breakdownList.innerHTML = "";  // Clear previous
        
        if (data.metrics.cost_breakdown && Array.isArray(data.metrics.cost_breakdown)) {
            data.metrics.cost_breakdown.forEach(item => {
                const li = document.createElement("li");
                li.textContent = item;
                breakdownList.appendChild(li);
            });
            document.getElementById("costBreakdownContainer").style.display = "block";
        } else {
            document.getElementById("costBreakdownContainer").style.display = "none";
        }



        // ✅ Tax Calculation
        if (data.tax) {
            const taxInfo = document.getElementById("taxInfo");
            taxInfo.innerHTML = "<strong>Tax Summary (₹):</strong>";
        
            const table = document.createElement("table");
            table.style.width = "100%";
            table.style.borderCollapse = "collapse";
            table.style.marginTop = "10px";
        
            table.innerHTML = `
                <thead>
                    <tr style="background: #f0f0f0;">
                        <th style="padding: 6px; border: 1px solid #ccc;">Strike</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Type</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Lots</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Premium</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">STT</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Stamp Duty</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">SEBI</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Txn</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Brokerage</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">GST</th>
                        <th style="padding: 6px; border: 1px solid #ccc;">Note</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.tax.breakdown.map(t => `
                        <tr>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.strike}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.type}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.lots}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.premium}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.stt}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.stamp_duty}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.sebi_fee}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.txn_charge}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.brokerage}</td>
                            <td style="padding: 6px; border: 1px solid #ccc;">${t.gst}</td>
                            <td style="padding: 6px; border: 1px solid #ccc; font-style: italic;">${t.note}</td>
                        </tr>
                    `).join('')}
                </tbody>
                <tfoot>
                    <tr style="font-weight: bold; background: #eaeaea;">
                        <td colspan="4" style="padding: 6px; border: 1px solid #ccc;">Total</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.stt}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.stamp_duty}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.sebi_fee}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.txn_charges}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.brokerage}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">₹${data.tax.charges.gst}</td>
                        <td style="padding: 6px; border: 1px solid #ccc;">Total Tax: ₹${data.tax.total_cost}</td>
                    </tr>
                </tfoot>
            `;
        
            taxInfo.appendChild(table);
        }


        // ✅ Option Greeks
        if (data.greeks && Array.isArray(data.greeks)) {
            const greekTable = document.getElementById("greeksTable");
            greekTable.innerHTML = `
                <thead>
                    <tr>
                        <th>Strike</th>
                        <th>Type</th>
                        <th>Δ (Delta)</th>
                        <th>Γ (Gamma)</th>
                        <th>Θ (Theta)</th>
                        <th>Vega</th>
                        <th>Rho</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.greeks.map(g => `
                        <tr>
                            <td>${g.option.strike}</td>
                            <td>${g.option.op_type.toUpperCase()}</td>
                            <td>${g.delta.toFixed(2)}</td>
                            <td>${g.gamma.toFixed(2)}</td>
                            <td>${g.theta.toFixed(2)}</td>
                            <td>${g.vega.toFixed(2)}</td>
                            <td>${g.rho.toFixed(2)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            `;
        }

    } catch (error) {
        console.error("Error fetching payoff chart:", error);
        chartContainer.innerHTML = `
            <div style="color: red; text-align: center; padding: 20px;">
                Error: ${error.message}
            </div>`;
    }
}
