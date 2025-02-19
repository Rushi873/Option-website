document.addEventListener("DOMContentLoaded", function () {
    console.log("JavaScript Loaded Successfully!");

    const assetSelect = document.getElementById("asset-select");
    const fetchButton = document.getElementById("fetch-btn");
    
    fetchButton.addEventListener("click", fetchData);
    assetSelect.addEventListener("change", fetchData);

    // Auto-refresh every 3 seconds
    setInterval(fetchData, 3000);
});

// Fetch option chain data
function fetchData() {
    const asset = document.getElementById("asset-select").value;

    console.log("Fetching data for:", asset);

    fetch('/fetch_data', {
        method: "POST",
        body: new URLSearchParams({ asset: asset }),
        headers: { "Content-Type": "application/x-www-form-urlencoded" }
    }).then(response => response.json())
    .then(data => {
        console.log("Data received:", data);

        const tableBody = document.querySelector("#option-chain tbody");
        tableBody.innerHTML = "";

        // Display only ATM ± 5 strikes
        const atmIndex = Math.floor(data.length / 2);
        const filteredData = data.slice(Math.max(0, atmIndex - 5), atmIndex + 5);

        filteredData.forEach(row => {
            tableBody.innerHTML += `<tr>
                <td>${row.last_price || '-'}</td>
                <td>${row.strike_price}</td>
                <td>${row.ask_price || '-'}</td>
            </tr>`;
        });
    }).catch(error => console.error("Error fetching data:", error));
}
