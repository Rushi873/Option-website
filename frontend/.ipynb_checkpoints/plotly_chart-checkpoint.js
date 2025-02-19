function updatePayoffChart() {
    let data = [{
        x: [17500, 18000, 18500, 19000], 
        y: [-1000, 500, 2000, 3500], 
        type: 'scatter'
    }];
    Plotly.newPlot("payoff-chart", data);
}
updatePayoffChart();
