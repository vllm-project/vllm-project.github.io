document.addEventListener('DOMContentLoaded', function() {
  // A4000 Inference data
  const inferenceDataA4000 = {
    "ModelA": {
      name: "Qwen3-0.6B",
      wake: [0.44, 0.43, 0.43],
      cold: [2.64, 2.5, 2.63]
    },
    "ModelB": {
      name: "Phi-3-vision-128k(4B)",
      wake: [2.04, 1.73, 1.61],
      cold: [9.78, 9.01, 9.79]
    }
  };

  function calcStatsInfA4000(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsInfA4000 = Object.keys(inferenceDataA4000);
  const wakeStatsInfA4000 = modelsInfA4000.map(m => calcStatsInfA4000(inferenceDataA4000[m].wake));
  const coldStatsInfA4000 = modelsInfA4000.map(m => calcStatsInfA4000(inferenceDataA4000[m].cold));

  const wakeTraceInfA4000 = {
    x: modelsInfA4000.map(m => inferenceDataA4000[m].name),
    y: wakeStatsInfA4000.map(s => s.mean),
    name: "Wake Mode (Warmed Up)",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStatsInfA4000.map(s => s.errorPlus),
      arrayminus: wakeStatsInfA4000.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStatsInfA4000.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Mode: %{y:.2f}s<extra></extra>"
  };

  const coldTraceInfA4000 = {
    x: modelsInfA4000.map(m => inferenceDataA4000[m].name),
    y: coldStatsInfA4000.map(s => s.mean),
    name: "Cold Start (Just Loaded)",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStatsInfA4000.map(s => s.errorPlus),
      arrayminus: coldStatsInfA4000.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStatsInfA4000.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  const speedupsInfA4000 = wakeStatsInfA4000.map((w, i) => {
    const reduction = ((coldStatsInfA4000[i].mean - w.mean) / coldStatsInfA4000[i].mean * 100).toFixed(0);
    return reduction;
  });

  Plotly.newPlot("plotly-inference-a4000", [wakeTraceInfA4000, coldTraceInfA4000], {
    barmode: "group",
    bargap: 0.15,
    bargroupgap: 0.1,
    margin: { l: 60, r: 30, t: 40, b: 50 },
    xaxis: {
      title: "",
      tickangle: 0
    },
    yaxis: {
      title: "Inference Time (seconds)",
      range: [0, Math.max(...coldStatsInfA4000.map(s => s.mean + s.errorPlus)) * 1.2]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsInfA4000.map((m, i) => ({
      x: inferenceDataA4000[m].name,
      y: coldStatsInfA4000[i].mean + coldStatsInfA4000[i].errorPlus + 0.6,
      text: `<b>${speedupsInfA4000[i]}% faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
