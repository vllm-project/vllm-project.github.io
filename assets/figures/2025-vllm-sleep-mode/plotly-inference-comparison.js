document.addEventListener('DOMContentLoaded', function() {
  // Raw data: Wake Inference Time vs Cold Start Inference Time
  const inferenceData = {
    "ModelA": {
      name: "Qwen3-235B-A22B (TP=4)",
      wake: [1.8, 1.7, 0.92],
      cold: [3.8, 3.7, 3.72]
    },
    "ModelB": {
      name: "Qwen3-Coder-30B (TP=1)",
      wake: [1.0, 0.93, 0.54],
      cold: [3.7, 2.9, 2.45]
    }
  };

  // Calculate mean and error bars for each model
  function calcStats(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  // Prepare traces for both models
  const models = Object.keys(inferenceData);
  const wakeStats = models.map(m => calcStats(inferenceData[m].wake));
  const coldStats = models.map(m => calcStats(inferenceData[m].cold));

  const wakeTrace = {
    x: models.map(m => inferenceData[m].name),
    y: wakeStats.map(s => s.mean),
    name: "Wake Mode (Warmed Up)",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStats.map(s => s.errorPlus),
      arrayminus: wakeStats.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStats.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Mode: %{y:.2f}s<extra></extra>"
  };

  const coldTrace = {
    x: models.map(m => inferenceData[m].name),
    y: coldStats.map(s => s.mean),
    name: "Cold Start (Just Loaded)",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStats.map(s => s.errorPlus),
      arrayminus: coldStats.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStats.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  // Calculate speedup percentages for annotation
  const speedups = wakeStats.map((w, i) => {
    const reduction = ((coldStats[i].mean - w.mean) / coldStats[i].mean * 100).toFixed(0);
    return reduction;
  });

  Plotly.newPlot("plotly-inference-comparison", [wakeTrace, coldTrace], {
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
      range: [0, Math.max(...coldStats.map(s => s.mean + s.errorPlus)) * 1.2]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: models.map((m, i) => ({
      x: inferenceData[m].name,
      y: coldStats[i].mean + coldStats[i].errorPlus + 0.3,
      text: `<b>${speedups[i]}% faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
