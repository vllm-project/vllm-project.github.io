document.addEventListener('DOMContentLoaded', function() {
  // Level 2 inference data
  const level2InferenceData = {
    "ModelA": {
      name: "Qwen3-0.6B",
      wake: [0.68, 0.46, 0.44],
      cold: [4.66, 3.8, 2.56]
    },
    "ModelB": {
      name: "Phi-3-vision-128k",
      wake: [0.78, 0.77, 0.72],
      cold: [6.55, 6.21, 6.15]
    }
  };

  function calcStatsLevel2Inf(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsLevel2Inf = Object.keys(level2InferenceData);
  const wakeStatsLevel2Inf = modelsLevel2Inf.map(m => calcStatsLevel2Inf(level2InferenceData[m].wake));
  const coldStatsLevel2Inf = modelsLevel2Inf.map(m => calcStatsLevel2Inf(level2InferenceData[m].cold));

  const wakeTraceLevel2Inf = {
    x: modelsLevel2Inf.map(m => level2InferenceData[m].name),
    y: wakeStatsLevel2Inf.map(s => s.mean),
    name: "Wake Mode (Level 2)",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStatsLevel2Inf.map(s => s.errorPlus),
      arrayminus: wakeStatsLevel2Inf.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStatsLevel2Inf.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Mode: %{y:.2f}s<extra></extra>"
  };

  const coldTraceLevel2Inf = {
    x: modelsLevel2Inf.map(m => level2InferenceData[m].name),
    y: coldStatsLevel2Inf.map(s => s.mean),
    name: "Cold Start",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStatsLevel2Inf.map(s => s.errorPlus),
      arrayminus: coldStatsLevel2Inf.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStatsLevel2Inf.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  const speedupsLevel2Inf = wakeStatsLevel2Inf.map((w, i) => {
    const reduction = ((coldStatsLevel2Inf[i].mean - w.mean) / coldStatsLevel2Inf[i].mean * 100).toFixed(0);
    return reduction;
  });

  Plotly.newPlot("plotly-level2-inference", [wakeTraceLevel2Inf, coldTraceLevel2Inf], {
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
      range: [0, Math.max(...coldStatsLevel2Inf.map(s => s.mean + s.errorPlus)) * 1.2]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsLevel2Inf.map((m, i) => ({
      x: level2InferenceData[m].name,
      y: coldStatsLevel2Inf[i].mean + coldStatsLevel2Inf[i].errorPlus + 0.4,
      text: `<b>${speedupsLevel2Inf[i]}% faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
