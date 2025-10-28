document.addEventListener('DOMContentLoaded', function() {
  // Level 2 switching data
  const level2SwitchingData = {
    "ModelA": {
      name: "Qwen3-0.6B",
      wake: [0.91, 0.78, 0.85],
      cold: [38.53, 37.21, 38.15]
    },
    "ModelB": {
      name: "Phi-3-vision-128k",
      wake: [2.55, 2.62, 2.58],
      cold: [58.52, 57.65, 58.2]
    }
  };

  function calcStatsLevel2Switch(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsLevel2Switch = Object.keys(level2SwitchingData);
  const wakeStatsLevel2Switch = modelsLevel2Switch.map(m => calcStatsLevel2Switch(level2SwitchingData[m].wake));
  const coldStatsLevel2Switch = modelsLevel2Switch.map(m => calcStatsLevel2Switch(level2SwitchingData[m].cold));

  const wakeTraceLevel2Switch = {
    x: modelsLevel2Switch.map(m => level2SwitchingData[m].name),
    y: wakeStatsLevel2Switch.map(s => s.mean),
    name: "Wake from Sleep (Level 2)",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStatsLevel2Switch.map(s => s.errorPlus),
      arrayminus: wakeStatsLevel2Switch.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStatsLevel2Switch.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Time: %{y:.2f}s<extra></extra>"
  };

  const coldTraceLevel2Switch = {
    x: modelsLevel2Switch.map(m => level2SwitchingData[m].name),
    y: coldStatsLevel2Switch.map(s => s.mean),
    name: "Cold Start (Fresh Load)",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStatsLevel2Switch.map(s => s.errorPlus),
      arrayminus: coldStatsLevel2Switch.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStatsLevel2Switch.map(s => s.mean.toFixed(1) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  const speedupsLevel2Switch = wakeStatsLevel2Switch.map((w, i) => {
    const speedup = (coldStatsLevel2Switch[i].mean / w.mean).toFixed(0);
    return speedup;
  });

  Plotly.newPlot("plotly-level2-switching", [wakeTraceLevel2Switch, coldTraceLevel2Switch], {
    barmode: "group",
    bargap: 0.15,
    bargroupgap: 0.1,
    margin: { l: 60, r: 30, t: 40, b: 50 },
    xaxis: {
      title: "",
      tickangle: 0
    },
    yaxis: {
      title: "Switching Time (seconds)",
      range: [0, Math.max(...coldStatsLevel2Switch.map(s => s.mean + s.errorPlus)) * 1.15]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsLevel2Switch.map((m, i) => ({
      x: level2SwitchingData[m].name,
      y: coldStatsLevel2Switch[i].mean + coldStatsLevel2Switch[i].errorPlus + 3,
      text: `<b>${speedupsLevel2Switch[i]}x faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
