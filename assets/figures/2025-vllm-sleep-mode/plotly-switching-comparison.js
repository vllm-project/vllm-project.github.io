document.addEventListener('DOMContentLoaded', function() {
  // Raw data: Wake Time vs Cold Start Time
  const switchingData = {
    "ModelA": {
      name: "Qwen3-235B-A22B (TP=4)",
      wake: [5.66, 5.29, 5.27],
      cold: [97.9, 97.4, 97.71]
    },
    "ModelB": {
      name: "Qwen3-Coder-30B (TP=1)",
      wake: [2.89, 2.86, 2.85],
      cold: [47.33, 47.47, 47.46]
    }
  };

  // Calculate mean and error bars for each model
  function calcStatsSwitch(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  // Prepare traces for both models
  const modelsSwitch = Object.keys(switchingData);
  const wakeStatsSwitch = modelsSwitch.map(m => calcStatsSwitch(switchingData[m].wake));
  const coldStatsSwitch = modelsSwitch.map(m => calcStatsSwitch(switchingData[m].cold));

  const wakeTraceSwitch = {
    x: modelsSwitch.map(m => switchingData[m].name),
    y: wakeStatsSwitch.map(s => s.mean),
    name: "Wake from Sleep",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStatsSwitch.map(s => s.errorPlus),
      arrayminus: wakeStatsSwitch.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStatsSwitch.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Time: %{y:.2f}s<extra></extra>"
  };

  const coldTraceSwitch = {
    x: modelsSwitch.map(m => switchingData[m].name),
    y: coldStatsSwitch.map(s => s.mean),
    name: "Cold Start (Fresh Load)",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStatsSwitch.map(s => s.errorPlus),
      arrayminus: coldStatsSwitch.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStatsSwitch.map(s => s.mean.toFixed(1) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  // Calculate speedup multiples for annotation
  const speedupsSwitch = wakeStatsSwitch.map((w, i) => {
    const speedup = (coldStatsSwitch[i].mean / w.mean).toFixed(0);
    return speedup;
  });

  Plotly.newPlot("plotly-switching-comparison", [wakeTraceSwitch, coldTraceSwitch], {
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
      range: [0, Math.max(...coldStatsSwitch.map(s => s.mean + s.errorPlus)) * 1.15]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsSwitch.map((m, i) => ({
      x: switchingData[m].name,
      y: coldStatsSwitch[i].mean + coldStatsSwitch[i].errorPlus + 5,
      text: `<b>${speedupsSwitch[i]}x faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
