document.addEventListener('DOMContentLoaded', function() {
  // A4000 Switching data
  const switchingDataA4000 = {
    "ModelA": {
      name: "Qwen3-0.6B",
      wake: [0.11, 0.1, 0.1],
      cold: [21.04, 20.98, 20.98]
    },
    "ModelB": {
      name: "Phi-3-vision-128k(4B)",
      wake: [0.8, 0.8, 0.8],
      cold: [46.01, 46.02, 46.02]
    }
  };

  function calcStatsSwitchA4000(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsSwitchA4000 = Object.keys(switchingDataA4000);
  const wakeStatsSwitchA4000 = modelsSwitchA4000.map(m => calcStatsSwitchA4000(switchingDataA4000[m].wake));
  const coldStatsSwitchA4000 = modelsSwitchA4000.map(m => calcStatsSwitchA4000(switchingDataA4000[m].cold));

  const wakeTraceSwitchA4000 = {
    x: modelsSwitchA4000.map(m => switchingDataA4000[m].name),
    y: wakeStatsSwitchA4000.map(s => s.mean),
    name: "Wake from Sleep",
    type: "bar",
    marker: { color: "#2ca02c" },
    error_y: {
      type: "data",
      symmetric: false,
      array: wakeStatsSwitchA4000.map(s => s.errorPlus),
      arrayminus: wakeStatsSwitchA4000.map(s => s.errorMinus),
      color: "#1a5e1a",
      thickness: 2,
      width: 6
    },
    text: wakeStatsSwitchA4000.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#2ca02c", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Wake Time: %{y:.2f}s<extra></extra>"
  };

  const coldTraceSwitchA4000 = {
    x: modelsSwitchA4000.map(m => switchingDataA4000[m].name),
    y: coldStatsSwitchA4000.map(s => s.mean),
    name: "Cold Start (Fresh Load)",
    type: "bar",
    marker: { color: "#d62728" },
    error_y: {
      type: "data",
      symmetric: false,
      array: coldStatsSwitchA4000.map(s => s.errorPlus),
      arrayminus: coldStatsSwitchA4000.map(s => s.errorMinus),
      color: "#8b1518",
      thickness: 2,
      width: 6
    },
    text: coldStatsSwitchA4000.map(s => s.mean.toFixed(1) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#d62728", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>Cold Start: %{y:.2f}s<extra></extra>"
  };

  const speedupsSwitchA4000 = wakeStatsSwitchA4000.map((w, i) => {
    const speedup = (coldStatsSwitchA4000[i].mean / w.mean).toFixed(0);
    return speedup;
  });

  Plotly.newPlot("plotly-switching-a4000", [wakeTraceSwitchA4000, coldTraceSwitchA4000], {
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
      range: [0, Math.max(...coldStatsSwitchA4000.map(s => s.mean + s.errorPlus)) * 1.15]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsSwitchA4000.map((m, i) => ({
      x: switchingDataA4000[m].name,
      y: coldStatsSwitchA4000[i].mean + coldStatsSwitchA4000[i].errorPlus + 3,
      text: `<b>${speedupsSwitchA4000[i]}x faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#2ca02c", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
