document.addEventListener('DOMContentLoaded', function() {
  // Ablation switching data: BF16 vs FP8
  const ablationSwitchingData = {
    "ModelA": {
      name: "Qwen3-0.6B",
      bf16: [0.28, 0.27, 0.27],
      fp8: [0.18, 0.19, 0.16]
    },
    "ModelB": {
      name: "Phi-3-vision-128k",
      bf16: [0.89, 0.93, 0.88],
      fp8: [0.79, 0.77, 0.78]
    }
  };

  function calcStatsAblSwitch(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsAblSwitch = Object.keys(ablationSwitchingData);
  const bf16StatsSwitch = modelsAblSwitch.map(m => calcStatsAblSwitch(ablationSwitchingData[m].bf16));
  const fp8StatsSwitch = modelsAblSwitch.map(m => calcStatsAblSwitch(ablationSwitchingData[m].fp8));

  const bf16TraceSwitch = {
    x: modelsAblSwitch.map(m => ablationSwitchingData[m].name),
    y: bf16StatsSwitch.map(s => s.mean),
    name: "BF16",
    type: "bar",
    marker: { color: "#1f77b4" },
    error_y: {
      type: "data",
      symmetric: false,
      array: bf16StatsSwitch.map(s => s.errorPlus),
      arrayminus: bf16StatsSwitch.map(s => s.errorMinus),
      color: "#0d4a6e",
      thickness: 2,
      width: 6
    },
    text: bf16StatsSwitch.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#1f77b4", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>BF16: %{y:.2f}s<extra></extra>"
  };

  const fp8TraceSwitch = {
    x: modelsAblSwitch.map(m => ablationSwitchingData[m].name),
    y: fp8StatsSwitch.map(s => s.mean),
    name: "FP8",
    type: "bar",
    marker: { color: "#ff7f0e" },
    error_y: {
      type: "data",
      symmetric: false,
      array: fp8StatsSwitch.map(s => s.errorPlus),
      arrayminus: fp8StatsSwitch.map(s => s.errorMinus),
      color: "#cc6600",
      thickness: 2,
      width: 6
    },
    text: fp8StatsSwitch.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#ff7f0e", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>FP8: %{y:.2f}s<extra></extra>"
  };

  // Calculate speedup percentages for annotation
  const speedupsSwitchAbl = bf16StatsSwitch.map((bf16, i) => {
    const reduction = ((bf16.mean - fp8StatsSwitch[i].mean) / bf16.mean * 100).toFixed(0);
    return reduction;
  });

  Plotly.newPlot("plotly-ablation-switching", [bf16TraceSwitch, fp8TraceSwitch], {
    barmode: "group",
    bargap: 0.15,
    bargroupgap: 0.1,
    margin: { l: 60, r: 30, t: 40, b: 50 },
    xaxis: {
      title: "",
      tickangle: 0
    },
    yaxis: {
      title: "Wake Time (seconds)",
      range: [0, Math.max(...bf16StatsSwitch.map(s => s.mean + s.errorPlus)) * 1.3]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    },
    annotations: modelsAblSwitch.map((m, i) => ({
      x: ablationSwitchingData[m].name,
      y: bf16StatsSwitch[i].mean + bf16StatsSwitch[i].errorPlus + 0.07,
      text: `<b>${speedupsSwitchAbl[i]}% faster</b>`,
      showarrow: false,
      font: { size: 11, color: "#ff7f0e", weight: "bold" },
      xanchor: "center"
    }))
  }, {displayModeBar: true, responsive: true});
});
