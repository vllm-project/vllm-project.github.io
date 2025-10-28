document.addEventListener('DOMContentLoaded', function() {
  // Ablation inference data: BF16 vs FP8
  const ablationInferenceData = {
    "ModelA": {
      name: "Qwen3-0.6B",
      bf16: [0.41, 0.4, 0.41],
      fp8: [0.43, 0.43, 0.45]
    },
    "ModelB": {
      name: "Phi-3-vision-128k",
      bf16: [0.9, 0.74, 0.8],
      fp8: [0.69, 0.59, 0.44]
    }
  };

  function calcStatsAblInf(values) {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    return { mean, errorMinus: mean - min, errorPlus: max - mean };
  }

  const modelsAblInf = Object.keys(ablationInferenceData);
  const bf16StatsInf = modelsAblInf.map(m => calcStatsAblInf(ablationInferenceData[m].bf16));
  const fp8StatsInf = modelsAblInf.map(m => calcStatsAblInf(ablationInferenceData[m].fp8));

  const bf16TraceInf = {
    x: modelsAblInf.map(m => ablationInferenceData[m].name),
    y: bf16StatsInf.map(s => s.mean),
    name: "BF16",
    type: "bar",
    marker: { color: "#1f77b4" },
    error_y: {
      type: "data",
      symmetric: false,
      array: bf16StatsInf.map(s => s.errorPlus),
      arrayminus: bf16StatsInf.map(s => s.errorMinus),
      color: "#0d4a6e",
      thickness: 2,
      width: 6
    },
    text: bf16StatsInf.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#1f77b4", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>BF16: %{y:.2f}s<extra></extra>"
  };

  const fp8TraceInf = {
    x: modelsAblInf.map(m => ablationInferenceData[m].name),
    y: fp8StatsInf.map(s => s.mean),
    name: "FP8",
    type: "bar",
    marker: { color: "#ff7f0e" },
    error_y: {
      type: "data",
      symmetric: false,
      array: fp8StatsInf.map(s => s.errorPlus),
      arrayminus: fp8StatsInf.map(s => s.errorMinus),
      color: "#cc6600",
      thickness: 2,
      width: 6
    },
    text: fp8StatsInf.map(s => s.mean.toFixed(2) + "s"),
    textposition: "outside",
    textfont: { size: 12, color: "#ff7f0e", weight: "bold" },
    hovertemplate: "<b>%{x}</b><br>FP8: %{y:.2f}s<extra></extra>"
  };

  Plotly.newPlot("plotly-ablation-inference", [bf16TraceInf, fp8TraceInf], {
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
      range: [0, Math.max(...bf16StatsInf.map(s => s.mean + s.errorPlus), ...fp8StatsInf.map(s => s.mean + s.errorPlus)) * 1.25]
    },
    hovermode: "closest",
    legend: {
      x: 0.5,
      y: 1.15,
      xanchor: "center",
      yanchor: "top",
      orientation: "h"
    }
  }, {displayModeBar: true, responsive: true});
});
