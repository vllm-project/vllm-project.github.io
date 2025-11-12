document.addEventListener('DOMContentLoaded', function() {
  // Ablation study: BF16 vs FP8 quantization
  const timingDataAblation = {
    "Sleep Mode (BF16)": [
      { event: "A Model Load", duration: 32.56 },
      { event: "A Model Warm Up", duration: 2.69 },
      { event: "B Model Load", duration: 57.96 },
      { event: "B Model Warm Up", duration: 5.92 },
      { event: "A Model Wake up", duration: 0.28 },
      { event: "A Model Prompt", duration: 0.41 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.89 },
      { event: "B Model Prompt", duration: 0.9 },
      { event: "B Model Sleep", duration: 0.48 },
      { event: "A Model Wake up", duration: 0.27 },
      { event: "A Model Prompt", duration: 0.4 },
      { event: "A Model Sleep", duration: 0.1 },
      { event: "B Model Wake Up", duration: 0.93 },
      { event: "B Model Prompt", duration: 0.74 },
      { event: "B Model Sleep", duration: 0.5 },
      { event: "A Model Wake up", duration: 0.27 },
      { event: "A Model Prompt", duration: 0.41 },
      { event: "A Model Sleep", duration: 0.1 },
      { event: "B Model Wake Up", duration: 0.88 },
      { event: "B Model Prompt", duration: 0.8 }
    ],
    "Sleep Mode (FP8)": [
      { event: "A Model Load", duration: 37.71 },
      { event: "A Model Warm Up", duration: 2.34 },
      { event: "B Model Load", duration: 57.79 },
      { event: "B Model Warm Up", duration: 6.37 },
      { event: "A Model Wake up", duration: 0.18 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.06 },
      { event: "B Model Wake Up", duration: 0.79 },
      { event: "B Model Prompt", duration: 0.69 },
      { event: "B Model Sleep", duration: 0.31 },
      { event: "A Model Wake up", duration: 0.19 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.06 },
      { event: "B Model Wake Up", duration: 0.77 },
      { event: "B Model Prompt", duration: 0.59 },
      { event: "B Model Sleep", duration: 0.31 },
      { event: "A Model Wake up", duration: 0.16 },
      { event: "A Model Prompt", duration: 0.45 },
      { event: "A Model Sleep", duration: 0.07 },
      { event: "B Model Wake Up", duration: 0.78 },
      { event: "B Model Prompt", duration: 0.44 }
    ]
  };

  // Convert to segment format
  function createSegmentsAblation(timingData) {
    const segments = [];

    Object.entries(timingData).forEach(([scenario, events]) => {
      let cumulativeTime = 0;

      events.forEach(({ event, duration }) => {
        const [who, ...stageParts] = event.split(' ');
        const stage = stageParts.join(' ');

        let action, category;
        if (stage.includes('Load')) {
          action = 'Load';
          category = `${who} Load`;
        } else if (stage.includes('Wake')) {
          action = 'Wake';
          category = `${who} Wake`;
        } else if (stage.includes('Prompt')) {
          action = 'Prompt';
          category = `${who} Prompt`;
        } else if (stage.includes('Sleep')) {
          action = 'Sleep';
          category = `${who} Sleep`;
        } else if (stage.includes('Warm')) {
          action = 'Load';
          category = `${who} Load`;
        }

        segments.push({
          scenario,
          who,
          stage,
          action,
          start: cumulativeTime,
          end: cumulativeTime + duration,
          duration,
          category
        });

        cumulativeTime += duration;
      });
    });

    return segments;
  }

  const segmentsAblation = createSegmentsAblation(timingDataAblation);
  const colorMapAblation = {"A Load": "#1f77b4", "B Load": "#ff7f0e", "A Wake": "#2ca02c", "B Wake": "#17becf", "A Sleep": "#9467bd", "B Sleep": "#8c564b", "A Prompt": "#e377c2", "B Prompt": "#7f7f7f"};
  const categoriesAblation = Object.keys(colorMapAblation);

  const xAblation = segmentsAblation.map(d => d.duration);
  const baseAblation = segmentsAblation.map(d => d.start);
  const yAblation = segmentsAblation.map(d => d.scenario);
  const colorsAblation = segmentsAblation.map(d => colorMapAblation[d.category]);
  const customAblation = segmentsAblation.map(d => [d.scenario, d.category, d.stage, d.start, d.end]);

  const barsAblation = {
    type: "bar",
    orientation: "h",
    x: xAblation, base: baseAblation, y: yAblation,
    marker: { color: colorsAblation, line: {width:1, color:"rgba(0,0,0,0.35)"} },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]}<br>"+
      "Start %{customdata[3]:.2f}s → End %{customdata[4]:.2f}s<br>"+
      "<b>%{x:.2f}s</b><extra></extra>",
    customdata: customAblation,
    showlegend: false
  };

  const legendTracesAblation = categoriesAblation.map(name => ({
    type: "scatter", mode: "markers", x:[null], y:[null],
    name, marker: {color: colorMapAblation[name], size: 10},
    hoverinfo:"skip", showlegend:true
  }));

  Plotly.newPlot("plotly-ablation-quant", [barsAblation, ...legendTracesAblation], {
    barmode: "overlay",
    bargap: 0.05,
    margin: {l: 140, r: 30, t: 20, b: 40},
    xaxis: { title: "Time (seconds)", range: [0, 115] },
    yaxis: {
      categoryorder: "array",
      categoryarray: ["Sleep Mode (FP8)", "Sleep Mode (BF16)"]
    },
    hovermode: "closest",
    dragmode: "pan"
  }, {displayModeBar: true, responsive: true});
});
