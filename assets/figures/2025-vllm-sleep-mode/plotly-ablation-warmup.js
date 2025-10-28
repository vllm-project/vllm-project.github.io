document.addEventListener('DOMContentLoaded', function() {
  // Ablation study: With vs Without Warm-Up
  const timingDataWarmup = {
    "With Warm-Up": [
      { event: "A Model Load", duration: 37.65 },
      { event: "A Model Warm Up", duration: 2.39 },
      { event: "B Model Load", duration: 62.69 },
      { event: "B Model Warm Up", duration: 6 },
      { event: "A Model Wake up", duration: 0.24 },
      { event: "A Model Prompt", duration: 0.45 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.89 },
      { event: "B Model Prompt", duration: 0.93 },
      { event: "B Model Sleep", duration: 0.47 },
      { event: "A Model Wake up", duration: 0.23 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.1 },
      { event: "B Model Wake Up", duration: 0.87 },
      { event: "B Model Prompt", duration: 0.73 },
      { event: "B Model Sleep", duration: 0.46 },
      { event: "A Model Wake up", duration: 0.23 },
      { event: "A Model Prompt", duration: 0.46 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.85 },
      { event: "B Model Prompt", duration: 0.73 }
    ],
    "Without Warm-Up": [
      { event: "A Model Load", duration: 37.91 },
      { event: "B Model Load", duration: 63.16 },
      { event: "A Model Wake up", duration: 0.24 },
      { event: "A Model Prompt", duration: 2.59 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.91 },
      { event: "B Model Prompt", duration: 6.61 },
      { event: "B Model Sleep", duration: 0.44 },
      { event: "A Model Wake up", duration: 0.26 },
      { event: "A Model Prompt", duration: 0.41 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.87 },
      { event: "B Model Prompt", duration: 0.7 },
      { event: "B Model Sleep", duration: 0.43 },
      { event: "A Model Wake up", duration: 0.27 },
      { event: "A Model Prompt", duration: 0.42 },
      { event: "A Model Sleep", duration: 0.1 },
      { event: "B Model Wake Up", duration: 0.86 },
      { event: "B Model Prompt", duration: 0.7 }
    ]
  };

  // Convert to segment format
  function createSegmentsWarmup(timingData) {
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

  const segmentsWarmup = createSegmentsWarmup(timingDataWarmup);
  const colorMapWarmup = {"A Load": "#1f77b4", "B Load": "#ff7f0e", "A Wake": "#2ca02c", "B Wake": "#17becf", "A Sleep": "#9467bd", "B Sleep": "#8c564b", "A Prompt": "#e377c2", "B Prompt": "#7f7f7f"};
  const categoriesWarmup = Object.keys(colorMapWarmup);

  const xWarmup = segmentsWarmup.map(d => d.duration);
  const baseWarmup = segmentsWarmup.map(d => d.start);
  const yWarmup = segmentsWarmup.map(d => d.scenario);
  const colorsWarmup = segmentsWarmup.map(d => colorMapWarmup[d.category]);
  const customWarmup = segmentsWarmup.map(d => [d.scenario, d.category, d.stage, d.start, d.end]);

  const barsWarmup = {
    type: "bar",
    orientation: "h",
    x: xWarmup, base: baseWarmup, y: yWarmup,
    marker: { color: colorsWarmup, line: {width:1, color:"rgba(0,0,0,0.35)"} },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]}<br>"+
      "Start %{customdata[3]:.2f}s → End %{customdata[4]:.2f}s<br>"+
      "<b>%{x:.2f}s</b><extra></extra>",
    customdata: customWarmup,
    showlegend: false
  };

  const legendTracesWarmup = categoriesWarmup.map(name => ({
    type: "scatter", mode: "markers", x:[null], y:[null],
    name, marker: {color: colorMapWarmup[name], size: 10},
    hoverinfo:"skip", showlegend:true
  }));

  Plotly.newPlot("plotly-ablation-warmup", [barsWarmup, ...legendTracesWarmup], {
    barmode: "overlay",
    bargap: 0.05,
    margin: {l: 140, r: 30, t: 20, b: 40},
    xaxis: { title: "Time (seconds)", range: [0, 120] },
    yaxis: {
      categoryorder: "array",
      categoryarray: ["Without Warm-Up", "With Warm-Up"]
    },
    hovermode: "closest",
    dragmode: "pan"
  }, {displayModeBar: true, responsive: true});
});
