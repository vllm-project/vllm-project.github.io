document.addEventListener('DOMContentLoaded', function() {
  // Sleep Levels Comparison timing data
  const timingDataLevelsComp = {
    "Sleep Mode (Level 1)": [
      { event: "A Model Load", duration: 36.27 },
      { event: "A Model Warm Up", duration: 2.53 },
      { event: "B Model Load", duration: 58.24 },
      { event: "B Model Warm Up", duration: 5.95 },
      { event: "A Model Wake up", duration: 0.25 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.82 },
      { event: "B Model Prompt", duration: 0.86 },
      { event: "B Model Sleep", duration: 0.41 },
      { event: "A Model Wake up", duration: 0.28 },
      { event: "A Model Prompt", duration: 0.41 },
      { event: "A Model Sleep", duration: 0.1 },
      { event: "B Model Wake Up", duration: 0.82 },
      { event: "B Model Prompt", duration: 0.71 },
      { event: "B Model Sleep", duration: 0.42 },
      { event: "A Model Wake up", duration: 0.25 },
      { event: "A Model Prompt", duration: 0.45 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 0.83 },
      { event: "B Model Prompt", duration: 0.71 }
    ],
    "Sleep Mode (Level 2)": [
      { event: "A Model Load", duration: 38.55 },
      { event: "A Model Warm Up", duration: 2.53 },
      { event: "B Model Load", duration: 61.23 },
      { event: "B Model Warm Up", duration: 5.75 },
      { event: "A Model Wake up", duration: 0.91 },
      { event: "A Model Prompt", duration: 0.68 },
      { event: "A Model Sleep", duration: 0.13 },
      { event: "B Model Wake Up", duration: 2.55 },
      { event: "B Model Prompt", duration: 0.78 },
      { event: "B Model Sleep", duration: 0.46 },
      { event: "A Model Wake up", duration: 0.78 },
      { event: "A Model Prompt", duration: 0.46 },
      { event: "A Model Sleep", duration: 0.12 },
      { event: "B Model Wake Up", duration: 2.62 },
      { event: "B Model Prompt", duration: 0.77 },
      { event: "B Model Sleep", duration: 0.45 },
      { event: "A Model Wake up", duration: 0.85 },
      { event: "A Model Prompt", duration: 0.44 },
      { event: "A Model Sleep", duration: 0.09 },
      { event: "B Model Wake Up", duration: 2.58 },
      { event: "B Model Prompt", duration: 0.72 }
    ],
    "WITHOUT Sleep Mode": [
      { event: "A Model Load", duration: 38.53 },
      { event: "A Model Prompt", duration: 4.66 },
      { event: "B Model Load", duration: 58.52 },
      { event: "B Model Prompt", duration: 6.55 },
      { event: "A Model Load", duration: 37.21 },
      { event: "A Model Prompt", duration: 3.8 },
      { event: "B Model Load", duration: 57.65 },
      { event: "B Model Prompt", duration: 6.21 },
      { event: "A Model Load", duration: 38.15 },
      { event: "A Model Prompt", duration: 2.56 },
      { event: "B Model Load", duration: 58.2 },
      { event: "B Model Prompt", duration: 6.15 }
    ]
  };

  // Convert to segment format
  function createSegmentsLevelsComp(timingData) {
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

  const segmentsLevelsComp = createSegmentsLevelsComp(timingDataLevelsComp);
  const colorMapLevelsComp = {"A Load": "#1f77b4", "B Load": "#ff7f0e", "A Wake": "#2ca02c", "B Wake": "#17becf", "A Sleep": "#9467bd", "B Sleep": "#8c564b", "A Prompt": "#e377c2", "B Prompt": "#7f7f7f"};
  const categoriesLevelsComp = Object.keys(colorMapLevelsComp);

  const xLevelsComp = segmentsLevelsComp.map(d => d.duration);
  const baseLevelsComp = segmentsLevelsComp.map(d => d.start);
  const yLevelsComp = segmentsLevelsComp.map(d => d.scenario);
  const colorsLevelsComp = segmentsLevelsComp.map(d => colorMapLevelsComp[d.category]);
  const customLevelsComp = segmentsLevelsComp.map(d => [d.scenario, d.category, d.stage, d.start, d.end]);

  const barsLevelsComp = {
    type: "bar",
    orientation: "h",
    x: xLevelsComp, base: baseLevelsComp, y: yLevelsComp,
    marker: { color: colorsLevelsComp, line: {width:1, color:"rgba(0,0,0,0.35)"} },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]}<br>"+
      "Start %{customdata[3]:.2f}s → End %{customdata[4]:.2f}s<br>"+
      "<b>%{x:.2f}s</b><extra></extra>",
    customdata: customLevelsComp,
    showlegend: false
  };

  const legendTracesLevelsComp = categoriesLevelsComp.map(name => ({
    type: "scatter", mode: "markers", x:[null], y:[null],
    name, marker: {color: colorMapLevelsComp[name], size: 10},
    hoverinfo:"skip", showlegend:true
  }));

  Plotly.newPlot("plotly-sleep-levels-comparison", [barsLevelsComp, ...legendTracesLevelsComp], {
    barmode: "overlay",
    bargap: 0.05,
    margin: {l: 160, r: 30, t: 20, b: 40},
    xaxis: { title: "Time (seconds)", range: [0, 365] },
    yaxis: {
      categoryorder: "array",
      categoryarray: ["WITHOUT Sleep Mode", "Sleep Mode (Level 2)", "Sleep Mode (Level 1)"]
    },
    hovermode: "closest",
    dragmode: "pan"
  }, {displayModeBar: true, responsive: true});
});
