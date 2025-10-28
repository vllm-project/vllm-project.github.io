document.addEventListener('DOMContentLoaded', function() {
  const timingData = {
    "WITH Sleep Mode (L1)": [
      { event: "A Model Load", duration: 97.61 },
      { event: "A Model Warm up", duration: 2.38 },
      { event: "B Model Load", duration: 47.63 },
      { event: "B Model Warm up", duration: 2.42 },
      { event: "A Model Wake up", duration: 5.66 },
      { event: "A Model Prompt", duration: 1.8 },
      { event: "A Model Sleep", duration: 6.01 },
      { event: "B Model Wake Up", duration: 2.89 },
      { event: "B Model Prompt", duration: 1 },
      { event: "B Model Sleep", duration: 2.78 },
      { event: "A Model Wake up", duration: 5.29 },
      { event: "A Model Prompt", duration: 1.7 },
      { event: "A Model Sleep", duration: 5.78 },
      { event: "B Model Wake Up", duration: 2.86 },
      { event: "B Model Prompt", duration: 0.93 },
      { event: "B Model Sleep", duration: 2.78 },
      { event: "A Model Wake up", duration: 5.27 },
      { event: "A Model Prompt", duration: 0.92 },
      { event: "A Model Sleep", duration: 5.89 },
      { event: "B Model Wake Up", duration: 2.85 },
      { event: "B Model Prompt", duration: 0.54 }
    ],
    "WITHOUT Sleep Mode": [
      { event: "A Model Load", duration: 97.9 },
      { event: "A Model Prompt", duration: 3.8 },
      { event: "B Model Load", duration: 47.33 },
      { event: "B Model Prompt", duration: 3.7 },
      { event: "A Model Load", duration: 97.4 },
      { event: "A Model Prompt", duration: 3.7 },
      { event: "B Model Load", duration: 47.47 },
      { event: "B Model Prompt", duration: 2.9 },
      { event: "A Model Load", duration: 97.71 },
      { event: "A Model Prompt", duration: 3.72 },
      { event: "B Model Load", duration: 47.46 },
      { event: "B Model Prompt", duration: 2.45 }
    ]
  };

  function createSegments(timingData) {
    const segments = [];

    Object.entries(timingData).forEach(([scenario, events]) => {
      let cumulativeTime = 0;

      events.forEach(({ event, duration }) => {
        const [who, ...stageParts] = event.split(' ');
        const stage = stageParts.join(' ');

        // Determine action and category from stage
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
        } else if (stage.includes('Warm up')) {
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

  const segments = createSegments(timingData);
  const colorMap = {"A Load": "#1f77b4", "B Load": "#ff7f0e", "A Wake": "#2ca02c", "B Wake": "#17becf", "A Sleep": "#9467bd", "B Sleep": "#8c564b", "A Prompt": "#e377c2", "B Prompt": "#7f7f7f"};
  const categories = Object.keys(colorMap);

  // Build arrays for a single stacked-horizontal bar trace using "base"
  const x = segments.map(d => d.duration);
  const base = segments.map(d => d.start);
  const y = segments.map(d => d.scenario);
  const colors = segments.map(d => colorMap[d.category]);
  const custom = segments.map(d => [d.scenario, d.category, d.stage, d.start, d.end]);

  const bars = {
    type: "bar",
    orientation: "h",
    x, base, y,
    marker: { color: colors, line: {width:1, color:"rgba(0,0,0,0.35)"} },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]}<br>"+
      "Start %{customdata[3]:.2f}s → End %{customdata[4]:.2f}s<br>"+
      "<b>%{x:.2f}s</b><extra></extra>",
    customdata: custom,
    showlegend: false
  };

  const legendTraces = categories.map(name => ({
    type: "scatter", mode: "markers", x:[null], y:[null],
    name, marker: {color: colorMap[name], size: 10},
    hoverinfo:"skip", showlegend:true
  }));

  Plotly.newPlot("plotly-sleep-mode", [bars, ...legendTraces], {
    barmode: "overlay",
    bargap: 0.05,
    margin: {l: 140, r: 30, t: 20, b: 40},
    xaxis: { title: "Time (seconds)", range: [0, 478.32] },
    yaxis: {
      categoryorder: "array",
      categoryarray: ["WITHOUT Sleep Mode", "WITH Sleep Mode (L1)"]
    },
    hovermode: "closest",
    dragmode: "pan"
  }, {displayModeBar: true, responsive: true});
});
