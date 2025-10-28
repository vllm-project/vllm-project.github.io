document.addEventListener('DOMContentLoaded', function() {
  // A4000 GPU timing data
  const timingDataA4000 = {
    "WITH Sleep Mode (L1)": [
      { event: "A Model Load", duration: 21.01 },
      { event: "A Model Warm up", duration: 2.49 },
      { event: "B Model Load", duration: 46.01 },
      { event: "B Model Warm up", duration: 7.37 },
      { event: "A Model Wake up", duration: 0.11 },
      { event: "A Model Prompt", duration: 0.44 },
      { event: "A Model Sleep", duration: 0.13 },
      { event: "B Model Wake Up", duration: 0.8 },
      { event: "B Model Prompt", duration: 2.04 },
      { event: "B Model Sleep", duration: 0.68 },
      { event: "A Model Wake up", duration: 0.1 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.13 },
      { event: "B Model Wake Up", duration: 0.8 },
      { event: "B Model Prompt", duration: 1.73 },
      { event: "B Model Sleep", duration: 0.68 },
      { event: "A Model Wake up", duration: 0.1 },
      { event: "A Model Prompt", duration: 0.43 },
      { event: "A Model Sleep", duration: 0.13 },
      { event: "B Model Wake Up", duration: 0.8 },
      { event: "B Model Prompt", duration: 1.61 }
    ],
    "WITHOUT Sleep Mode": [
      { event: "A Model Load", duration: 21.04 },
      { event: "A Model Prompt", duration: 2.64 },
      { event: "B Model Load", duration: 46.01 },
      { event: "B Model Prompt", duration: 9.78 },
      { event: "A Model Load", duration: 20.98 },
      { event: "A Model Prompt", duration: 2.5 },
      { event: "B Model Load", duration: 46.02 },
      { event: "B Model Prompt", duration: 9.01 },
      { event: "A Model Load", duration: 20.98 },
      { event: "A Model Prompt", duration: 2.63 },
      { event: "B Model Load", duration: 46.02 },
      { event: "B Model Prompt", duration: 9.79 }
    ]
  };

  // Convert simplified data to full segment format
  function createSegmentsA4000(timingData) {
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

  const segmentsA4000 = createSegmentsA4000(timingDataA4000);
  const colorMapA4000 = {"A Load": "#1f77b4", "B Load": "#ff7f0e", "A Wake": "#2ca02c", "B Wake": "#17becf", "A Sleep": "#9467bd", "B Sleep": "#8c564b", "A Prompt": "#e377c2", "B Prompt": "#7f7f7f"};
  const categoriesA4000 = Object.keys(colorMapA4000);

  // Build arrays for a single stacked-horizontal bar trace using "base"
  const xA4000 = segmentsA4000.map(d => d.duration);
  const baseA4000 = segmentsA4000.map(d => d.start);
  const yA4000 = segmentsA4000.map(d => d.scenario);
  const colorsA4000 = segmentsA4000.map(d => colorMapA4000[d.category]);
  const customA4000 = segmentsA4000.map(d => [d.scenario, d.category, d.stage, d.start, d.end]);

  const barsA4000 = {
    type: "bar",
    orientation: "h",
    x: xA4000, base: baseA4000, y: yA4000,
    marker: { color: colorsA4000, line: {width:1, color:"rgba(0,0,0,0.35)"} },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>%{customdata[1]} — %{customdata[2]}<br>"+
      "Start %{customdata[3]:.2f}s → End %{customdata[4]:.2f}s<br>"+
      "<b>%{x:.2f}s</b><extra></extra>",
    customdata: customA4000,
    showlegend: false
  };

  // Legend-only dummies to produce a clean 8-item legend
  const legendTracesA4000 = categoriesA4000.map(name => ({
    type: "scatter", mode: "markers", x:[null], y:[null],
    name, marker: {color: colorMapA4000[name], size: 10},
    hoverinfo:"skip", showlegend:true
  }));

  Plotly.newPlot("plotly-sleep-mode-a4000", [barsA4000, ...legendTracesA4000], {
    barmode: "overlay",
    bargap: 0.05,
    margin: {l: 140, r: 30, t: 20, b: 40},
    xaxis: { title: "Time (seconds)", range: [0, 235] },
    yaxis: {
      categoryorder: "array",
      categoryarray: ["WITHOUT Sleep Mode", "WITH Sleep Mode (L1)"]
    },
    hovermode: "closest",
    dragmode: "pan"
  }, {displayModeBar: true, responsive: true});
});
