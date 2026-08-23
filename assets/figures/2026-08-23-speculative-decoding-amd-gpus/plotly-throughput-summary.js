(function () {
  const datasets = ["GSM8K", "MATH500", "HumanEval", "MBPP"];
  const methodColors = {
    "Baseline": "#cbd5e1",
    "Gemma 4 MTP": "#f2b56b",
    "Native MTP": "#f2b56b",
    "EAGLE-3": "#80aee8",
    "DFlash": "#6bbf9a",
    "DSpark": "#b99bea",
  };

  const baselineThroughputs = {
    "google/gemma-4-26B-A4B-it": [2344, 2181, 1854, 2163],
    "google/gemma-4-31B-it": [1631, 1365, 1228, 1519],
    "Qwen/Qwen3-8B": [3698, 3530, 3226, 3268],
    "Qwen/Qwen3.5-27B": [1555, 1500, 1256, 1418],
    "Qwen/Qwen3.5-122B-A10B": [1494, 1446, 1105, 1459],
    "Qwen/Qwen3.6-27B": [1521, 1514, 1481, 1495],
    "Qwen/Qwen3.6-35B-A3B": [2275, 2235, 2193, 2258],
    "moonshotai/Kimi-K2.5": [324, 310, 301, 311],
    "MiniMaxAI/MiniMax-M3-MXFP8": [2086, 2468, 2317, 2277],
  };

  const measurementRows = [
    { target: "google/gemma-4-26B-A4B-it", method: "Gemma 4 MTP", speedups: [2.74, 2.83, 2.59, 2.62], throughputs: [6434, 6161, 4810, 5662], n: [5, 5, 5, 5] },
    { target: "google/gemma-4-26B-A4B-it", method: "EAGLE-3", speedups: [2.16, 2.27, 2.16, 2.11], throughputs: [5063, 4953, 3997, 4574], n: [3, 4, 4, 4] },
    { target: "google/gemma-4-26B-A4B-it", method: "DFlash", speedups: [2.70, 2.87, 2.79, 2.41], throughputs: [6327, 6267, 5183, 5214], n: [7, 7, 7, 7] },

    { target: "google/gemma-4-31B-it", method: "Gemma 4 MTP", speedups: [2.00, 2.20, 1.97, 1.99], throughputs: [3267, 3006, 2424, 3020], n: [4, 4, 4, 4] },
    { target: "google/gemma-4-31B-it", method: "EAGLE-3", speedups: [1.79, 2.12, 1.86, 1.84], throughputs: [2915, 2891, 2278, 2793], n: [3, 5, 5, 3] },
    { target: "google/gemma-4-31B-it", method: "DFlash", speedups: [1.95, 2.34, 2.05, 1.92], throughputs: [3183, 3197, 2514, 2914], n: [7, 7, 11, 7] },
    { target: "google/gemma-4-31B-it", method: "DSpark", speedups: [1.82, 2.20, 1.98, 1.84], throughputs: [2971, 3004, 2425, 2797], n: [7, 7, 7, 3] },

    { target: "Qwen/Qwen3-8B", method: "EAGLE-3", speedups: [1.18, 0.88, 1.05, 1.16], throughputs: [4347, 3105, 3376, 3798], n: [5, 7, 5, 5] },
    { target: "Qwen/Qwen3-8B", method: "DFlash", speedups: [1.27, 1.08, 1.27, 1.22], throughputs: [4678, 3828, 4103, 3982], n: [11, 11, 7, 7] },
    { target: "Qwen/Qwen3-8B", method: "DSpark", speedups: [1.63, 1.15, 1.48, 1.51], throughputs: [6032, 4048, 4769, 4936], n: [7, 7, 7, 7] },

    { target: "Qwen/Qwen3.5-27B", method: "Native MTP", speedups: [1.66, 1.70, 1.63, 1.60], throughputs: [2537, 2564, 2044, 2268], n: [5, 5, 4, 4] },
    { target: "Qwen/Qwen3.5-27B", method: "DFlash", speedups: [1.54, 1.65, 1.46, 1.44], throughputs: [2397, 2482, 1829, 2042], n: [7, 11, 3, 3] },

    { target: "Qwen/Qwen3.5-122B-A10B", method: "Native MTP", speedups: [2.08, 2.20, 1.85, 1.88], throughputs: [3107, 3183, 2044, 2747], n: [7, 7, 7, 7] },
    { target: "Qwen/Qwen3.5-122B-A10B", method: "DFlash", speedups: [1.58, 1.78, 1.66, 1.38], throughputs: [2356, 2572, 1838, 2019], n: [7, 7, 7, 3] },

    { target: "Qwen/Qwen3.6-27B", method: "Native MTP", speedups: [1.72, 1.78, 1.60, 1.61], throughputs: [2609, 2701, 2373, 2411], n: [5, 5, 4, 4] },
    { target: "Qwen/Qwen3.6-27B", method: "DFlash", speedups: [1.43, 1.59, 1.40, 1.38], throughputs: [2176, 2411, 2070, 2069], n: [7, 11, 7, 3] },

    { target: "Qwen/Qwen3.6-35B-A3B", method: "Native MTP", speedups: [1.43, 1.49, 1.28, 1.29], throughputs: [3253, 3334, 2811, 2903], n: [6, 6, 6, 6] },
    { target: "Qwen/Qwen3.6-35B-A3B", method: "DFlash", speedups: [1.88, 2.06, 1.84, 1.77], throughputs: [4276, 4600, 4036, 3990], n: [7, 7, 7, 7] },

    { target: "moonshotai/Kimi-K2.5", method: "EAGLE-3", speedups: [2.24, 2.33, 2.16, 1.99], throughputs: [728, 722, 649, 619], n: [4, 4, 4, 4] },
    { target: "moonshotai/Kimi-K2.5", method: "DFlash", speedups: [2.37, 2.68, 2.42, 2.21], throughputs: [768, 832, 727, 687], n: [7, 7, 7, 7] },

    { target: "MiniMaxAI/MiniMax-M3-MXFP8", method: "EAGLE-3", speedups: [1.82, 1.93, 2.09, 1.97], throughputs: [3807, 4772, 4835, 4487], n: [4, 4, 4, 4] },
  ];

  const targets = [...new Set(measurementRows.map((row) => row.target))];
  const rows = targets.flatMap((target) => [
    {
      target,
      method: "Baseline",
      speedups: datasets.map(() => 1),
      throughputs: baselineThroughputs[target],
      n: datasets.map(() => null),
    },
    ...measurementRows.filter((row) => row.target === target),
  ]);
  const initialTarget = targets[0];
  const chartId = "plotly-throughput-summary";
  const selectId = "plotly-throughput-target";
  const numberFormat = new Intl.NumberFormat("en-US");
  const visibleFor = (target) => rows.map((row) => row.target === target);
  const yFor = () => rows.map((row) => row.throughputs);
  const textFor = () => rows.map((row) => row.speedups.map((speedup, idx) => {
    if (!row.n[idx]) {
      return `${numberFormat.format(row.throughputs[idx])}`;
    }
    return `${numberFormat.format(row.throughputs[idx])}<br>N=${row.n[idx]}`;
  }));
  const yAxis = () => ({
    title: "Output throughput (tok/s)",
    ticksuffix: "",
    rangemode: "tozero",
    gridcolor: "rgba(148, 163, 184, 0.25)",
    zerolinecolor: "rgba(148, 163, 184, 0.5)",
  });

  const traces = rows.map((row) => ({
    type: "bar",
    name: row.method,
    x: datasets,
    y: row.throughputs,
    customdata: row.n.map((n, idx) => [
      row.target,
      n ? `N=${n}` : "non-speculative",
      row.speedups[idx],
      row.throughputs[idx],
    ]),
    text: row.speedups.map((speedup, idx) => (
      row.n[idx] ? `${numberFormat.format(row.throughputs[idx])}<br>N=${row.n[idx]}` : `${numberFormat.format(row.throughputs[idx])}`
    )),
    textposition: "auto",
    marker: {
      color: methodColors[row.method] || "#94a3b8",
      line: {
        color: row.method === "Baseline" ? "rgba(71, 85, 105, 0.45)" : "rgba(17, 24, 39, 0.2)",
        width: 1,
      },
    },
    hovertemplate:
      "<b>%{customdata[0]}</b><br>" +
      "Method: " + row.method + "<br>" +
      "Dataset: %{x}<br>" +
      "Output throughput: %{customdata[3]:,.0f} tok/s<br>" +
      "Speedup: %{customdata[2]:.2f}x<br>" +
      "%{customdata[1]}<extra></extra>",
    visible: row.target === initialTarget,
  }));

  const layout = {
    barmode: "group",
    bargap: 0.22,
    bargroupgap: 0.08,
    height: 560,
    margin: { l: 58, r: 24, t: 48, b: 76 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size: 13 },
    yaxis: yAxis(),
    xaxis: {
      title: "Experiment",
      tickangle: 0,
    },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.08,
      xanchor: "left",
      yanchor: "bottom",
    },
  };

  Plotly.newPlot(chartId, traces, layout, {
    responsive: true,
    displaylogo: false,
  });

  const select = document.getElementById(selectId);
  const updateChart = () => {
    const target = select ? select.value : initialTarget;
    Plotly.update(chartId, {
      visible: visibleFor(target),
      y: yFor(),
      text: textFor(),
    }, {
      yaxis: yAxis(),
    });
  };

  if (select) {
    targets.forEach((target) => {
      const option = document.createElement("option");
      option.value = target;
      option.textContent = target;
      select.appendChild(option);
    });

    select.value = initialTarget;
    select.addEventListener("change", updateChart);
  }

})();
