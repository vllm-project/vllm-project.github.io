/* H3-Image figures. Visual roles follow the DeepSeek-V4.1-Flash blog's figkit. */
(async function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const svg = $('chart'), tip = $('tip');
  const NS = 'http://www.w3.org/2000/svg';
  const kind = document.body.dataset.figure;
  const fmt = (v, digits = 0) => v.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const money = v => '$' + fmt(v, 2);
  const query = new URLSearchParams(location.search);
  if (query.has('embed')) document.body.dataset.embed = '1';
  if (['light', 'dark'].includes(query.get('theme'))) document.documentElement.dataset.theme = query.get('theme');
  const node = (tag, attrs = {}, parent = svg) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    parent.appendChild(n); return n;
  };
  const text = (x, y, value, cls = 't-lab', attrs = {}) => {
    const n = node('text', {x, y, class: cls, ...attrs}); n.textContent = value; return n;
  };
  const reportHeight = () => {
    if (window.self !== window.top) window.parent.postMessage({type: 'vllm-embed-resize', height: Math.ceil($('fig').getBoundingClientRect().height + 12)}, '*');
  };
  new ResizeObserver(reportHeight).observe($('fig'));
  const color = p => 'var(--' + (p.platform === 'Vera Rubin' ? (p.topology === 'SP1×DP4' ? 'gold' : 'blue') : p.platform === 'GB300' ? 'green' : 'stone') + '-line)';
  const platformColor = p => p.platform === 'Vera Rubin' ? 'var(--gold-line)' : color(p);
  const shortName = p => (p.platform === 'Vera Rubin' ? 'Rubin' : p.platform) + ' ' + p.topology;
  const seriesKey = p => [p.platform, p.topology, p.precision].join(' / ');
  const dashed = p => p.platform !== 'Vera Rubin' ? '6 4' : '';
  let points;
  try {
    const response = await fetch('performance-benchmarks-data.json?v=4978697ff082');
    if (!response.ok) throw new Error('HTTP ' + response.status);
    points = (await response.json()).points;
  } catch (e) {
    $('note').textContent = 'Unable to load benchmark data. Please reload this figure.';
    document.body.dataset.error = e.message; reportHeight(); return;
  }
  const primary = points.filter(p => p.primary_view);
  const platforms = ['Vera Rubin', 'GB300', 'GB200'];
  const peakFor = platform => primary.filter(p => p.platform === platform).reduce((a, b) => a.images_per_hour > b.images_per_hour ? a : b);
  const c1For = platform => primary.filter(p => p.platform === platform && p.topology === 'SP4' && p.concurrency === 1)[0];
  const revenuePerGPU = (p, price, utilization) => p.images_per_hour * price * utilization / 4;
  function selectedPoints() {
    const filter = $('precision').value;
    return points.filter(p => filter === 'primary' ? p.primary_view : filter === 'sp4' ? p.topology === 'SP4' : filter === 'rubin' ? p.platform === 'Vera Rubin' : false);
  }
  function tooltip(target, lines) {
    target.setAttribute('tabindex', '0');
    target.setAttribute('aria-label', lines.join('. '));
    function show(x, y) {
      tip.replaceChildren();
      lines.forEach((line, i) => { const e = document.createElement(i ? 'div' : 'strong'); e.textContent = line; tip.appendChild(e); });
      tip.classList.add('show');
      const b = tip.getBoundingClientRect();
      tip.style.left = Math.max(4, Math.min(x + 14, innerWidth - b.width - 8)) + 'px';
      tip.style.top = Math.max(4, Math.min(y - b.height - 12, innerHeight - b.height - 4)) + 'px';
    }
    target.addEventListener('mousemove', e => show(e.clientX, e.clientY));
    target.addEventListener('focus', () => { const r = target.getBoundingClientRect(); show(r.x + r.width / 2, r.y); });
    ['mouseleave', 'blur'].forEach(e => target.addEventListener(e, () => tip.classList.remove('show')));
  }
  function legend(entries) {
    $('legend').replaceChildren();
    for (const entry of entries) {
      const span = document.createElement('span'), i = document.createElement('i');
      i.className = 'sw'; i.style.borderColor = entry.color;
      if (entry.dash) i.style.borderTopStyle = 'dashed';
      span.append(i, document.createTextNode(entry.label)); $('legend').appendChild(span);
    }
  }
  function reset(label) {
    svg.replaceChildren(); svg.setAttribute('aria-label', label);
    node('title').textContent = label;
    tip.classList.remove('show');
  }
  function drawCurve() {
    const data = selectedPoints(), metric = $('metric').value;
    const log = true, zoom = false;
    const compact = typeof innerWidth === 'number' && innerWidth < 560;
    svg.setAttribute('viewBox', compact ? '0 0 400 390' : '0 0 736 364');
    const maxLatency = Math.max(...data.map(p => p[metric]));
    const xmax = zoom ? Math.min(5, maxLatency) : log ? Math.pow(2, Math.ceil(Math.log2(maxLatency))) : Math.ceil(maxLatency / 5) * 5;
    const xmin = log ? .5 : 0;
    const ymax = Math.ceil(Math.max(...data.map(p => p.images_per_gpu_hour)) / 400) * 400;
    const x0 = compact ? 52 : 60, x1 = compact ? 380 : 700, top = 32, bottom = compact ? 320 : 296;
    const X = v => x0 + (log ? Math.log(v / xmin) / Math.log(xmax / xmin) : v / xmax) * (x1 - x0);
    const Y = v => bottom - v / ymax * (bottom - top);
    const metricName = metric === 'median_latency_s' ? 'median' : metric === 'p95_latency_s' ? 'p95' : 'p99';
    reset('H3-Image throughput per GPU versus ' + metricName + ' HTTP latency. Higher and further left is better.');
    text(x0, 12, 'images / GPU / hour', 't-head');
    for (let v = 0; v <= ymax; v += 400) {
      node('line', {x1: x0, x2: x1, y1: Y(v), y2: Y(v), class: 'm-rule'});
      text(x0 - 8, Y(v) + 4.5, fmt(v), 't-faint', {'text-anchor': 'end'});
    }
    const ticks = log ? [.5, 1, 2, 4, 8, 16, 32, 64, 128].filter(v => v <= xmax) : Array.from({length: 6}, (_, i) => i * xmax / 5);
    ticks.forEach(v => text(X(v), bottom + 20, fmt(v, v % 1 ? 1 : 0), 't-mut', {'text-anchor': 'middle'}));
    text((x0 + x1) / 2, bottom + 46, metricName + ' HTTP latency, seconds' + (log ? ' · log scale' : ''), 't-faint', {'text-anchor': 'middle'});
    const defs = node('defs'), clip = node('clipPath', {id: 'plot-clip'}, defs);
    node('rect', {x: x0 - 6, y: top - 10, width: x1 - x0 + 12, height: bottom - top + 20}, clip);
    const plot = node('g', {'clip-path': 'url(#plot-clip)'});
    const groups = new Map();
    data.forEach(p => { const k = seriesKey(p); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(p); });
    const labels = [];
    for (const group of groups.values()) {
      group.sort((a, b) => a.concurrency - b.concurrency);
      const p0 = group[0], stroke = color(p0);
      node('polyline', {points: group.map(p => [X(p[metric]), Y(p.images_per_gpu_hour)].join(',')).join(' '), class: 'ln', style: 'stroke:' + stroke, 'stroke-dasharray': dashed(p0), 'data-series': seriesKey(p0)}, plot);
      group.forEach(p => {
        if (p[metric] > xmax) return;
        const x = X(p[metric]), y = Y(p.images_per_gpu_hour);
        node('circle', {cx: x, cy: y, r: 4.5, class: 'pt', style: 'fill:' + stroke}, plot);
        if (groups.size <= 4 && (!compact || p.concurrency === 1 || p.concurrency === 16)) text(x, y + (p.platform === 'Vera Rubin' ? -11 : 20), 'C' + p.concurrency, 't-mut', {'text-anchor': 'middle'});
        const hit = node('circle', {cx: x, cy: y, r: 11, class: 'hit', 'data-point': seriesKey(p) + '/C' + p.concurrency});
        tooltip(hit, [shortName(p) + ' · ' + p.precision + ' · C' + p.concurrency, fmt(p.images_per_gpu_hour, 1) + ' images/GPU/hour', metricName + ' HTTP latency: ' + fmt(p[metric], 3) + ' s', 'Sample p95: ' + fmt(p.p95_latency_s, 3) + ' s · 160 measured requests', fmt(p.images_per_hour, 1) + ' images/hour across four GPUs']);
      });
      if (!zoom && groups.size <= 4 && !compact) {
        const a = group[group.length - 2], b = group[group.length - 1];
        labels.push([(X(a[metric]) + X(b[metric])) / 2, (Y(a.images_per_gpu_hour) + Y(b.images_per_gpu_hour)) / 2 + (p0.platform === 'GB200' ? 43 : -34), shortName(p0)]);
      }
    }
    labels.forEach(([x, y, s]) => text(x, y, s, 't-lab-b', {'text-anchor': 'middle'}));
    legend([...groups.values()].map(g => ({color: color(g[0]), dash: dashed(g[0]), label: shortName(g[0]) + ' · ' + g[0].precision.replace(' DiT', '')})));
    $('note').textContent = 'Native 4096×4096 T2I; six steps; four GPUs; 20 short prompts (122–160 characters), eight seeds each; 160 requests per point after warmup. C = closed-loop client concurrency, not DiT batch size. HTTP latency includes queueing, generation, PNG encoding and response receipt over loopback; client decoding/validation/saving is excluded from latency and included in throughput. ' + 'All curves use MXFP8 DiT linears; SP4 curves share the same parallel layout. Peak configurations have different response latencies.';
    reportHeight();
  }
  let barMode = kind === 'hardware' ? 'latency' : 'revenue';
  function drawBars() {
    const latency = barMode === 'latency', capacity = barMode === 'capacity', throughput = barMode === 'throughput';
    const compact = typeof innerWidth === 'number' && innerWidth < 560;
    svg.setAttribute('viewBox', compact ? '0 0 400 410' : '0 0 736 364');
    if ($('figure-heading')) $('figure-heading').textContent = latency ? 'Latency' : capacity || throughput ? 'Throughput' : 'Revenue';
    const data = platforms.map(platform => latency ? c1For(platform) : peakFor(platform));
    const value = p => latency ? p.median_latency_s : capacity ? p.images_per_hour : throughput ? p.images_per_gpu_hour : revenuePerGPU(p, .06879, 1);
    const vmax = latency ? 2.5 : capacity ? 8000 : throughput ? 2000 : 120;
    const step = latency ? .5 : capacity ? 2000 : throughput ? 500 : 30;
    const unit = latency ? 'median HTTP latency, seconds · lower is better' : capacity ? 'images / hour / four-GPU system' : throughput ? 'images / GPU / hour · higher is better' : 'gross USD / GPU / hour';
    const x0 = compact ? 8 : 136, x1 = compact ? 302 : 656, top = compact ? 64 : 44, bottom = compact ? 370 : 300;
    const X = v => x0 + v / vmax * (x1 - x0);
    const rubin = data[0], gb200 = data[2], gb300 = data[1];
    $('title').textContent = latency ? 'Native 4K in ' + fmt(value(rubin), 3) + ' seconds' : capacity ? fmt(value(rubin)) + ' images/hour from four Rubin GPUs' : throughput ? 'Rubin: ' + fmt(value(rubin) / value(gb200), 2) + '× GB200 throughput' : money(value(rubin)) + ' per GPU-hour';
    $('subtitle').textContent = latency || throughput ? (latency ? 'C1 on SP4: ' + fmt(value(gb200) / value(rubin), 2) + '× faster than GB200 and ' + fmt(value(gb300) / value(rubin), 2) + '× faster than GB300.' : 'Best measured serving configurations: ' + fmt(value(rubin) / value(gb300), 2) + '× GB300 peak throughput. Peaks have different response times.') : capacity ? 'Peak measured throughput. Four GPUs per system.' : 'Gross revenue at $0.06879/image and 100% billable utilization.';
    reset($('title').textContent + '. ' + unit);
    text(x0, 15, unit, compact ? 't-mut' : 't-head');
    for (let v = 0; v <= vmax; v += step) {
      node('line', {x1: X(v), x2: X(v), y1: top - 5, y2: bottom, class: 'm-rule'});
      text(X(v), bottom + 22, capacity ? (v / 1000) + 'K' : latency ? fmt(v, v % 1 ? 1 : 0) : fmt(v), 't-faint', {'text-anchor': 'middle'});
    }
    data.forEach((p, i) => {
      const y = (compact ? 64 : 54) + i * (compact ? 110 : 80), v = value(p), c = platformColor(p);
      text(compact ? x0 : x0 - 14, compact ? y - 10 : y + 20, p.platform === 'Vera Rubin' ? 'Vera Rubin' : p.platform, 't-lab-b', {'text-anchor': compact ? 'start' : 'end'});
      const bar = node('rect', {x: x0, y, width: X(v) - x0, height: 32, style: 'fill:' + c + ';stroke:none', 'data-bar': p.platform});
      text(X(v) + 9, y + 21, latency ? fmt(v, 3) + ' s' : throughput ? fmt(v, 1) : capacity ? fmt(v) : money(v), 't-num');
      text(x0, y + 51, p.topology + ' · C' + p.concurrency + ' · ' + p.precision + (latency || compact ? '' : ' · p50 ' + fmt(p.median_latency_s, 3) + ' s'), 't-mut');
      if (compact && !latency) text(x0, y + 68, 'p50 ' + fmt(p.median_latency_s, 3) + ' s', 't-mut');
      tooltip(bar, [p.platform + ' · ' + p.topology + ' · ' + p.precision, unit + ': ' + fmt(v, latency ? 3 : 2), 'p50 ' + fmt(p.median_latency_s, 3) + ' s · p95 ' + fmt(p.p95_latency_s, 3) + ' s', fmt(p.images_per_hour, 1) + ' images/hour across four GPUs']);
    });
    $('note').textContent = latency ? 'All three C1 runs use SP4 and MXFP8 DiT linears. Native 4096×4096; six steps; 160 short-prompt requests per point (20 prompts × eight seeds). HTTP latency includes PNG encoding and the full base64 response.' : capacity || throughput ? 'Four GPUs per system. Best measured throughput: Rubin SP1×DP4 at C8; GB300 and GB200 SP4 at C2. All use MXFP8. Peak configurations have different response latencies, shown under each bar.' : 'MiniMax Standard 4K API example: $0.06879/image before launch discounts. Gross revenue assumes 100% billable utilization, before operating costs. At 75%: ' + money(revenuePerGPU(rubin, .06879, .75)) + '/GPU/hour gross.';
    $('legend').replaceChildren();
    reportHeight();
  }
  if (kind === 'curve') {
    ['metric', 'precision'].forEach(id => $(id).addEventListener('change', drawCurve));
    drawCurve();
  } else {
    document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
      barMode = button.dataset.mode; document.querySelectorAll('[data-mode]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); drawBars();
    })); drawBars();
  }
  if ($('measurement-details')) $('measurement-details').addEventListener('toggle', reportHeight);
  window.H3 = {points, selectedPoints, revenuePerGPU};
  window.addEventListener('resize', kind === 'curve' ? drawCurve : drawBars);
  document.body.dataset.ready = '1';
  document.fonts.ready.then(reportHeight);
})();
