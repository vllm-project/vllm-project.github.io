(async function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const mode = document.body.dataset.workload;
  const reportHeight = () => window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  let data;
  try { const response=await fetch('workload-data.json'); if(!response.ok)throw new Error('HTTP '+response.status);data=await response.json(); }
  catch(error){$('note').textContent='Unable to load measured results. Please reload.';document.body.dataset.error=error.message;return;}
  const titles={'farm-to-table':'Food photography · text-to-image','podcast-edit':'Podcast edit · one reference'};
  const shapes={'2048x2048':'2048 × 2048 · 4.2 MP','4096x4096':'4096 × 4096 · 16.8 MP','6272x2688':'6272 × 2688 · 16.9 MP · ultrawide','4480x4480':'4480 × 4480 · 20.1 MP'};
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  function options(id, pairs, selected){for(const [value,label] of pairs){const e=el('option',label);e.value=value;$(id).appendChild(e);}$(id).value=selected;}
  options('task',Object.entries(titles),'farm-to-table');
  const pick=(task,aspect)=>data.rows.filter(p=>p.case===task&&p.aspect===aspect);
  const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const palette={'GB200':'#4d7ba6','GB300':'#b5873a'};
  function chart(task,aspect){
    const selected=pick(task,aspect);if(selected.length!==8)throw new Error('Expected four areas per hardware');
    const max=Math.max(...selected.map(p=>p.max_s));const step=max<=2?0.5:max<=4?1:2;const ymax=Math.ceil(max/step)*step;
    const X=v=>70+(v/22)*570,Y=v=>300-(v/ymax)*244;
    let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 736 364" width="736" height="364" role="img" aria-label="'+escape(titles[task]+'; '+aspect+'; mean HTTP seconds, lower is better')+'">';
    svg+='<style>text{font-family:Inter,Arial,sans-serif;fill:#23201e;font-size:12px} .axis{font-family:JetBrains Mono,monospace;font-size:11px;fill:#605b56}</style>';
    svg+='<text x="70" y="22" class="axis">MEAN HTTP SECONDS · LOWER IS BETTER</text>';
    for(let v=0;v<=ymax+1e-9;v+=step){svg+='<line x1="70" x2="640" y1="'+Y(v)+'" y2="'+Y(v)+'" stroke="#e6e1da"/><text x="58" y="'+(Y(v)+4)+'" text-anchor="end">'+v.toFixed(step<1?1:0)+'</text>';}
    for(const v of [0,4,8,12,16,20])svg+='<text x="'+X(v)+'" y="322" text-anchor="middle">'+v+'</text>';
    svg+='<text x="355" y="350" class="axis" text-anchor="middle">GENERATED IMAGE AREA · MEGAPIXELS</text>';
    for(const hw of ['GB200','GB300']){
      const points=selected.filter(p=>p.hardware===hw).sort((a,b)=>a.actual_mp-b.actual_mp),color=palette[hw];
      svg+='<polyline points="'+points.map(p=>X(p.actual_mp)+','+Y(p.mean_s)).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="2.3"/>';
      for(const p of points){
        const x=X(p.actual_mp),y=Y(p.mean_s),title=hw+' · 4 GPUs / SP4 · '+p.width+'×'+p.height+' · mean '+p.mean_s.toFixed(3)+' s · range '+p.min_s.toFixed(3)+'–'+p.max_s.toFixed(3)+' s · n=3';
        svg+='<g tabindex="0" aria-label="'+escape(title)+'"><title>'+escape(title)+'</title><line x1="'+x+'" x2="'+x+'" y1="'+Y(p.min_s)+'" y2="'+Y(p.max_s)+'" stroke="'+color+'" stroke-width="1.3"/><circle cx="'+x+'" cy="'+y+'" r="4.5" fill="'+color+'" stroke="white" stroke-width="1.5"/></g>';
      }
      const last=points[points.length-1];svg+='<text x="'+(X(last.actual_mp)+12)+'" y="'+(Y(last.mean_s)+4)+'">'+hw+'</text>';
    }
    return svg+'</svg>';
  }
  function drawGallery(){
    const task=$('task').value,shape=$('canvas').value;
    const selected=data.rows.filter(p=>p.case===task&&p.width+'x'+p.height===shape);
    if(selected.length!==2||selected.some(p=>!p.preview))throw new Error('Missing paired preview');
    $('outputs').replaceChildren();
    for(const p of selected){
      const card=el('figure',undefined,'output-card'),head=el('div',undefined,'output-heading');
      head.append(el('span','4× '+p.hardware+' · SP4'),el('span',p.mean_s.toFixed(2)+' s mean'));
      const image=el('img');image.src=p.preview;image.alt=titles[task]+' generated on four '+p.hardware+' GPUs, seed 42';image.loading='lazy';image.addEventListener('load',reportHeight);
      const caption=el('figcaption','MXFP8 DiT · '+p.width+'×'+p.height+'\nThree seeds · range '+p.min_s.toFixed(3)+'–'+p.max_s.toFixed(3)+' s. Shown: seed 42 ('+p.gallery_request_s.toFixed(3)+' s).');
      card.append(head,image,caption);$('outputs').appendChild(card);
    }
    const c=data.cases.find(c=>c.id===task);$('prompt').textContent=c.prompt;
    $('references').replaceChildren();
    if(c.reference_assets.length){const a=el('a','View MiniMax reference image');a.href='cases/podcast-reference.png';a.target='_blank';a.rel='noopener';const img=el('img');img.src=a.href;img.alt='Original podcast scene before editing';img.addEventListener('load',reportHeight);a.appendChild(img);$('references').appendChild(a);}
    $('note').textContent='Same prompt, reference bytes and seeds on both machines. Six steps; resident weights. Timings are three-request arithmetic means after warmup. Images are unretouched, downscaled 512-pixel previews; compare composition here, not full-resolution detail.';
    reportHeight();
  }
  function drawChart(){
    $('chart').innerHTML=chart($('task').value,$('aspect').value);
    $('note').textContent='Four GPUs per system · Ulysses SP4 · DiT TP1 · encoder TP4 · MXFP8 DiT · six steps · no offload. Each point averages seeds 0, 42 and 1234 after warmup; whiskers show min–max. Hover or focus a point for exact dimensions. HTTP latency includes the complete base64 PNG response and excludes client decoding, validation and saving.';
    reportHeight();
  }
  if(mode==='gallery'){options('canvas',Object.entries(shapes),'4096x4096');$('canvas').addEventListener('change',drawGallery);$('prompt-details').addEventListener('toggle',reportHeight);$('task').addEventListener('change',drawGallery);drawGallery();}
  else {options('aspect',data.aspect_ratios.map(x=>[x,x]),'1:1');$('aspect').addEventListener('change',drawChart);$('task').addEventListener('change',drawChart);drawChart();}
  window.Workloads={chart,pick,rows:data.rows};document.body.dataset.ready='1';
  if(document.fonts)document.fonts.ready.then(reportHeight);
  new ResizeObserver(reportHeight).observe(document.querySelector('main'));
})();
