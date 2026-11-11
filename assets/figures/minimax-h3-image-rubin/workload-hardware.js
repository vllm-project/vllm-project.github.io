(async function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const height = () => window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  const params = new URLSearchParams(location.search);
  let data, page=0;
  const pageSize=6;
  try {
    const response=await fetch('hardware-workload-data.json');
    if(!response.ok)throw new Error('HTTP '+response.status);
    data=await response.json();
  } catch(error){$('note').textContent='Unable to load measurements. Please reload.';document.body.dataset.error=error.message;return;}
  function options(id, items, selected){for(const x of items){const o=el('option',x.label);o.value=x.id;$(id).appendChild(o);}$(id).value=selected;}
  const hardware=Object.fromEntries(data.hardware.map(h=>[h.id,h]));
  const tones={gb200:'blue',gb300:'gold',rubin:'green',h100:'blue',h200:'gold',a100:'stone',rtx5090:'blue',rtx5080:'gold',rtx6000d:'green',gb10:'stone'};
  function choose(caseId, hw){
    const attempts=data.rows.filter(r=>r.case_id===caseId&&r.hardware===hw&&r.precision===hardware[hw].precision);
    const complete=attempts.filter(r=>r.status==='completed'&&r.n===3);
    const baseline=complete.filter(r=>Object.keys(r.runtime_flags).length===0);
    if(baseline.length>1||!baseline.length&&complete.length>1)throw new Error('Ambiguous configuration; select an explicit recipe');
    const primary=baseline[0]||complete[0]||attempts.find(r=>r.status==='stopped_at_request_limit')||attempts.find(r=>r.status==='partial')||attempts[0]||null;
    return {hardware:hw,primary,attempts};
  }
  function axisLimit(max){
    if(!max)return {step:0.2,max:1};
    const raw=max/4,power=Math.pow(10,Math.floor(Math.log10(raw)));
    const step=[1,2,2.5,5,10].find(x=>x*power>=raw)*power;
    return {step,max:Math.ceil(max/step)*step};
  }
  const placement=r=>!r.offload.length?'Resident weights':'CPU offload: '+r.offload.map(x=>x==='dit'?'DiT':'encoder').join(' + ');
  const flags=r=>Object.entries(r.runtime_flags).map(([k,v])=>({
    PYTORCH_CUDA_ALLOC_CONF:'Expandable allocator',H3_IMAGE_VAE_PHASE_OFFLOAD:v==='1'?'VAE CPU offload':'VAE stays resident',
    H3_IMAGE_RELEASE_ATTN_BUFFERS:'Release attention buffers',H3_IMAGE_MLP_CHUNK_ROWS:'MLP chunks: '+v+' rows',
    H3_IMAGE_ENCODER_INPLACE_MLP:'Encoder buffer reuse',VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:'Worker timeout: '+v+' s'
  }[k]||k+'='+v)).join('; ');
  function status(r){
    if(!r)return 'Not measured';
    if(r.status==='completed'&&r.n===3)return r.mean_s.toFixed(2)+' s';
    if(r.status==='stopped_at_request_limit')return 'Stopped: >10 min';
    if(r.status==='partial')return 'Partial: '+r.n+'/3';
    return 'Failed';
  }
  function detailsText(r){
    const value=r.n?'n='+r.n+'; mean '+r.mean_s.toFixed(3)+' s; range '+r.min_s.toFixed(3)+'–'+r.max_s.toFixed(3)+' s; seeds '+Object.entries(r.samples_s).map(([s,v])=>s+': '+v.toFixed(3)+' s').join(', '):'No measured samples';
    return status(r)+' · '+r.gpus+' GPU(s), SP'+r.sp+', DiT TP'+r.dit_tp+', encoder TP'+r.encoder_tp+', '+r.precision.toUpperCase()+' DiT. '+placement(r)+'. VAE '+(r.eager?'eager':'CUDA Graph')+'. '+value+'. '+(flags(r)||'Baseline runtime settings')+'.'+(r.historical_samples_over_600s?' Historical measurement: request(s) exceeded 600 seconds before the cutoff policy.':'')+(r.prior_failed_requests?' Prior failed requests in this configuration: '+r.prior_failed_requests+'.':'');
  }
  function axisElement(axis){
    const node=el('div',undefined,'hardware-axis');node.setAttribute('aria-hidden','true');
    const count=Math.round(axis.max/axis.step);
    for(let i=0;i<=count;i++){const tick=el('span',String(Number((i*axis.step).toFixed(3))),i===0?'first':i===count?'last':'');tick.style.left=(100*i/count)+'%';node.appendChild(tick);}
    return node;
  }
  function draw(){
    const suite=data.suites.find(s=>s.id===$('suite').value),group=data.groups.find(g=>g.id===$('group').value);
    const pages=Math.ceil(suite.cases.length/pageSize);page=Math.min(page,pages-1);
    const ids=suite.cases.slice(page*pageSize,(page+1)*pageSize);
    const panels=ids.map(id=>({case:data.cases[id],results:group.hardware.map(hw=>choose(id,hw))}));
    const values=panels.flatMap(p=>p.results.filter(r=>r.primary&&r.primary.status==='completed'&&r.primary.n===3).map(r=>r.primary.max_s));
    const axis=axisLimit(values.length?Math.max(...values):0);
    $('panels').replaceChildren();
    for(const panel of panels){
      const c=panel.case,card=el('section',undefined,'workload-panel');card.dataset.caseId=c.id;
      card.append(el('h2',c.title,'workload-panel-title'),el('p',c.width+'×'+c.height+' · '+c.reference_count+' reference'+(c.reference_count===1?'':'s'),'workload-panel-meta'));
      if(c.generation_canvas[0]!==c.width||c.generation_canvas[1]!==c.height)card.appendChild(el('p','Generated at '+c.generation_canvas.join('×')+', then resized','workload-panel-meta'));
      const list=el('div',undefined,'hardware-bars');list.setAttribute('role','list');
      for(const result of panel.results){
        const p=result.primary,h=hardware[result.hardware],row=el('div',undefined,'workload-bar-row');row.setAttribute('role','listitem');row.dataset.hardware=h.id;
        const head=el('div',undefined,'hardware-heading');head.append(el('span',h.gpus+'× '+h.label),el('strong',status(p),'tnum'));row.appendChild(head);
        row.appendChild(el('p',(p?'SP'+p.sp+' · '+p.precision.toUpperCase()+' DiT · '+placement(p):'No matching record in this snapshot'),'workload-bar-config'));
        if(p&&p.status==='completed'&&p.n===3){
          const track=el('div',undefined,'hardware-track');track.setAttribute('aria-hidden','true');
          const bar=el('span',undefined,'hardware-bar tone-'+tones[h.id]);bar.style.width=(100*p.mean_s/axis.max)+'%';
          const range=el('span',undefined,'hardware-range');range.style.left=(100*p.min_s/axis.max)+'%';range.style.width=(100*(p.max_s-p.min_s)/axis.max)+'%';track.append(bar,range);row.appendChild(track);
          if(Object.keys(p.runtime_flags).length)row.appendChild(el('p','Adjusted configuration · see details','workload-bar-flag'));
          if(p.historical_samples_over_600s)row.appendChild(el('p','Historical result · exceeds 10-minute limit','workload-bar-flag'));
        }else row.className+=' workload-unmeasured';
        list.appendChild(row);
      }
      card.append(list,axisElement(axis));
      const detail=el('details',undefined,'workload-measurements');detail.appendChild(el('summary','Measurements and configurations'));detail.addEventListener('toggle',height);
      for(const result of panel.results){detail.appendChild(el('h3',hardware[result.hardware].label));
        if(!result.attempts.length)detail.appendChild(el('p','No matching measurement.'));
        for(const r of result.attempts)detail.appendChild(el('p',(r===result.primary?'Shown: ':'Other recorded configuration: ')+detailsText(r)));
      }
      const prompt=el('details',undefined,'workload-measurements');prompt.appendChild(el('summary','Exact prompt'));
      if(c.prompt)prompt.appendChild(el('pre',c.prompt,'workload-prompt'));
      else if(c.prompt_source){const a=el('a','Read the original prompt in the source guide');a.href=c.prompt_source;a.target='_blank';a.rel='noopener';prompt.appendChild(a);}
      prompt.addEventListener('toggle',height);
      detail.appendChild(el('p','Linear axis starts at zero. Mean of seeds 0, 42 and 1234 after warmup; whiskers show min–max. Timing completion is not a semantic-quality score.'));
      card.append(detail,prompt);$('panels').appendChild(card);
    }
    $('page-label').textContent=(page*pageSize+1)+'–'+Math.min((page+1)*pageSize,suite.cases.length)+' of '+suite.cases.length+' workloads';
    $('prev').disabled=page===0;$('next').disabled=page===pages-1;$('pagination').hidden=pages===1;
    $('note').textContent='Same seconds scale across the displayed panels. HTTP latency includes text encoding, generation, PNG encoding and the complete response; excludes prompt enhancement, warmup and client saving. Match inputs within each workload. MXFP8 denotes DiT linears; H100/H200/A100 use BF16. Failed, partial and cutoff records have no latency bar. Data snapshot: '+data.snapshot_utc.slice(0,16).replace('T',' ')+' UTC.';
    $('quality-note').textContent=(suite.id==='references'||suite.id==='museum_reference_extremes')?'Reference compositions measure execution time and capacity. Some outputs miss or duplicate requested objects; these timings do not certify reference retention.':suite.id==='openai_image25_examples'?'These are H3-Image runs of public example prompts, not a measured head-to-head comparison with GPT Image.':'';
    window.HardwareWorkloads.selection={suite:suite.id,group:group.id,page,pages,panels,axis};
    height();
  }
  window.HardwareWorkloads={data,choose,axisLimit};
  const initial=params.get('suite');options('suite',data.suites,data.suites.some(s=>s.id===initial)?initial:'creative');
  options('group',data.groups,'blackwell');
  $('suite').addEventListener('change',()=>{page=0;draw();});$('group').addEventListener('change',draw);
  $('prev').addEventListener('click',()=>{if(page>0){page--;draw();}});
  $('next').addEventListener('click',()=>{const n=data.suites.find(s=>s.id===$('suite').value).cases.length;if((page+1)*pageSize<n){page++;draw();}});
  draw();document.body.dataset.ready='1';
  if(document.fonts)document.fonts.ready.then(height);
  new ResizeObserver(height).observe(document.querySelector('main'));
})();
