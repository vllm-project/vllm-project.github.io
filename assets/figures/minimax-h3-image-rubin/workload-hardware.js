(async function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const height = () => window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  const params = new URLSearchParams(location.search);
  let data;
  // Read the downloadable CSV directly, so timings have one published source.
  function parseCSV(source){
    const records=[];let row=[],field='',quoted=false;
    for(let i=0;i<=source.length;i++){
      const ch=i===source.length?'\n':source[i];
      if(quoted){
        if(i===source.length)throw new Error('Unclosed CSV quote');
        if(ch==='"'&&source[i+1]==='"'){field+='"';i++;}
        else if(ch==='"')quoted=false;
        else field+=ch;
      }else if(ch==='"'){if(field)throw new Error('Invalid CSV quote');quoted=true;}
      else if(ch===','){row.push(field);field='';}
      else if(ch==='\n'){row.push(field);if(row.some(v=>v!==''))records.push(row);row=[];field='';}
      else if(ch!=='\r')field+=ch;
    }
    const keys=records.shift();if(!keys||new Set(keys).size!==keys.length)throw new Error('Invalid CSV header');
    return records.map(values=>{if(values.length!==keys.length)throw new Error('Invalid CSV row');return Object.fromEntries(keys.map((k,i)=>[k,values[i]]));});
  }
  try {
    const responses=await Promise.all([fetch('api-workload-data.json?v=1f71a5771d53'),fetch('api-workload-results.csv?v=2b927b91c91f')]);
    for(const response of responses)if(!response.ok)throw new Error('HTTP '+response.status);
    const [metadata,csv]=await Promise.all([responses[0].json(),responses[1].text()]);
    data={...metadata,cases:Object.fromEntries(Object.entries(metadata.cases).map(([id,c])=>[id,{
      ...metadata.prompts[c.prompt_id],id,workload:id.slice(id.indexOf('/')+1),width:c.width,height:c.height,
      generation_canvas:c.generation_canvas,model_revision:metadata.model_revision,encoder_revision:metadata.encoder_revision
    }])),rows:parseCSV(csv).map((r,i)=>({
      id:i+1,case_id:r.corpus+'/'+r.workload,hardware:r.hardware,precision:r.precision,
      gpus:Number(r.gpus),sp:Number(r.sp),dit_tp:Number(r.dit_tp),encoder_tp:Number(r.encoder_tp),
      offload:r.cpu_offload_components==='none'?[]:r.cpu_offload_components.split(','),
      eager:r.eager==='True',runtime_flags:JSON.parse(r.runtime_flags),status:r.status,n:Number(r.n),
      samples_s:Object.fromEntries([0,42,1234].filter(s=>r['seed_'+s+'_s']!=='').map(s=>[s,Number(r['seed_'+s+'_s'])])),
      mean_s:r.mean_s===''?null:Number(r.mean_s),min_s:r.min_s===''?null:Number(r.min_s),max_s:r.max_s===''?null:Number(r.max_s),
      historical_samples_over_600s:Number(r.historical_samples_over_600s),prior_failed_requests:Number(r.prior_failed_requests),profile_id:r.profile_id,first_sample_unix_s:Number(r.first_sample_unix_s)
    }))};
    delete data.prompts;
  } catch(error){$('note').textContent='Unable to load measurements. Please reload.';document.body.dataset.error=error.message;return;}
  function options(id, items, selected){$(id).replaceChildren();for(const x of items){const o=el('option',x.label);o.value=x.id;$(id).appendChild(o);}$(id).value=selected;}
  const hardware=Object.fromEntries(data.hardware.map(h=>[h.id,h]));
  const tones={gb200:'blue',gb300:'gold',rubin:'green',h100:'blue',h200:'gold',h200_single:'green',a100:'stone',a100_sxm:'blue',rtx5090:'blue',rtx5080:'gold',rtx6000d:'green',rtxpro6000:'stone'};
  function choose(caseId, hw){
    const attempts=data.rows.filter(r=>r.case_id===caseId&&r.hardware===hw&&r.precision===hardware[hw].precision);
    const complete=attempts.filter(r=>r.status==='completed'&&r.n===3);
    const baseline=complete.filter(r=>Object.keys(r.runtime_flags).length===0);
    const oldest=(a,b)=>a.first_sample_unix_s-b.first_sample_unix_s||a.profile_id.localeCompare(b.profile_id);
    baseline.sort(oldest);complete.sort(oldest);
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
    if(r.status==='completed'&&r.n===3)return r.mean_s.toFixed(2)+' s';
    if(r.status==='stopped_at_request_limit')return r.historical_samples_over_600s?'Over 10 min':'Stopped at 10 min';
    throw new Error('Cannot display a result without a measured value or a verified limit');
  }
  function limitReason(r){
    return r.historical_samples_over_600s
      ?'Earlier measurements exceeded the 10-minute cutoff; excluded from latency bars.'
      :'Generation stopped at the 10-minute limit; no completed image.';
  }
  function detailsText(r){
    const value=r.n?'n='+r.n+'; mean '+r.mean_s.toFixed(3)+' s; range '+r.min_s.toFixed(3)+'–'+r.max_s.toFixed(3)+' s; seeds '+Object.entries(r.samples_s).map(([s,v])=>s+': '+v.toFixed(3)+' s').join(', '):'No measured samples';
    return r.gpus+' GPU(s), SP'+r.sp+', DiT TP'+r.dit_tp+', encoder TP'+r.encoder_tp+', '+r.precision.toUpperCase()+' DiT. '+placement(r)+'. VAE '+(r.eager?'eager':'CUDA Graph')+'. '+value+'. '+(flags(r)||'Default runtime settings')+'.';
  }
  function axisElement(axis){
    const node=el('div',undefined,'hardware-axis');node.setAttribute('aria-hidden','true');
    const count=Math.round(axis.max/axis.step);
    for(let i=0;i<=count;i++){const tick=el('span',String(Number((i*axis.step).toFixed(3))),i===0?'first':i===count?'last':'');tick.style.left=(100*i/count)+'%';node.appendChild(tick);}
    return node;
  }
  function matching(){
    return Object.keys(data.cases).filter(id=>{
      const c=data.cases[id];return String(c.reference_count)===$('references').value&&c.width+'x'+c.height===$('resolution').value;
    });
  }
  function sortResults(results){
    const order=$('sort').value;
    if(order==='hardware')return results;
    return [...results].sort((a,b)=>{
      const ac=a.primary.status==='completed'&&a.primary.n===3,bc=b.primary.status==='completed'&&b.primary.n===3;
      if(ac!==bc)return ac?-1:1;
      if(!ac)return 0;
      return (order==='latency-desc'?-1:1)*(a.primary.mean_s-b.primary.mean_s);
    });
  }
  function draw(){
    const group=data.groups.find(g=>g.id===$('group').value),matches=matching();
    const ids=matches;
    const panels=ids.map(id=>({case:data.cases[id],results:sortResults(group.hardware.filter(hw=>!hardware[hw].workloads||hardware[hw].workloads.includes(id)).map(hw=>choose(id,hw)).filter(r=>r.primary&&((r.primary.status==='completed'&&r.primary.n===3)||r.primary.status==='stopped_at_request_limit')))}));
    const values=panels.flatMap(p=>p.results.filter(r=>r.primary&&r.primary.status==='completed'&&r.primary.n===3).map(r=>r.primary.max_s));
    const axis=axisLimit(values.length?Math.max(...values):0);
    $('panels').replaceChildren();
    for(const panel of panels){
      const c=panel.case,card=el('section',undefined,'workload-panel');card.dataset.caseId=c.id;
      card.append(el('h2',({1024:'1K',2048:'2K',4096:'4K'}[c.width]||'')+' · '+c.width+' × '+c.height,'workload-panel-title'),el('p',c.reference_count+' reference image'+(c.reference_count===1?'':'s')+' · 1 image per request · 6 sampling steps','workload-panel-meta'));
      if(c.generation_canvas[0]!==c.width||c.generation_canvas[1]!==c.height)card.appendChild(el('p','Generated at '+c.generation_canvas.join('×')+', then resized','workload-panel-meta'));
      const list=el('div',undefined,'hardware-bars');list.setAttribute('role','list');
      for(const result of panel.results){
        const p=result.primary,h=hardware[result.hardware],row=el('div',undefined,'workload-bar-row');row.setAttribute('role','listitem');row.dataset.hardware=h.id;
        const head=el('div',undefined,'hardware-heading');head.append(el('span',h.gpus+'× '+h.label),el('strong',status(p),'tnum'));row.appendChild(head);
        if(p&&p.status==='completed'&&p.n===3){
          const track=el('div',undefined,'hardware-track');track.setAttribute('aria-hidden','true');
          const bar=el('span',undefined,'hardware-bar tone-'+tones[h.id]);bar.style.width=(100*p.mean_s/axis.max)+'%';
          const range=el('span',undefined,'hardware-range');range.style.left=(100*p.min_s/axis.max)+'%';range.style.width=(100*(p.max_s-p.min_s)/axis.max)+'%';track.append(bar,range);row.appendChild(track);
          if(p.historical_samples_over_600s)row.appendChild(el('p','Measured before the 10-minute limit','workload-bar-flag'));
        }else{
          row.className+=' workload-unmeasured';
          row.appendChild(el('p',limitReason(p),'workload-bar-flag'));
        }
        list.appendChild(row);
      }
      card.append(list,axisElement(axis));
      const detail=el('details',undefined,'workload-measurements');detail.appendChild(el('summary','Test setup'));detail.addEventListener('toggle',height);
      for(const result of panel.results){detail.appendChild(el('h3',hardware[result.hardware].label));
        detail.appendChild(el('p',result.primary?detailsText(result.primary):'No completed measurements.'));
        if(result.primary.status==='stopped_at_request_limit')detail.appendChild(el('p',limitReason(result.primary)));
      }
      detail.appendChild(el('p','Bars show completed three-seed measurements within the 10-minute cutoff. Linear axis starts at zero. Mean of seeds 0, 42 and 1234 after warmup; whiskers show min–max. Timing completion is not a semantic-quality score.'));
      detail.appendChild(el('p','HTTP latency includes text encoding, generation, PNG encoding and the complete response. Prompt enhancement and client saving are excluded. MXFP8 applies to DiT linears; H100/H200/A100 use BF16. Each hardware comparison uses identical inputs.'));
      if(c.reference_count>1)detail.appendChild(el('p','Reference-composition results measure execution time and capacity. Some outputs miss or duplicate requested objects.'));
      card.append(detail);$('panels').appendChild(card);
    }
    $('note').textContent='Mean of three requests; whiskers show min–max.';
    $('quality-note').textContent='';
    window.HardwareWorkloads.selection={mode:$('mode').value,sort:$('sort').value,references:$('references').value,resolution:$('resolution').value,group:group.id,panels,axis};
    height();
  }
  function updateReferences(reference=$('references').value){
    const cases=Object.values(data.cases).filter(c=>modeMatches(c)&&c.width+'x'+c.height===$('resolution').value);
    const counts=[...new Set(cases.map(c=>c.reference_count))].sort((a,b)=>a-b);
    const selected=counts.some(n=>String(n)===reference)?reference:String(counts[0]);
    options('references',counts.map(n=>({id:String(n),label:n===0?'None':String(n)})),selected);
    $('reference-control').hidden=$('mode').value!=='multi-edit';
    draw();
  }
  function modeMatches(c){return $('mode').value==='t2i'?c.reference_count===0:$('mode').value==='single-edit'?c.reference_count===1:c.reference_count>1;}
  function updateSizes(reference=$('references').value){
    const sizes=new Map();for(const c of Object.values(data.cases).filter(modeMatches))sizes.set(c.width+'x'+c.height,c);
    const previous=$('resolution').value||params.get('resolution')||'4096x4096';
    options('resolution',[...sizes].sort((a,b)=>a[1].width-b[1].width).map(([id,c])=>({id,label:({1024:'1K',2048:'2K',4096:'4K'}[c.width]||'')+' · '+c.width+' × '+c.height})),sizes.has(previous)?previous:'4096x4096');
    updateReferences(reference);
  }
  window.HardwareWorkloads={data,choose,axisLimit,parseCSV};
  const requestedRefs=params.get('references'),requestedMode=params.get('mode');
  const initialMode=Number(requestedRefs)>1||requestedMode==='multi-edit'?'multi-edit':requestedRefs==='1'||requestedMode==='image-edit'||requestedMode==='single-edit'?'single-edit':'t2i';
  options('mode',[{id:'t2i',label:'Text-to-image'},{id:'single-edit',label:'Image edit · 1 reference'},{id:'multi-edit',label:'Image edit · multiple references'}],initialMode);
  const sorts=[{id:'latency-desc',label:'Latency: high to low'},{id:'latency-asc',label:'Latency: low to high'},{id:'hardware',label:'Hardware order'}];
  options('sort',sorts,sorts.some(s=>s.id===params.get('sort'))?params.get('sort'):'latency-desc');
  const groups=[data.groups.find(g=>g.id==='all'),...data.groups.filter(g=>g.id!=='all')];
  const group=params.get('group')||'all';
  options('group',groups.map(g=>({...g,label:g.id==='all'?'All tested hardware':g.label})),groups.some(g=>g.id===group)?group:'all');
  $('references').addEventListener('change',draw);
  $('resolution').addEventListener('change',()=>updateReferences());$('group').addEventListener('change',draw);$('sort').addEventListener('change',draw);
  $('mode').addEventListener('change',()=>updateSizes());
  updateSizes(requestedRefs||'0');
  document.body.dataset.ready='1';
  if(document.fonts)document.fonts.ready.then(height);
  new ResizeObserver(height).observe(document.querySelector('main'));
})();
