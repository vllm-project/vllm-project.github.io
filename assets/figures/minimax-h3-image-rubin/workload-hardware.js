(async function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
  const height = () => window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  const params = new URLSearchParams(location.search);
  let data, page=0;
  const pageSize=6;
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
    const responses=await Promise.all([fetch('hardware-workload-data.json'),fetch('extended-workload-results.csv')]);
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
      historical_samples_over_600s:Number(r.historical_samples_over_600s),prior_failed_requests:Number(r.prior_failed_requests)
    }))};
    delete data.prompts;
  } catch(error){$('note').textContent='Unable to load measurements. Please reload.';document.body.dataset.error=error.message;return;}
  function options(id, items, selected){$(id).replaceChildren();for(const x of items){const o=el('option',x.label);o.value=x.id;$(id).appendChild(o);}$(id).value=selected;}
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
    if(r.status==='stopped_at_request_limit')return 'Exceeded 10 min';
    if(r.status==='partial')return 'Not completed';
    return 'Could not complete';
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
    return data.scenes.find(s=>s.id===$('scene').value).cases.filter(id=>{
      const c=data.cases[id];return ($('references').value==='all'||String(c.reference_count)===$('references').value)&&($('resolution').value==='all'||c.width+'x'+c.height===$('resolution').value);
    });
  }
  function draw(){
    const group=data.groups.find(g=>g.id===$('group').value),matches=matching();
    const pages=Math.ceil(matches.length/pageSize);page=Math.min(page,pages-1);
    const ids=matches.slice(page*pageSize,(page+1)*pageSize);
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
        row.appendChild(el('p',(p?'SP'+p.sp+' · '+p.precision.toUpperCase()+' DiT · '+placement(p):'Not measured for this scene and size'),'workload-bar-config'));
        if(p&&p.status==='completed'&&p.n===3){
          const track=el('div',undefined,'hardware-track');track.setAttribute('aria-hidden','true');
          const bar=el('span',undefined,'hardware-bar tone-'+tones[h.id]);bar.style.width=(100*p.mean_s/axis.max)+'%';
          const range=el('span',undefined,'hardware-range');range.style.left=(100*p.min_s/axis.max)+'%';range.style.width=(100*(p.max_s-p.min_s)/axis.max)+'%';track.append(bar,range);row.appendChild(track);
          if(Object.keys(p.runtime_flags).length)row.appendChild(el('p','Custom configuration · see setup','workload-bar-flag'));
          if(p.historical_samples_over_600s)row.appendChild(el('p','Measured before the 10-minute limit','workload-bar-flag'));
        }else row.className+=' workload-unmeasured';
        list.appendChild(row);
      }
      card.append(list,axisElement(axis));
      const detail=el('details',undefined,'workload-measurements');detail.appendChild(el('summary','Test setup'));detail.addEventListener('toggle',height);
      for(const result of panel.results){detail.appendChild(el('h3',hardware[result.hardware].label));
        detail.appendChild(el('p',result.primary?detailsText(result.primary):'Not measured for this scene and size.'));
      }
      const prompt=el('details',undefined,'workload-measurements');prompt.appendChild(el('summary','Exact prompt'));
      if(c.prompt)prompt.appendChild(el('pre',c.prompt,'workload-prompt'));
      else if(c.prompt_source){const a=el('a','Read the original prompt in the source guide');a.href=c.prompt_source;a.target='_blank';a.rel='noopener';prompt.appendChild(a);}
      prompt.addEventListener('toggle',height);
      detail.appendChild(el('p','Linear axis starts at zero. Mean of seeds 0, 42 and 1234 after warmup; whiskers show min–max. Timing completion is not a semantic-quality score.'));
      card.append(detail,prompt);$('panels').appendChild(card);
    }
    $('page-label').textContent=(page*pageSize+1)+'–'+Math.min((page+1)*pageSize,matches.length)+' of '+matches.length+' workloads';
    $('prev').disabled=page===0;$('next').disabled=page===pages-1;$('pagination').hidden=pages===1;
    $('note').textContent='Shared seconds scale. Mean of three seeds after warmup; whiskers show min–max. HTTP latency includes text encoding, generation, PNG encoding and the complete response. Prompt enhancement and client saving are excluded. MXFP8 applies to DiT linears; H100/H200/A100 use BF16. Each hardware comparison uses identical inputs.';
    $('quality-note').textContent=$('scene').value==='reference-composition'?'Reference-composition timings measure execution time and capacity. Some outputs miss or duplicate requested objects.':panels.some(p=>p.case.corpus==='openai_image25_examples')?'H3-Image results using public example prompts; these are not head-to-head GPT Image measurements.':'';
    window.HardwareWorkloads.selection={scene:$('scene').value,references:$('references').value,resolution:$('resolution').value,group:group.id,page,pages,panels,axis};
    height();
  }
  function updateFilters(reference=$('references').value,resolution=$('resolution').value){
    const cases=data.scenes.find(s=>s.id===$('scene').value).cases.map(id=>data.cases[id]);
    const counts=[...new Set(cases.map(c=>c.reference_count))].sort((a,b)=>a-b);
    options('references',[{id:'all',label:'All reference counts'},...counts.map(n=>({id:String(n),label:n===0?'None · text-to-image':n+' reference'+(n===1?'':'s')}))],counts.some(n=>String(n)===reference)?reference:'all');
    const sizes=new Map();for(const c of cases.filter(c=>$('references').value==='all'||String(c.reference_count)===$('references').value))sizes.set(c.width+'x'+c.height,c);
    options('resolution',[{id:'all',label:'All tested sizes'},...[...sizes].sort((a,b)=>a[1].width*a[1].height-b[1].width*b[1].height).map(([id,c])=>({id,label:c.width+' × '+c.height+' · '+(c.width*c.height/1e6).toFixed(1)+' MP'}))],sizes.has(resolution)?resolution:'all');
    page=0;draw();
  }
  window.HardwareWorkloads={data,choose,axisLimit,parseCSV};
  const initial=params.get('scene')||(params.get('suite')==='references'?'reference-composition':'featured');
  options('scene',data.scenes,data.scenes.some(s=>s.id===initial)?initial:'featured');
  options('group',data.groups,'blackwell');
  $('scene').addEventListener('change',()=>updateFilters());$('references').addEventListener('change',()=>updateFilters());
  $('resolution').addEventListener('change',()=>{page=0;draw();});$('group').addEventListener('change',draw);
  $('prev').addEventListener('click',()=>{if(page>0){page--;draw();}});
  $('next').addEventListener('click',()=>{if((page+1)*pageSize<matching().length){page++;draw();}});
  updateFilters(params.get('references')||'all',params.get('resolution')||(params.get('suite')==='references'?'4096x4096':'all'));
  document.body.dataset.ready='1';
  if(document.fonts)document.fonts.ready.then(height);
  new ResizeObserver(height).observe(document.querySelector('main'));
})();
