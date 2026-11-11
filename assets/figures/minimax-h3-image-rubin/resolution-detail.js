(async function () {
  'use strict';
  const $=id=>document.getElementById(id);
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const height=()=>window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  let data;
  try{const response=await fetch('resolution-detail-data.json?v=cd5a0203dcf8');if(!response.ok)throw new Error('HTTP '+response.status);data=await response.json();}
  catch(error){$('status').textContent='Unable to load image sources. Please reload.';document.body.dataset.error=error.message;return;}
  const state={scene:data.cases[0].id,mode:'same-image',zoom:1,x:0.5,y:0.5,active:'4K',view:'wipe',lower:'1K',split:50};
  const images=new Map();let cards=[],revision=0;
  function loadImage(output){
    const url=output.image+'?v='+output.image_sha256.slice(0,12);
    if(!images.has(url))images.set(url,new Promise((resolve,reject)=>{
      const image=new Image();image.decoding='async';
      image.onload=()=>image.naturalWidth===output.width&&image.naturalHeight===output.height?resolve(image):reject(new Error('Unexpected image size'));
      image.onerror=()=>{images.delete(url);reject(new Error('Image failed to load'));};image.src=url;
    }));
    return images.get(url);
  }
  const clamp=(v,lo,hi)=>Math.min(hi,Math.max(lo,v));
  function updateView(){
    const edge=0.5/state.zoom;state.x=clamp(state.x,edge,1-edge);state.y=clamp(state.y,edge,1-edge);
    $('zoom').value=String(state.zoom);$('zoom-label').textContent=Number(state.zoom.toFixed(2))+'×';
    for(const card of cards){
      card.canvas.style.width=(state.zoom*100)+'%';
      card.canvas.style.left=(50-state.x*state.zoom*100)+'%';
      card.canvas.style.top=(50-state.y*state.zoom*100)+'%';
      card.viewport.dataset.zoomed=String(state.zoom>1);
      card.node.dataset.active=String(card.output.tier===state.active);
      card.node.dataset.wipeLow=String(card.output.tier===state.lower);
      card.node.style.clipPath=state.view==='wipe'&&card.output.tier===state.lower?'inset(0 '+(100-state.split)+'% 0 0)':'none';
    }
    $('comparison-stage').dataset.view=state.view;
    $('pair-controls').hidden=state.view!=='wipe';$('resolution-tabs').hidden=state.view==='wipe';
    $('pair-labels').hidden=state.view!=='wipe';
    const pairLabel=tier=>{const output=cards.find(c=>c.output.tier===tier)?.output;return output?tier+' · '+output.width+' × '+output.height+(state.mode==='measured'?' · '+output.mean_s.toFixed(2)+' s mean':''):tier;};
    $('left-label').textContent=pairLabel(state.lower);$('right-label').textContent=pairLabel('4K');
    $('wipe-divider').style.left=state.split+'%';$('wipe-handle').setAttribute('aria-valuenow',String(Math.round(state.split)));
    $('pair-1k').setAttribute('aria-pressed',String(state.lower==='1K'));$('pair-2k').setAttribute('aria-pressed',String(state.lower==='2K'));
    for(const b of $('resolution-tabs').children)b.setAttribute('aria-pressed',String(b.dataset.tier===state.active));
    height();
  }
  function zoom(value){state.zoom=clamp(Number(value)||1,1,8);updateView();}
  function reset(){state.zoom=1;state.x=state.y=0.5;updateView();}
  function pointAt(event,viewport){const r=viewport.getBoundingClientRect();return{x:state.x+(event.clientX-r.left-r.width/2)/r.width/state.zoom,y:state.y+(event.clientY-r.top-r.height/2)/r.height/state.zoom};}
  function bindViewport(viewport){
    let drag=null;
    viewport.addEventListener('click',event=>{if(state.zoom!==1)return;const point=pointAt(event,viewport);state.x=point.x;state.y=point.y;zoom(4);});
    viewport.addEventListener('pointerdown',event=>{
      if(state.zoom===1||event.button!==0)return;
      const r=viewport.getBoundingClientRect();drag={pointer:event.pointerId,cx:event.clientX,cy:event.clientY,x:state.x,y:state.y,width:r.width,height:r.height};
      viewport.setPointerCapture(event.pointerId);viewport.dataset.dragging='true';event.preventDefault();
    });
    viewport.addEventListener('pointermove',event=>{
      if(!drag||drag.pointer!==event.pointerId)return;
      state.x=drag.x-(event.clientX-drag.cx)/drag.width/state.zoom;state.y=drag.y-(event.clientY-drag.cy)/drag.height/state.zoom;updateView();
    });
    const end=()=>{drag=null;viewport.dataset.dragging='false';};
    viewport.addEventListener('pointerup',end);viewport.addEventListener('pointercancel',end);viewport.addEventListener('lostpointercapture',end);
    viewport.addEventListener('keydown',event=>{
      const step=0.12/state.zoom;
      if(event.key==='ArrowLeft')state.x-=step;else if(event.key==='ArrowRight')state.x+=step;
      else if(event.key==='ArrowUp')state.y-=step;else if(event.key==='ArrowDown')state.y+=step;
      else if(event.key==='+'||event.key==='=')state.zoom=clamp(state.zoom+1,1,8);
      else if(event.key==='-')state.zoom=clamp(state.zoom-1,1,8);
      else if(event.key==='0'||event.key==='Home'){reset();event.preventDefault();return;}
      else return;event.preventDefault();updateView();
    });
  }
  async function draw(){
    const current=++revision,scene=data.cases.find(c=>c.id===state.scene);
    document.body.dataset.imagesReady='0';
    cards=[];$('outputs').replaceChildren();$('resolution-tabs').replaceChildren();
    $('comparison-note').textContent=state.mode==='same-image'
      ?'One 4K image, shown at 1K, 2K and 4K. The 1K and 2K versions are resized in the browser to keep the composition identical.'
      :'Separate API outputs with the same prompt, references and seed. 1K is generated at 2K and resized; 2K and 4K are native generations. Composition may change.';
    $('setup').textContent=data.hardware+' · '+data.configuration+'. Full-resolution, lossless source images.';
    $('prompt').textContent=scene.prompt;$('reference').replaceChildren();
    if(scene.reference_count){const link=el('a','Reference image');link.href='cases/podcast-reference.png';link.target='_blank';link.rel='noopener';const image=el('img');image.src=link.href;image.alt='Reference studio image before editing';image.loading='lazy';image.addEventListener('load',height);link.appendChild(image);$('reference').appendChild(link);}
    $('status').textContent='Loading full-resolution images…';
    for(const output of scene.outputs){
      const node=el('figure',undefined,'resolution-card'),heading=el('div',output.tier,'resolution-heading');node.dataset.tier=output.tier;
      heading.appendChild(el('span',output.width+' × '+output.height));node.appendChild(heading);
      const viewport=el('div',undefined,'resolution-viewport');viewport.tabIndex=0;viewport.setAttribute('role','img');
      viewport.setAttribute('aria-label',scene.title+', '+output.tier+'. Click to zoom; arrow keys pan; plus and minus zoom; 0 resets.');
      const canvas=el('canvas');canvas.width=output.width;canvas.height=output.height;canvas.setAttribute('aria-hidden','true');viewport.appendChild(canvas);bindViewport(viewport);node.appendChild(viewport);
      const description=state.mode==='same-image'?(output.tier==='4K'?'Original 4K output':'Resized from the same 4K output'):(output.mean_s.toFixed(2)+' s mean · '+(output.tier==='1K'?'2K generation → 1K output':'Native '+output.tier+' generation'));
      node.appendChild(el('figcaption',description));$('outputs').appendChild(node);cards.push({node,viewport,canvas,output});
      const tab=el('button',output.tier);tab.type='button';tab.dataset.tier=output.tier;tab.addEventListener('click',()=>{state.active=output.tier;updateView();});$('resolution-tabs').appendChild(tab);
    }
    updateView();
    const currentCards=cards,mode=state.mode;
    const results=await Promise.allSettled(currentCards.map(async card=>{
      const source=mode==='same-image'?scene.outputs.find(o=>o.tier==='4K'):card.output;
      const image=await loadImage(source);if(current!==revision)return;
      const context=card.canvas.getContext('2d');if(!context)throw new Error('Canvas is unavailable');
      context.imageSmoothingEnabled=true;context.imageSmoothingQuality='high';
      context.drawImage(image,0,0,card.output.width,card.output.height);card.node.dataset.loaded='true';
    }));
    if(current!==revision)return;
    results.forEach((r,i)=>{if(r.status==='rejected')currentCards[i].node.dataset.error='true';});
    const errors=results.filter(r=>r.status==='rejected').length;
    $('status').textContent=errors?'Some full-resolution images could not load. Re-select the comparison to retry.':'';
    document.body.dataset.imagesReady=errors?'0':'1';height();
  }
  for(const scene of data.cases){const option=el('option',scene.title);option.value=scene.id;$('scene').appendChild(option);}
  $('scene').value=state.scene;$('comparison').value=state.mode;
  $('view').value=state.view;
  $('scene').addEventListener('change',()=>{state.scene=$('scene').value;reset();draw();});
  $('comparison').addEventListener('change',()=>{state.mode=$('comparison').value;draw();});
  $('zoom').addEventListener('input',()=>zoom($('zoom').value));$('reset').addEventListener('click',reset);
  $('view').addEventListener('change',()=>{state.view=$('view').value;updateView();});
  $('pair-1k').addEventListener('click',()=>{state.lower='1K';updateView();});$('pair-2k').addEventListener('click',()=>{state.lower='2K';updateView();});
  let wipePointer=null;
  const splitAt=event=>{const r=$('comparison-stage').getBoundingClientRect();state.split=clamp(100*(event.clientX-r.left)/r.width,0,100);updateView();};
  $('wipe-handle').addEventListener('pointerdown',event=>{if(event.button!==0)return;wipePointer=event.pointerId;$('wipe-handle').setPointerCapture(event.pointerId);event.preventDefault();});
  $('wipe-handle').addEventListener('pointermove',event=>{if(wipePointer===event.pointerId)splitAt(event);});
  for(const name of ['pointerup','pointercancel','lostpointercapture'])$('wipe-handle').addEventListener(name,()=>{wipePointer=null;});
  $('wipe-handle').addEventListener('keydown',event=>{if(event.key==='ArrowLeft')state.split-=5;else if(event.key==='ArrowRight')state.split+=5;else if(event.key==='Home')state.split=0;else if(event.key==='End')state.split=100;else return;state.split=clamp(state.split,0,100);event.preventDefault();updateView();});
  $('prompt-details').addEventListener('toggle',height);
  window.ResolutionDetail={state,data,zoom,reset,draw,get cards(){return cards;}};
  document.body.dataset.ready='1';draw();
  if(document.fonts)document.fonts.ready.then(height);
  new ResizeObserver(height).observe(document.querySelector('main'));
})();
