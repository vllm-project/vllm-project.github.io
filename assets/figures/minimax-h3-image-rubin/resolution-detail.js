(async function(){
  'use strict';
  const $=id=>document.getElementById(id);
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const height=()=>window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.documentElement.scrollHeight)},'*');
  let data;
  try{const r=await fetch('resolution-detail-data.json?v=511127ee3d52');if(!r.ok)throw new Error('HTTP '+r.status);data=await r.json();}
  catch(error){$('status').textContent='Unable to load image sources. Please reload.';document.body.dataset.error=error.message;return;}
  const state={scene:data.cases[0].id,left:'1K',right:'4K',zoom:1,x:.5,y:.5,split:50};
  const images=new Map();let cards=[],revision=0,motion=0,interaction=0;
  const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
  const selected=()=>data.cases.find(c=>c.id===state.scene);
  const source=()=>selected().outputs.find(o=>o.tier==='4K');
  const dimensions=tier=>{const s=source(),divisor={'1K':4,'2K':2,'4K':1}[tier];return {width:s.width/divisor,height:s.height/divisor};};
  const imageURL=o=>o.image+'?v='+o.image_sha256.slice(0,12);
  function loadImage(output){
    const url=imageURL(output);
    if(!images.has(url))images.set(url,new Promise((resolve,reject)=>{
      const im=new Image();im.decoding='async';
      const fail=()=>{images.delete(url);reject(new Error('Image failed to load'));};
      im.onload=()=>im.naturalWidth===output.width&&im.naturalHeight===output.height?resolve(im):fail();
      im.onerror=fail;im.src=url;
    }));return images.get(url);
  }
  function updateView(notify=true){
    const edge=.5/state.zoom;state.x=clamp(state.x,edge,1-edge);state.y=clamp(state.y,edge,1-edge);
    $('zoom').value=String(state.zoom);$('zoom-label').textContent=Number(state.zoom.toFixed(2))+'×';
    $('left-resolution').value=state.left;$('right-resolution').value=state.right;
    for(const card of cards){
      card.canvas.style.width=100*state.zoom+'%';card.canvas.style.left=50-100*state.x*state.zoom+'%';card.canvas.style.top=50-100*state.y*state.zoom+'%';
      card.node.style.clipPath=card.side==='left'?'inset(0 '+(100-state.split)+'% 0 0)':'none';card.viewport.dataset.zoomed=String(state.zoom>1);
    }
    for(const side of ['left','right']){const d=dimensions(state[side]);$(side+'-label').textContent=state[side]+' · '+d.width+' × '+d.height;}
    $('wipe-divider').style.left=state.split+'%';$('wipe-handle').setAttribute('aria-valuenow',String(Math.round(state.split)));if(notify)height();
  }
  function stopMotion(manual=false){if(motion)cancelAnimationFrame(motion);motion=0;document.body.dataset.animating='0';if(manual)interaction++;}
  function focusDetail(){
    stopMotion();
    const pair=[state.left,state.right].sort().join('/');
    const target={zoom:pair==='1K/2K'?6:pair==='2K/4K'?5:4,x:selected().detail_focus[0],y:selected().detail_focus[1]};
    if(typeof requestAnimationFrame!=='function'||(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches)){
      Object.assign(state,target);updateView();return;
    }
    const start={zoom:state.zoom,x:state.x,y:state.y},began=performance.now();document.body.dataset.animating='1';
    function frame(now){
      const progress=clamp((now-began)/700,0,1),eased=1-Math.pow(1-progress,3);
      for(const key of ['zoom','x','y'])state[key]=start[key]+(target[key]-start[key])*eased;
      updateView(false);
      if(progress<1)motion=requestAnimationFrame(frame);else{motion=0;document.body.dataset.animating='0';height();}
    }
    motion=requestAnimationFrame(frame);
  }
  function zoom(v){stopMotion(true);state.zoom=clamp(Number(v)||1,1,8);updateView();}
  function reset(){stopMotion(true);state.zoom=1;state.x=state.y=.5;updateView();}
  function bindViewport(viewport){
    let drag=null;
    viewport.addEventListener('click',e=>{if(state.zoom!==1)return;const r=viewport.getBoundingClientRect();state.x=(e.clientX-r.left)/r.width;state.y=(e.clientY-r.top)/r.height;zoom(4);});
    viewport.addEventListener('pointerdown',e=>{stopMotion(true);if(state.zoom===1||e.button!==0)return;const r=viewport.getBoundingClientRect();drag={id:e.pointerId,cx:e.clientX,cy:e.clientY,x:state.x,y:state.y,w:r.width,h:r.height};viewport.setPointerCapture(e.pointerId);viewport.dataset.dragging='true';e.preventDefault();});
    viewport.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;state.x=drag.x-(e.clientX-drag.cx)/drag.w/state.zoom;state.y=drag.y-(e.clientY-drag.cy)/drag.h/state.zoom;updateView();});
    for(const event of ['pointerup','pointercancel','lostpointercapture'])viewport.addEventListener(event,()=>{drag=null;viewport.dataset.dragging='false';});
    viewport.addEventListener('keydown',e=>{stopMotion(true);const delta=.12/state.zoom;
      if(e.key==='ArrowLeft')state.x-=delta;else if(e.key==='ArrowRight')state.x+=delta;else if(e.key==='ArrowUp')state.y-=delta;else if(e.key==='ArrowDown')state.y+=delta;
      else if(e.key==='+'||e.key==='=')state.zoom=clamp(state.zoom+1,1,8);else if(e.key==='-')state.zoom=clamp(state.zoom-1,1,8);
      else if(e.key==='0'||e.key==='Home'){reset();e.preventDefault();return;}else return;e.preventDefault();updateView();
    });
  }
  async function draw(){
    stopMotion();const current=++revision,scene=selected(),original=source(),focusVersion=interaction;document.body.dataset.imagesReady='0';
    cards=[];$('outputs').replaceChildren();$('status').textContent='Loading full-resolution image…';
    const ratio=original.width/original.height;
    $('comparison-stage').style.aspectRatio=original.width+' / '+original.height;$('comparison-stage').style.maxWidth=Math.round(660*ratio)+'px';$('pair-labels').style.maxWidth=Math.round(660*ratio)+'px';
    for(const side of ['left','right']){
      const tier=state[side],d=dimensions(tier),node=el('figure',undefined,'resolution-card');node.dataset.side=side;
      const viewport=el('div',undefined,'resolution-viewport');viewport.tabIndex=0;viewport.setAttribute('role','img');viewport.setAttribute('aria-label',scene.title+', '+side+' '+tier+'. Click to zoom; arrows pan; plus and minus zoom; 0 resets.');
      const canvas=el('canvas');canvas.width=d.width;canvas.height=d.height;canvas.setAttribute('aria-hidden','true');viewport.appendChild(canvas);node.appendChild(viewport);bindViewport(viewport);$('outputs').appendChild(node);cards.push({side,tier,node,viewport,canvas});
    }
    $('demo-title').textContent=scene.demo_title;
    $('detail-hint').textContent=scene.detail_hint;
    $('setup').textContent=scene.hardware+' · '+scene.configuration+'.';
    $('prompt').textContent=scene.prompt;$('originals').replaceChildren();
    for(const output of scene.outputs){const a=el('a',output.width+' × '+output.height+' · '+output.mean_s.toFixed(2)+' s mean');a.href=imageURL(output);a.target='_blank';a.rel='noopener';$('originals').appendChild(a);}
    $('reference').replaceChildren();if(scene.reference_count){const a=el('a','Reference image');a.href='cases/podcast-reference.png';a.target='_blank';a.rel='noopener';$('reference').appendChild(a);}
    updateView();const currentCards=cards;
    try{const image=await loadImage(original);if(current!==revision)return;
      for(const card of currentCards){const ctx=card.canvas.getContext('2d');if(!ctx)throw new Error('Canvas is unavailable');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(image,0,0,card.canvas.width,card.canvas.height);}
      $('status').textContent='';document.body.dataset.imagesReady='1';if(focusVersion===interaction)focusDetail();
    }catch(error){if(current!==revision)return;$('status').textContent='Image could not load. Re-select the scene to retry.';document.body.dataset.imagesReady='0';}
    height();
  }
  for(const scene of data.cases){const option=el('option',scene.title);option.value=scene.id;$('scene').appendChild(option);}$('scene').value=state.scene;
  $('scene').addEventListener('change',()=>{state.scene=$('scene').value;images.clear();state.x=state.y=.5;state.zoom=1;draw();});
  for(const side of ['left','right'])$(side+'-resolution').addEventListener('change',()=>{state[side]=$(side+'-resolution').value;draw();});
  $('swap').addEventListener('click',()=>{[state.left,state.right]=[state.right,state.left];draw();});
  $('zoom').addEventListener('input',()=>zoom($('zoom').value));$('reset').addEventListener('click',reset);
  $('detail').addEventListener('click',focusDetail);
  let pointer=null;
  $('wipe-handle').addEventListener('pointerdown',e=>{stopMotion(true);if(e.button!==0)return;pointer=e.pointerId;$('wipe-handle').setPointerCapture(e.pointerId);e.preventDefault();});
  $('wipe-handle').addEventListener('pointermove',e=>{if(pointer!==e.pointerId)return;const r=$('comparison-stage').getBoundingClientRect();state.split=clamp(100*(e.clientX-r.left)/r.width,0,100);updateView();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])$('wipe-handle').addEventListener(event,()=>{pointer=null;});
  $('wipe-handle').addEventListener('keydown',e=>{stopMotion(true);if(e.key==='ArrowLeft')state.split-=5;else if(e.key==='ArrowRight')state.split+=5;else if(e.key==='Home')state.split=0;else if(e.key==='End')state.split=100;else return;state.split=clamp(state.split,0,100);e.preventDefault();updateView();});
  $('source-details').addEventListener('toggle',height);
  window.ResolutionDetail={state,data,draw,reset,zoom,focusDetail,get cards(){return cards;}};document.body.dataset.ready='1';draw();
  if(document.fonts)document.fonts.ready.then(height);new ResizeObserver(height).observe(document.querySelector('main'));
})();
