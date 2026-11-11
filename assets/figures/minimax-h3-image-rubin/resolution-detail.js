(async function(){
  'use strict';
  const $=id=>document.getElementById(id);
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const height=()=>window.parent.postMessage({type:'vllm-embed-resize',height:Math.ceil(document.querySelector('main').getBoundingClientRect().height+8)},'*');
  let data;
  try{const r=await fetch('resolution-detail-data.json?v=b2ae6e0911ee');if(!r.ok)throw new Error('HTTP '+r.status);data=await r.json();}
  catch(error){$('status').textContent='Unable to load image sources. Please reload.';document.body.dataset.error=error.message;return;}
  const state={scene:data.cases[0].id,left:'1K',right:'4K',zoom:1,x:.5,y:.5,split:50};
  const images=new Map(),sceneButtons=[];let cards=[],revision=0,motion=0,interaction=0,pendingFocus=false;
  let inView=typeof IntersectionObserver!=='function';
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
  function warmNext(){
    if(!inView||document.body.dataset.imagesReady!=='1')return;
    const index=data.cases.findIndex(c=>c.id===state.scene),next=data.cases[(index+1)%data.cases.length].outputs.find(o=>o.tier==='4K');
    const keep=new Set([imageURL(source()),imageURL(next)]);
    for(const key of images.keys())if(!keep.has(key))images.delete(key);
    loadImage(next).catch(()=>{});
  }
  function sceneNavigation(){
    const index=data.cases.findIndex(c=>c.id===state.scene);
    sceneButtons.forEach((button,i)=>button.setAttribute('aria-pressed',String(i===index)));
  }
  function chooseScene(id){
    if(!data.cases.some(c=>c.id===id))return;
    if(state.scene===id&&document.body.dataset.imagesReady==='1')return;
    state.scene=id;state.x=state.y=.5;state.zoom=1;draw();
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
  function stopMotion(manual=false){if(motion)cancelAnimationFrame(motion);motion=0;document.body.dataset.animating='0';if(manual){interaction++;pendingFocus=false;}}
  function playFocus(){
    if(!pendingFocus||!inView||document.body.dataset.imagesReady!=='1')return;
    stopMotion();
    const pair=[state.left,state.right].sort().join('/');
    const target={zoom:pair==='1K/2K'?6:pair==='2K/4K'?5:4,x:selected().detail_focus[0],y:selected().detail_focus[1]};
    if(typeof requestAnimationFrame!=='function'||(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches)){
      Object.assign(state,target);pendingFocus=false;updateView();return;
    }
    state.zoom=1;state.x=state.y=.5;updateView(false);
    const start={zoom:state.zoom,x:state.x,y:state.y},began=performance.now();document.body.dataset.animating='1';
    function frame(now){
      const progress=clamp((now-began-180)/1100,0,1),eased=1-Math.pow(1-progress,3);
      for(const key of ['zoom','x','y'])state[key]=start[key]+(target[key]-start[key])*eased;
      updateView(false);
      if(progress<1)motion=requestAnimationFrame(frame);else{motion=0;pendingFocus=false;document.body.dataset.animating='0';height();}
    }
    motion=requestAnimationFrame(frame);
  }
  function focusDetail(){pendingFocus=true;playFocus();}
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
    stopMotion();pendingFocus=false;const current=++revision,scene=selected(),original=source(),focusVersion=interaction;document.body.dataset.imagesReady='0';
    $('comparison-stage').setAttribute('aria-busy','true');$('status').textContent='Loading '+scene.title.toLowerCase()+'…';sceneNavigation();
    const currentCards=[];
    for(const side of ['left','right']){
      const tier=state[side],d=dimensions(tier),node=el('figure',undefined,'resolution-card');node.dataset.side=side;
      const viewport=el('div',undefined,'resolution-viewport');viewport.tabIndex=0;viewport.setAttribute('role','img');viewport.setAttribute('aria-label',scene.title+', '+side+' '+tier+'. Click to zoom; arrows pan; plus and minus zoom; 0 resets.');
      const canvas=el('canvas');canvas.width=d.width;canvas.height=d.height;canvas.setAttribute('aria-hidden','true');viewport.appendChild(canvas);node.appendChild(viewport);bindViewport(viewport);currentCards.push({side,tier,node,viewport,canvas});
    }
    height();
    try{const image=await loadImage(original);if(current!==revision)return;
      for(const card of currentCards){const ctx=card.canvas.getContext('2d');if(!ctx)throw new Error('Canvas is unavailable');ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(image,0,0,card.canvas.width,card.canvas.height);}
      cards=currentCards;$('outputs').replaceChildren(...cards.map(c=>c.node));
      const ratio=original.width/original.height;
      $('comparison-stage').style.aspectRatio=original.width+' / '+original.height;$('comparison-stage').style.maxWidth=Math.round(660*ratio)+'px';$('pair-labels').style.maxWidth=Math.round(660*ratio)+'px';
      $('demo-title').textContent=scene.demo_title;
      $('detail-hint').textContent=scene.detail_hint;
      updateView();$('comparison-stage').setAttribute('aria-busy','false');
      if($('outputs').animate&&!(window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches))$('outputs').animate([{opacity:.5},{opacity:1}],{duration:180,easing:'ease-out'});
      $('status').textContent='';document.body.dataset.imagesReady='1';if(focusVersion===interaction)focusDetail();warmNext();
    }catch(error){if(current!==revision)return;$('status').textContent='Image could not load. Re-select the scene to retry.';document.body.dataset.imagesReady='0';}
    height();
  }
  for(const [index,scene] of data.cases.entries()){
    const button=el('button',undefined,'scene-choice');button.type='button';button.dataset.scene=scene.id;button.setAttribute('aria-controls','comparison-stage');
    const thumbnail=el('img');thumbnail.src=scene.thumbnail;thumbnail.alt='';thumbnail.width=240;thumbnail.height=120;thumbnail.decoding='async';
    button.appendChild(thumbnail);button.appendChild(el('span',scene.title));
    button.addEventListener('click',()=>chooseScene(scene.id));
    button.addEventListener('keydown',e=>{let next=index;if(e.key==='ArrowRight')next=(index+1)%data.cases.length;else if(e.key==='ArrowLeft')next=(index+data.cases.length-1)%data.cases.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=data.cases.length-1;else return;e.preventDefault();sceneButtons[next].focus();chooseScene(data.cases[next].id);});
    sceneButtons.push(button);$('scenes').appendChild(button);
  }
  for(const side of ['left','right'])$(side+'-resolution').addEventListener('change',()=>{state[side]=$(side+'-resolution').value;draw();});
  $('swap').addEventListener('click',()=>{[state.left,state.right]=[state.right,state.left];draw();});
  $('zoom').addEventListener('input',()=>zoom($('zoom').value));$('reset').addEventListener('click',reset);
  $('detail').addEventListener('click',focusDetail);
  let pointer=null;
  $('wipe-handle').addEventListener('pointerdown',e=>{stopMotion(true);if(e.button!==0)return;pointer=e.pointerId;$('wipe-handle').setPointerCapture(e.pointerId);e.preventDefault();});
  $('wipe-handle').addEventListener('pointermove',e=>{if(pointer!==e.pointerId)return;const r=$('comparison-stage').getBoundingClientRect();state.split=clamp(100*(e.clientX-r.left)/r.width,0,100);updateView();});
  for(const event of ['pointerup','pointercancel','lostpointercapture'])$('wipe-handle').addEventListener(event,()=>{pointer=null;});
  $('wipe-handle').addEventListener('keydown',e=>{stopMotion(true);if(e.key==='ArrowLeft')state.split-=5;else if(e.key==='ArrowRight')state.split+=5;else if(e.key==='Home')state.split=0;else if(e.key==='End')state.split=100;else return;state.split=clamp(state.split,0,100);e.preventDefault();updateView();});
  if(typeof IntersectionObserver==='function')new IntersectionObserver(entries=>{
    inView=entries[0].isIntersecting&&entries[0].intersectionRatio>=.3;
    document.body.dataset.inView=String(inView);
    if(inView){if(!motion)playFocus();warmNext();}else stopMotion();
  },{threshold:[0,.3]}).observe($('comparison-stage'));
  window.ResolutionDetail={state,data,draw,reset,zoom,focusDetail,chooseScene,get cards(){return cards;}};document.body.dataset.ready='1';draw();
  if(document.fonts)document.fonts.ready.then(height);new ResizeObserver(height).observe(document.querySelector('main'));
})();
