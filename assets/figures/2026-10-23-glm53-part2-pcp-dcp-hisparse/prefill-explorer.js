// Interactive Figure 3 for the GLM-5.3 Part 2 post: one prefill layer on 8 GPUs under DP8, TP8, PCP8 and PCP8 x DCP8.
(function(){
var root=document.getElementById("ppx");if(!root)return;
if(!document.querySelector("style[data-ppx]")){var st=document.createElement("style");st.setAttribute("data-ppx","");st.textContent=String.raw`@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap");
.ppx{--bg:#ffffff;--fg:#17202b;--muted:#5b6878;--faint:#8a96a5;--line:#dde3ea;--cell:#eef2f6;--kv:#13897a;--new:#4f8fd6;--q:#5b4fd1;--tmp:#e08a4c;--tmp-soft:#f7d9c3;--wire:#d4762f;--wire-soft:#fdf0e5;--cur:#efedfc;--dot:#1d1640;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--fg);font-size:15px;line-height:1.55;margin:28px 0 8px;display:grid;gap:14px}
@media (prefers-color-scheme:dark){.ppx{--bg:#161d25;--fg:#e6ebf1;--muted:#9aa7b6;--faint:#6c7887;--line:#2a3440;--cell:#1f2832;--kv:#2bb5a2;--new:#5d9fe2;--q:#b3a9ff;--tmp:#f0a06a;--tmp-soft:#5a3a24;--wire:#f0a06a;--wire-soft:#2d2119;--cur:#252244;--dot:#f4f1ff}}
.ppx *{box-sizing:border-box}
.ppx .ppx-modes{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.ppx .ppx-mode{all:unset;cursor:pointer;background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px;display:grid;gap:2px}
.ppx .ppx-mode:hover{border-color:var(--faint)}
.ppx .ppx-mode:focus-visible,.ppx .ppx-pill:focus-visible,.ppx .ppx-btn:focus-visible{outline:2px solid var(--q);outline-offset:2px}
.ppx .ppx-mode b{font-weight:700;font-size:15px;line-height:1.2}
.ppx .ppx-mode span{font-size:12.5px;color:var(--muted)}
.ppx .ppx-mode[aria-pressed="true"]{border-color:var(--q);box-shadow:0 0 0 1px var(--q) inset}
.ppx .ppx-panel{background:var(--bg);border:1px solid var(--line);border-radius:14px;padding:18px;display:grid;gap:16px;min-width:0}
.ppx .ppx-timeline{display:flex;gap:3px;flex-wrap:wrap}
.ppx .ppx-pill{all:unset;cursor:pointer;display:flex;align-items:center;gap:5px;padding:3px 8px 3px 3px;border-radius:999px;border:1px solid var(--line);font-size:12px;color:var(--muted);white-space:nowrap}
.ppx .ppx-pill i{font:500 10.5px/1 var(--mono);font-style:normal;width:20px;height:20px;border-radius:50%;display:grid;place-items:center;background:var(--cell);color:var(--muted)}
.ppx .ppx-pill.comm i{background:var(--wire-soft);color:var(--wire)}
.ppx .ppx-pill.done{color:var(--fg)}
.ppx .ppx-pill[aria-current="step"]{border-color:var(--fg);color:var(--fg)}
.ppx .ppx-pill[aria-current="step"] i{background:var(--fg);color:var(--bg)}
.ppx .ppx-pill.comm[aria-current="step"]{border-color:var(--wire)}
.ppx .ppx-pill.comm[aria-current="step"] i{background:var(--wire);color:#fff}
.ppx .ppx-cap{display:grid;grid-template-columns:1fr auto;gap:4px 12px;align-items:baseline}
.ppx .ppx-t{font-weight:700;font-size:17px;line-height:1.3;margin:0}
.ppx .ppx-d{grid-column:1/3;margin:0;color:var(--muted)}
.ppx .ppx-nav{display:flex;gap:6px;align-self:start}
.ppx .ppx-btn{all:unset;cursor:pointer;font-size:13px;line-height:1;padding:8px 12px;border-radius:8px;border:1px solid var(--line);color:var(--fg)}
.ppx .ppx-btn:hover{border-color:var(--faint)}
.ppx .ppx-scroll{overflow-x:auto}
.ppx .ppx-grid{min-width:600px;display:grid;gap:5px}
.ppx .ppx-hdr,.ppx .ppx-row{display:grid;grid-template-columns:58px minmax(0,1fr) 14px 84px;align-items:center}
.ppx .ppx-hdr{font:500 10.5px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint);padding-bottom:4px}
.ppx .ppx-lbl{font:500 12px/1 var(--mono);color:var(--muted)}
.ppx .ppx-axis{display:grid;grid-template-columns:repeat(32,minmax(0,1fr));gap:3px}
.ppx .ppx-ruler{row-gap:2px;align-items:end}
.ppx .ppx-title{grid-column:1/33;color:var(--muted)}
.ppx .ppx-seg{position:relative;padding:0 2px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ppx .ppx-seg::after{content:"";position:absolute;left:0;right:0;bottom:0;height:5px;border:1px solid var(--faint);border-top:0}
.ppx .ppx-seg.ctx{grid-column:1/17}
.ppx .ppx-seg.cur{grid-column:17/33}
.ppx .ppx-heads{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:2px}
.ppx .c{height:18px;border-radius:4px;background:var(--cell);display:grid;place-items:center}
.ppx .c.stored{background:var(--kv)}
.ppx .c.new{background:var(--bg);box-shadow:inset 0 0 0 2px var(--new)}
.ppx .c.new.q{box-shadow:inset 0 0 0 2px var(--new),0 0 0 2px var(--bg),0 0 0 4px var(--q)}
.ppx .c.temp{background:var(--tmp-soft);box-shadow:inset 0 0 0 1px var(--tmp)}
.ppx .c.q{box-shadow:0 0 0 2px var(--bg),0 0 0 4px var(--q)}
.ppx .c.rx::after{content:"";width:6px;height:6px;border-radius:50%;background:var(--dot)}
.ppx .h{height:18px;border-radius:3px;background:var(--cell);display:grid;place-items:center}
.ppx .h.rx::after{content:"";width:4px;height:4px;border-radius:50%;background:var(--dot)}
.ppx .c.q.act{box-shadow:0 0 0 2px var(--bg),0 0 0 5px var(--q)}
.ppx .h.on{background:var(--q)}
.ppx .ppx-row.dim{opacity:.28}
.ppx .ppx-wire{display:grid;grid-template-columns:58px 1fr;align-items:center;margin-top:6px}
.ppx .ppx-lane{position:relative;height:36px;border-radius:8px;border:1px dashed var(--line);display:flex;align-items:center;gap:10px;padding:0 12px;overflow:hidden;color:var(--faint);font-size:13px}
.ppx .ppx-lane.on{border:1px solid var(--wire);background:var(--wire-soft);color:var(--fg)}
.ppx .ppx-tag{font:500 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--wire);white-space:nowrap}
.ppx .ppx-vol{font-family:var(--mono);font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ppx .ppx-pips{position:absolute;left:0;right:0;bottom:0;height:3px;opacity:0}
.ppx .ppx-lane.on .ppx-pips{opacity:1;background:repeating-linear-gradient(90deg,var(--wire) 0 10px,transparent 10px 22px);background-size:22px 3px}
@media (prefers-reduced-motion:no-preference){.ppx .ppx-lane.on .ppx-pips{animation:ppx-flow .9s linear infinite}.ppx .chg{animation:ppx-pop .45s ease-out}.ppx .c.enter{animation:ppx-enter .45s ease-out}.ppx .ppx-heads.act .h.on{animation:ppx-pulse 1.2s ease-in-out infinite}.ppx .c.q.act{animation:ppx-ring 1.2s ease-in-out infinite}}
@keyframes ppx-pulse{50%{opacity:.55}}
@keyframes ppx-enter{from{transform:translateX(14px);opacity:0}}
@keyframes ppx-ring{50%{box-shadow:0 0 0 2px var(--bg),0 0 0 3px var(--q)}}
@keyframes ppx-flow{to{background-position:22px 0}}
@keyframes ppx-pop{0%{transform:scale(.7)}60%{transform:scale(1.12)}100%{transform:scale(1)}}
.ppx .ppx-legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;color:var(--muted)}
.ppx .ppx-legend span{display:inline-flex;align-items:center;gap:7px}
.ppx .sw{width:14px;height:14px;border-radius:3px;background:var(--cell);display:inline-grid;place-items:center}
.ppx .sw.kv{background:var(--kv)}.ppx .sw.new{background:transparent;box-shadow:inset 0 0 0 2px var(--new)}.ppx .sw.tmp{background:var(--tmp-soft);box-shadow:inset 0 0 0 1px var(--tmp)}.ppx .sw.h{background:var(--q)}
.ppx .sw.q{background:transparent;box-shadow:0 0 0 2px var(--q) inset}
.ppx .sw.rx::after{content:"";width:5px;height:5px;border-radius:50%;background:var(--dot)}
.ppx table{width:100%;min-width:640px;border-collapse:collapse;font-size:13px;margin:0;border:0}
.ppx th,.ppx td{text-align:left;padding:8px 10px;border:0;border-bottom:1px solid var(--line);vertical-align:top;background:transparent;color:var(--fg)}
.ppx th{font:500 10.5px/1.3 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.ppx td:first-child{font-weight:700;white-space:nowrap}
.ppx tr,.ppx tbody tr:nth-child(even){background:transparent}
.ppx tr.cur td{background:var(--cur)}
.ppx tr.cur td:first-child{box-shadow:3px 0 0 var(--q) inset}
.ppx .ok{color:var(--kv);font-weight:500}.ppx .no{color:var(--muted)}
.ppx .ppx-foot{font-size:12px;color:var(--faint);margin:0}
@media (max-width:640px){.ppx .ppx-modes{grid-template-columns:repeat(2,minmax(0,1fr))}.ppx .ppx-cap{grid-template-columns:1fr}.ppx .ppx-nav{grid-column:1;grid-row:3}.ppx .ppx-d{grid-column:1}.ppx .ppx-btn{flex:1;text-align:center}}
.ppx [hidden]{display:none!important}`;document.head.appendChild(st)}
root.classList.add("ppx");
root.innerHTML=`<div class="ppx-modes" role="group" aria-label="Parallelism scheme">
<button class="ppx-mode" data-m="0"><b>DP8</b><span>splits requests</span></button>
<button class="ppx-mode" data-m="1"><b>TP8</b><span>splits heads</span></button>
<button class="ppx-mode" data-m="2"><b>PCP8</b><span>splits query tokens</span></button>
<button class="ppx-mode" data-m="3"><b>PCP8 × DCP8</b><span>splits queries and KV</span></button>
</div>
<div class="ppx-panel" tabindex="0" aria-label="Step-by-step view; use the left and right arrow keys to change step">
<div class="ppx-timeline" id="ppx-timeline"></div>
<div class="ppx-cap" aria-live="polite"><p class="ppx-t" id="ppx-t"></p><div class="ppx-nav"><button class="ppx-btn" id="ppx-prev" aria-label="Previous step">← Prev</button><button class="ppx-btn" id="ppx-next" aria-label="Next step">Next →</button></div><p class="ppx-d" id="ppx-d"></p></div>
<div class="ppx-scroll"><div class="ppx-grid" id="ppx-grid"></div></div>
<div class="ppx-legend"><span data-k="kv"><i class="sw kv"></i>in KV cache</span><span data-k="new"><i class="sw new"></i>new KV, not yet in cache</span><span data-k="tmp"><i class="sw tmp"></i>temporary copy, freed after the layer</span><span data-k="q"><i class="sw q"></i>queries computed here</span><span data-k="h"><i class="sw h"></i>heads computed here</span><span data-k="rx"><i class="sw rx"></i>received from other GPUs</span></div>
</div>
<div class="ppx-panel">
<div class="ppx-scroll">
<table>
<thead><tr><th>Scheme</th><th>Splits</th><th>KV per GPU</th><th>Query work per GPU</th><th>Attention traffic per layer</th><th>1M-token prompt</th></tr></thead>
<tbody id="ppx-cmp">
<tr><td>DP8</td><td>requests</td><td>full, on one GPU</td><td>all tokens, 64 heads, on one GPU</td><td>none</td><td class="no">does not fit (~300K tokens)</td></tr>
<tr><td>TP8</td><td>heads</td><td>full, replicated 8×</td><td>all tokens, 8 heads; indexer repeated 8×</td><td>all-reduce of outputs</td><td class="no">does not fit</td></tr>
<tr><td>PCP8</td><td>query tokens</td><td>full, replicated 8×</td><td>1/8 of tokens, 64 heads</td><td>all-gather of new KV</td><td class="no">does not fit</td></tr>
<tr><td>PCP8 × DCP8</td><td>queries and KV</td><td>1/8 (6.7 GiB at 1M)</td><td>1/8 of tokens, 64 heads</td><td>new KV + whole-context gather</td><td class="ok">fits</td></tr>
</tbody>
</table>
</div>
<p class="ppx-foot">C = tokens prefilled in earlier steps; N = this step's tokens, up to 16,384 (the prefill engine's max-num-batched-tokens). A 1M-token prompt takes 64 steps, and C grows by N each step. Each cell stands for a group of tokens; DCP actually deals single tokens round-robin (token i on GPU i mod 8). MoE expert-parallel traffic is the same in all four schemes and is left out.</p>
</div>`;
var own=function(g,i){return i===g||i===15-g};
var ALL=[0,1,2,3,4,5,6,7];
var MODES=[
 {rows:function(g){return g>0?"dim":""},heads:function(g){return g===0?ALL:[]},
  steps:[
   ["Start","The whole request lives on GPU 0. The other seven GPUs serve other requests and cannot help with this prompt."],
   ["Compute","GPU 0 alone computes queries and new KV for all 16 chunks, with all 64 heads."],
   ["Write","GPU 0 writes the new KV into its cache."],
   ["Attend","GPU 0 runs the indexer and sparse attention against its local cache for all 16 chunks. No communication, but one GPU does all of this prompt's attention and must hold all of its KV."],
   ["Next step","This step's tokens join GPU 0's earlier context and the next chunk of the prompt starts. The context keeps growing on one GPU."]],
  cell:function(g,r,i,s){if(g>0)return{};if(r===0)return{kv:"stored"};return{kv:s===0?"":s===1?"new":"stored",q:true}}},
 {rows:function(){return""},heads:function(g){return[g]},
  steps:[
   ["Start","Every GPU holds a full copy of the context. MLA's latent KV is shared by all 64 heads, so it cannot be split by head."],
   ["Compute","Every GPU computes the new KV for all 16 chunks (eight identical copies) and queries for every token, but only for its own 8 heads."],
   ["Write","Every GPU writes the same full new KV into its cache."],
   ["Attend","Every GPU repeats the full indexer, then attends every token for its 8 heads only."],
   ["All-reduce","Each GPU projects its 8 heads' outputs through its slice of the output projection, giving a partial hidden state for every token. The all-reduce sums the 8 partials, so every GPU gets each token's full output.",["All-reduce","attention output · N tokens × 6,144 hidden, BF16"]],
   ["Next step","This step's tokens join the earlier context and the next chunk starts. Every GPU keeps growing the same full copy."]],
  cell:function(g,r,i,s){if(r===0)return{kv:"stored"};return{kv:s===0?"":s===1?"new":"stored",q:true}}},
 {rows:function(){return""},heads:function(){return ALL},
  steps:[
   ["Start","Every GPU holds a full copy of the earlier context. Each owns two query chunks of this step, g and 15 − g, with all 64 heads."],
   ["Compute","Each GPU computes queries and new KV for its own two chunks only."],
   ["Gather new","Every GPU receives the rest of this step's new KV.",["All-gather","new KV · each GPU receives 7/8 × N tokens"]],
   ["Write","Every GPU writes all of it, so each keeps a full replica of the context."],
   ["Attend","Each GPU runs the indexer and sparse attention for its two chunks against its local full cache. No more communication."],
   ["Next step","This step's tokens join the earlier context and the next chunk is split the same way. Every GPU keeps a full, growing replica."]],
  cell:function(g,r,i,s){if(r===0)return{kv:"stored"};var q=own(g,i);
   if(s===0)return{q:q};if(s===1)return{kv:q?"new":"",q:q};if(s===2)return{kv:"new",q:q,rx:!q};return{kv:"stored",q:q}}},
 {rows:function(){return""},heads:function(){return ALL},
  steps:[
   ["Start","Each GPU stores only its eighth of the earlier context (token i on GPU i mod 8). Queries are split as in PCP: chunks g and 15 − g."],
   ["Compute","Each GPU computes queries and new KV for its own two chunks only."],
   ["Gather new","Every GPU receives the rest of this step's new KV.",["All-gather","new KV · each GPU receives 7/8 × N tokens"]],
   ["Write","Each GPU writes only the new tokens it owns (i mod 8 = g) and drops the rest. Its query chunks and its stored KV no longer line up."],
   ["Gather shards","Each GPU assembles a temporary copy of the whole context from everyone's shards.",["All-gather","KV shards + indexer keys · each GPU receives 7/8 × (C + N) tokens, BF16"]],
   ["Attend","Each GPU runs the indexer and sparse attention for its two chunks against the temporary copy."],
   ["Free","The copy is freed; only the 1/8 shards stay resident."],
   ["Next step","This step's tokens join the earlier context, still dealt round-robin, so each GPU keeps 1/8 of a larger context. The next chunk is split the same way."]],
  cell:function(g,r,i,s){var q=r===1&&own(g,i),mine=i%8===g,tmp=s===4||s===5;
   if(r===0)return{kv:mine?"stored":tmp?"temp":""};
   if(s===0)return{q:q};if(s===1)return{kv:q?"new":"",q:q};if(s===2)return{kv:"new",q:q,rx:!q};
   return{kv:mine?"stored":tmp?"temp":"",q:q}}}
];
var mode=3,step=0,prev={};
function $(id){return document.getElementById(id)}
function render(fresh){
  var m=MODES[mode],st=m.steps[step],comm=st[2],h,g,r,i,k;
  root.querySelectorAll(".ppx-mode").forEach(function(b,k){b.setAttribute("aria-pressed",k===mode)});
  [].forEach.call($("ppx-cmp").rows,function(tr,k){tr.classList.toggle("cur",k===mode)});
  $("ppx-timeline").innerHTML=m.steps.map(function(s,k){
    return '<button class="ppx-pill'+(s[2]?" comm":"")+(k<step?" done":"")+'" data-s="'+k+'"'+(k===step?' aria-current="step"':"")+"><i>"+(k+1)+"</i>"+s[0]+"</button>"}).join("");
  $("ppx-t").textContent=st[0];
  $("ppx-d").textContent=st[1];
  h='<div class="ppx-hdr"><span></span><div class="ppx-axis ppx-ruler"><span class="ppx-title">Token position in the request\'s block table →</span><span class="ppx-seg ctx">earlier context · '+(st[0]==="Next step"?"C + N":"C")+' tokens</span><span class="ppx-seg cur">'+(st[0]==="Next step"?"next step":"this step")+' · N tokens · 16 query chunks</span></div><span></span><span>Heads</span></div>';
  var att=st[0]==="Attend",ar=st[0]==="All-reduce",nxt=st[0]==="Next step",next={};
  function cellAt(g,r,i){if(!nxt)return m.cell(g,r,i,step);var c=m.cell(g,r,i,0);return r===0?c:{q:c.q}}
  function cls(key,base,fill){next[key]=fill;return base+(fresh&&fill&&prev[key]!==undefined&&prev[key]!==fill&&prev[key]!==fill+"+"?" chg":"")}
  for(g=0;g<8;g++){
    h+='<div class="ppx-row '+m.rows(g)+'"><span class="ppx-lbl">GPU '+g+"</span>";
    h+='<div class="ppx-axis">';
    for(r=0;r<2;r++){
      for(i=0;i<16;i++){var c=cellAt(g,r,i),x="c "+(c.kv||"");
        if(c.q)x+=" q";
        if(c.rx)x+=" rx";
        if(att&&c.q)x+=" act";
        if(ar&&r===1&&c.q)x+=" rx";
        if(nxt&&fresh&&r===1)x+=" enter";
        h+='<div class="'+cls(g+"-"+r+"-"+i,x,(c.kv||"")+(nxt&&r===0&&c.kv?"+":""))+'"></div>'}
    }
    h+="</div>";
    var hs=m.heads(g);
    h+='<span></span><div class="ppx-heads'+(att&&hs.length?" act":"")+'">'+ALL.map(function(k){
      var on=hs.indexOf(k)>=0,x="h"+(on?" on":"");
      return '<div class="'+cls(g+"-h"+k,x,on?"on":"")+'"></div>'}).join("")+"</div></div>";
  }
  prev=next;
  h+='<div class="ppx-wire"><span class="ppx-lbl">Wire</span><div class="ppx-lane'+(comm?" on":"")+'">'
    +(comm?'<span class="ppx-tag">'+comm[0]+'</span><span class="ppx-vol">'+comm[1]+"</span>":"<span>No communication in this step</span>")
    +'<span class="ppx-pips"></span></div></div>';
  $("ppx-grid").innerHTML=h;
  root.querySelectorAll(".ppx-pill").forEach(function(p){p.onclick=function(){step=+p.dataset.s;render(true)}});
}
function legend(){
  var m=MODES[mode],u={},s,g,r,i,c;
  for(s=0;s<m.steps.length;s++)for(g=0;g<8;g++){
    if(m.heads(g).length)u.h=1;
    if(m.steps[s][0]==="All-reduce")u.rx=1;
    for(r=0;r<2;r++)for(i=0;i<16;i++){c=m.cell(g,r,i,s);
      if(c.kv==="stored")u.kv=1;if(c.kv==="new")u["new"]=1;if(c.kv==="temp")u.tmp=1;if(c.q)u.q=1;if(c.rx)u.rx=1}}
  root.querySelectorAll(".ppx-legend [data-k]").forEach(function(e){e.hidden=!u[e.getAttribute("data-k")]});
}
function go(d){var n=MODES[mode].steps.length;step=(step+d+n)%n;render(true)}
$("ppx-prev").onclick=function(){go(-1)};$("ppx-next").onclick=function(){go(1)};
root.querySelectorAll(".ppx-mode").forEach(function(b){b.onclick=function(){mode=+b.dataset.m;step=0;prev={};legend();render(false)}});
root.addEventListener("keydown",function(e){
  if(e.key==="ArrowRight"){go(1);e.preventDefault()}
  if(e.key==="ArrowLeft"){go(-1);e.preventDefault()}
});
legend();
render(false);
})();
