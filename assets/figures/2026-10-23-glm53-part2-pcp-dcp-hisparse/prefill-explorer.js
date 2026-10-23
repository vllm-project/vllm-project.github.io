// Interactive Figure 3 for the GLM-5.3 Part 2 post: one prefill layer on 8 GPUs under DP8, TP8, PCP8 and PCP8 x DCP8.
(function(){
var root=document.getElementById("ppx");if(!root)return;
var me=document.currentScript;
if(me&&me.src&&!document.querySelector("link[data-ppx]")){var l=document.createElement("link");l.rel="stylesheet";l.href=me.src.replace(/\.js(\?.*)?$/,".css");l.setAttribute("data-ppx","");document.head.appendChild(l)}
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
