'use strict';
const TG=0,MOD=1,TG_STD=2,MOD_STD=3,SIM=4,TG_KNOWN=5,MOD_KNOWN=6;
const TABLE_LIMIT=300,HIT_RADIUS=24,MEASURED_PRIORITY_RADIUS=5,PAD={l:62,r:16,t:12,b:46};
const $=id=>document.getElementById(id);
const state={data:null,filtered:[],pareto:new Set(),selected:null,selectedMeasured:null,view:null,hover:null,sortKey:'tg',sortDir:-1};
const canvas=$('chart'),ctx=canvas.getContext('2d');

const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const firstId=id=>id.split(';')[0];
const dian=r=>state.data.dianhydrides[r.d];
const diam=r=>state.data.diamines[r.a];
const MODEL='film';
const vals=r=>r[MODEL];
const AXES={mod:{label:'引張弾性率',unit:'GPa',digits:2,log:true,axis:'引張弾性率 (GPa, 対数軸)',png:'assets/virtual_map.png',
    y:r=>r[MODEL][MOD],sim:r=>r[MODEL][SIM],measured:()=>state.data.measured[MODEL]},
  cte:{label:'CTE',unit:'ppm/℃',digits:0,log:false,axis:'CTE (ppm/℃)',png:'assets/virtual_map_cte.png',
    y:r=>r.cte[0],sim:r=>r.cte[2],measured:()=>state.data.measured.cte}};
const axis=()=>AXES[$('yAxis').value];
const yOf=r=>axis().y(r);
const simOf=r=>axis().sim(r);
const hasFluorine=smiles=>smiles.includes('F');
const excludeFluorine=()=>$('excludeFluorine').checked;
const measuredHasFluorine=m=>m[5].units.some(u=>hasFluorine(state.data.unitSmiles[u]));
const measured=()=>excludeFluorine()?axis().measured().filter(m=>!measuredHasFluorine(m)):axis().measured();
const yScale=v=>axis().log?Math.log10(Math.max(v,1e-6)):v;

function applyFilters(){
  const d=$('dianhydride').value,type=$('diamineType').value,q=$('search').value.trim().toLowerCase(),qd=$('searchDianhydride').value.trim().toLowerCase();
  const min=+$('similarity').value,hetero=$('excludeHetero').checked,fluorine=excludeFluorine();
  $('simValue').textContent=min.toFixed(2);
  state.filtered=state.data.rows.filter(r=>{
    const a=diam(r);
    if(d&&r.d!==d)return false;
    if(type&&a.type!==type)return false;
    if(simOf(r)<min)return false;
    if(hetero&&a.flags.includes('heteroaryl_amine'))return false;
    if(fluorine&&hasFluorine(r.p))return false;
    if(q&&!(a.id.toLowerCase().includes(q)||a.name.toLowerCase().includes(q)))return false;
    if(qd){const da=dian(r);if(!(da.id.toLowerCase().includes(qd)||da.name.toLowerCase().includes(qd)))return false}
    return true;
  });
  // Pareto front on high Tg and the chosen y direction, among points near the training data.
  state.pareto=new Set();
  let best=-Infinity;
  const sign=$('direction').value==='high'?1:-1;
  const near=state.filtered.filter(r=>simOf(r)>=state.data.near).sort((a,b)=>vals(b)[TG]-vals(a)[TG]);
  for(const r of near)if(sign*yOf(r)>best){state.pareto.add(r.i);best=sign*yOf(r)}
  render();
}

function fullView(){
  const xs=state.data.rows.map(r=>vals(r)[TG]).concat(measured().map(m=>m[0]));
  const ys=state.data.rows.map(yOf).concat(measured().map(m=>m[1])).filter(v=>!axis().log||v>0).map(yScale);
  const pad=(a,b)=>(b-a)*.04;
  const [x0,x1,y0,y1]=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
  return{x0:x0-pad(x0,x1),x1:x1+pad(x0,x1),y0:y0-pad(y0,y1),y1:y1+pad(y0,y1)};
}
function size(){
  const box=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1;
  if(canvas.width!==Math.round(box.width*ratio)||canvas.height!==Math.round(box.height*ratio)){canvas.width=Math.round(box.width*ratio);canvas.height=Math.round(box.height*ratio)}
  ctx.setTransform(ratio,0,0,ratio,0,0);
  return box;
}
function scales(box){
  const v=state.view;
  return{sx:x=>PAD.l+(x-v.x0)/(v.x1-v.x0)*(box.width-PAD.l-PAD.r),
         sy:y=>box.height-PAD.b-(yScale(y)-v.y0)/(v.y1-v.y0)*(box.height-PAD.t-PAD.b)};
}
function niceTicks(a,b,count){
  const step0=(b-a)/count,mag=10**Math.floor(Math.log10(step0)),step=[1,2,5,10].map(m=>m*mag).find(s=>s>=step0);
  const ticks=[];for(let t=Math.ceil(a/step)*step;t<=b+1e-9;t+=step)ticks.push(+t.toFixed(8));
  return ticks;
}
function logTicks(y0,y1){
  const out=[];
  for(let e=Math.floor(y0);e<=Math.ceil(y1);e++)for(const m of y1-y0>1.2?[1,2,5]:[1,1.5,2,3,5,7])
    {const v=m*10**e;if(Math.log10(v)>=y0&&Math.log10(v)<=y1)out.push(v)}
  return out;
}
function draw(){
  const box=size(),{sx,sy}=scales(box),line=css('--line'),muted=css('--muted'),ink=css('--ink');
  ctx.clearRect(0,0,box.width,box.height);
  ctx.font='12px system-ui,sans-serif';ctx.lineWidth=1;
  ctx.fillStyle=muted;ctx.strokeStyle=line;
  ctx.textAlign='center';ctx.textBaseline='top';
  for(const t of niceTicks(state.view.x0,state.view.x1,9)){const x=sx(t);ctx.beginPath();ctx.moveTo(x,PAD.t);ctx.lineTo(x,box.height-PAD.b);ctx.stroke();ctx.fillText(t,x,box.height-PAD.b+6)}
  ctx.textAlign='right';ctx.textBaseline='middle';
  for(const t of axis().log?logTicks(state.view.y0,state.view.y1):niceTicks(state.view.y0,state.view.y1,7)){const y=sy(t);ctx.beginPath();ctx.moveTo(PAD.l,y);ctx.lineTo(box.width-PAD.r,y);ctx.stroke();ctx.fillText(t,PAD.l-6,y)}
  ctx.textAlign='center';ctx.textBaseline='alphabetic';
  ctx.fillText('Tg (℃)',PAD.l+(box.width-PAD.l-PAD.r)/2,box.height-8);
  ctx.save();ctx.translate(14,PAD.t+(box.height-PAD.t-PAD.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText(axis().axis,0,0);ctx.restore();
  ctx.save();ctx.beginPath();ctx.rect(PAD.l,PAD.t,box.width-PAD.l-PAD.r,box.height-PAD.t-PAD.b);ctx.clip();
  const near=state.data.near;
  ctx.globalAlpha=.55;ctx.fillStyle=css('--far');
  for(const r of state.filtered)if(simOf(r)<near){ctx.beginPath();ctx.arc(sx(vals(r)[TG]),sy(yOf(r)),2.5,0,7);ctx.fill()}
  ctx.globalAlpha=.7;ctx.fillStyle=css('--near');
  for(const r of state.filtered)if(simOf(r)>=near){ctx.beginPath();ctx.arc(sx(vals(r)[TG]),sy(yOf(r)),3,0,7);ctx.fill()}
  ctx.globalAlpha=1;
  if($('showMeasured').checked){
    ctx.fillStyle=css('--measured');ctx.strokeStyle=css('--surface');ctx.lineWidth=.8;
    for(const m of measured()){const x=sx(m[0]),y=sy(m[1]);ctx.beginPath();ctx.moveTo(x,y-5);ctx.lineTo(x+4.5,y+3.5);ctx.lineTo(x-4.5,y+3.5);ctx.closePath();ctx.fill();ctx.stroke()}
  }
  const front=state.filtered.filter(r=>state.pareto.has(r.i)).sort((a,b)=>vals(a)[TG]-vals(b)[TG]);
  ctx.strokeStyle=ink;ctx.lineWidth=1.5;
  for(const r of front){ctx.beginPath();ctx.arc(sx(vals(r)[TG]),sy(yOf(r)),5,0,7);ctx.stroke()}
  if(front.length>1){ctx.lineWidth=1;ctx.beginPath();front.forEach((r,i)=>{const x=sx(vals(r)[TG]),y=sy(yOf(r));i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()}
  if(state.selectedMeasured&&$('showMeasured').checked){const m=state.selectedMeasured;ctx.strokeStyle=ink;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(m[0]),sy(m[1]),9,0,7);ctx.stroke()}
  if(state.selected){const r=state.selected;ctx.strokeStyle=ink;ctx.lineWidth=2.5;ctx.beginPath();ctx.arc(sx(vals(r)[TG]),sy(yOf(r)),8,0,7);ctx.stroke()}
  ctx.restore();
}

function nearest(mx,my){
  const box=canvas.getBoundingClientRect(),{sx,sy}=scales(box);
  let best=null,dist=HIT_RADIUS**2;
  for(const r of state.filtered){const d=(sx(vals(r)[TG])-mx)**2+(sy(yOf(r))-my)**2;if(d<dist){dist=d;best={kind:'pred',r}}}
  if($('showMeasured').checked){
    // Measured markers are drawn on top, so a click on one must not fall through to a closer prediction.
    let onTop=null,topDist=MEASURED_PRIORITY_RADIUS**2;
    for(const m of measured()){
      const d=(sx(m[0])-mx)**2+(sy(m[1])-my)**2;
      if(d<topDist){topDist=d;onTop=m}
      if(d<dist){dist=d;best={kind:'measured',m}}
    }
    if(onTop)return{kind:'measured',m:onTop};
  }
  return best;
}
function showTooltip(hit,mx,my){
  const t=$('tooltip');
  if(!hit){t.hidden=true;return}
  if(hit.kind==='pred'){
    const v=vals(hit.r);
    t.innerHTML=`<b>${esc(firstId(dian(hit.r).id))} + ${esc(firstId(diam(hit.r).id))}</b>${esc(diam(hit.r).name)}<br>Tg ${v[TG].toFixed(0)} ℃ ・ ${v[MOD].toFixed(2)} GPa ・ CTE ${hit.r.cte[0].toFixed(0)} ppm/℃<br>類似度 ${simOf(hit.r).toFixed(2)}`;
  }else{
    const m=hit.m;
    t.innerHTML=`<b>実測(${m[2]==='homopolymer'?'ホモポリマー':'共重合体'})</b>Tg ${m[0].toFixed(0)} ℃ ・ ${m[1].toFixed(axis().digits)} ${axis().unit}<br>測定数 Tg ${m[3]} / ${axis().label} ${m[4]}`;
  }
  t.hidden=false;
  const box=canvas.getBoundingClientRect();
  t.style.left=Math.min(mx+14,box.width-t.offsetWidth-4)+'px';
  t.style.top=Math.max(my-t.offsetHeight-10,4)+'px';
}

const KIND_LABELS={homopolymer:'ホモポリマー',copolymer:'共重合体'};
function selectMeasured(m){
  state.selected=null;state.selectedMeasured=m;
  const info=m[5],ax=axis(),range=(values,digits)=>values[0]===values[1]?'':`(範囲 ${values[0].toFixed(digits)}〜${values[1].toFixed(digits)})`;
  const refs=list=>list.map(ref=>String(ref).replace(/^0+/,'')).join(', ');
  const units=info.units.map((u,i)=>`<h3>反復単位${info.units.length>1?` ${i+1}(モル分率 ${info.fractions[i].toFixed(3)})`:''}</h3><img class="structure" src="structures/u${u}.svg" alt="反復単位の構造式"><p class="smiles">${esc(state.data.unitSmiles[u])}</p>`).join('');
  $('detail').innerHTML=`<h2>実測: ${KIND_LABELS[m[2]]??esc(m[2])}</h2>
  ${info.names.length?`<p class="muted">${info.names.map(esc).join('<br>')}</p>`:''}
  <dl>
   <dt>Tg 実測中央値</dt><dd>${m[0].toFixed(0)} ℃ ${range(info.tgRange,0)}、${m[3]}件</dd>
   <dt>${ax.label} 実測中央値</dt><dd>${m[1].toFixed(ax.digits)} ${ax.unit} ${range(info.yRange,ax.digits)}、${m[4]}件</dd>
   ${m[2]==='copolymer'?`<dt>共重合様式</dt><dd>${esc(info.sequence)}</dd>`:''}
   <dt>文献No. (Tg)</dt><dd>${esc(refs(info.refsTg))}</dd>
   <dt>文献No. (${ax.label})</dt><dd>${esc(refs(info.refsY))}</dd>
  </dl>${units}`;
  draw();renderTable();
}

function select(r){
  state.selected=r;state.selectedMeasured=null;
  const v=vals(r),d=dian(r),a=diam(r);
  const known=v[TG_KNOWN]!=null||v[MOD_KNOWN]!=null;
  const fmt=(x,n)=>x==null?'—':x.toFixed(n);
  const link=m=>m.url?`<a href="${esc(m.url)}" target="_blank" rel="noopener">${esc(m.id)}</a>`:esc(m.id);
  $('detail').innerHTML=`<h2>${esc(firstId(d.id))} + ${esc(firstId(a.id))}</h2>
  <h3>反復単位</h3><img class="structure" src="structures/p${r.i}.svg" alt="反復単位の構造式">
  <dl>
   <dt>Tg 予測</dt><dd>${v[TG].toFixed(0)} ℃ (モデル間 ±${v[TG_STD].toFixed(0)})</dd>
   <dt>弾性率 予測</dt><dd>${v[MOD].toFixed(2)} GPa (モデル間 ±${v[MOD_STD].toFixed(2)})</dd>
   <dt>CTE 予測</dt><dd>${r.cte[0].toFixed(0)} ppm/℃ (モデル間 ±${r.cte[1].toFixed(0)})</dd>
   <dt>類似度</dt><dd>弾性率 ${v[SIM].toFixed(2)} / CTE ${r.cte[2].toFixed(2)}${simOf(r)<state.data.near?'(外挿の可能性)':''}</dd>
   ${known||r.cte[3]!=null?`<dt>実測中央値</dt><dd>Tg ${fmt(v[TG_KNOWN],0)} ℃ / ${fmt(v[MOD_KNOWN],2)} GPa / ${fmt(r.cte[3],0)} ppm/℃(学習データ)</dd>`:''}
   <dt>繰り返し分子量</dt><dd>${r.mw.toFixed(1)}</dd>
  </dl>
  ${r.alt?'<p class="muted">両モノマーが非対称のため、別の位置異性体もあります。ここでは一方のみ予測しています。</p>':''}
  <p class="smiles">${esc(r.p)}</p>
  <h3>酸二無水物: ${link(d)}</h3><p class="muted">${esc(d.name)}</p><img class="structure monomer" src="structures/${d.key}.svg" alt="酸二無水物の構造式">
  <h3>ジアミン: ${link(a)}</h3><p class="muted">${esc(a.name)}${a.flags?`<br>注意: ${esc(a.flags)}`:''}</p><img class="structure monomer" src="structures/${a.key}.svg" alt="ジアミンの構造式">`;
  draw();renderTable();
}

const COLUMNS=[
  {key:'d',label:'酸二無水物',get:r=>firstId(dian(r).id),title:r=>dian(r).name},
  {key:'a',label:'ジアミン',get:r=>firstId(diam(r).id),title:r=>diam(r).name},
  {key:'name',label:'ジアミン名',get:r=>diam(r).name},
  {key:'tg',label:'Tg 予測 (℃)',num:1,get:r=>vals(r)[TG],fmt:x=>x.toFixed(0)},
  {key:'mod',label:'弾性率 予測 (GPa)',num:1,get:r=>vals(r)[MOD],fmt:x=>x.toFixed(2)},
  {key:'cte',label:'CTE 予測 (ppm/℃)',num:1,get:r=>r.cte[0],fmt:x=>x.toFixed(0)},
  {key:'sim',label:'類似度',num:1,get:r=>simOf(r),fmt:x=>x.toFixed(2)},
];
function currentRows(){
  const pool=$('paretoOnly').checked?state.filtered.filter(r=>state.pareto.has(r.i)):state.filtered;
  const col=COLUMNS.find(c=>c.key===state.sortKey);
  return[...pool].sort((a,b)=>{const x=col.get(a),y=col.get(b);return(x<y?-1:x>y?1:0)*state.sortDir});
}
function renderTable(){
  const rows=currentRows(),shown=rows.slice(0,TABLE_LIMIT);
  $('table').tHead.innerHTML='<tr>'+COLUMNS.map(c=>`<th class="${c.num?'num':''}" aria-sort="${state.sortKey===c.key?(state.sortDir>0?'ascending':'descending'):'none'}"><button type="button" data-key="${c.key}">${c.label}${state.sortKey===c.key?(state.sortDir>0?' ▲':' ▼'):''}</button></th>`).join('')+'</tr>';
  $('table').tBodies[0].innerHTML=shown.map(r=>`<tr tabindex="0" data-i="${r.i}" class="${state.selected?.i===r.i?'selected':''}">`+COLUMNS.map(c=>{const x=c.get(r);return`<td class="${c.num?'num':''}">${esc(c.fmt?c.fmt(x):x)}</td>`}).join('')+'</tr>').join('');
  $('tableTitle').textContent=`候補一覧(${rows.length.toLocaleString()}件)`;
  $('tableNote').textContent=rows.length>TABLE_LIMIT?`先頭 ${TABLE_LIMIT} 件を表示しています。全件はCSVで取得できます。`:state.filtered.length&&!rows.length?'この条件では学習データに近い候補がありません。「パレート前線のみ」を外してください。':'';
}
function renderTiles(){
  const m=state.data.metrics[MODEL],rows=state.filtered;
  const near=rows.filter(r=>simOf(r)>=state.data.near).length;
  $('tiles').innerHTML=[
    [rows.length.toLocaleString(),'表示中の仮想ポリイミド'],
    [near.toLocaleString(),`学習データに近い(類似度≥${state.data.near})`],
    [`R² ${m.Tg.r2.toFixed(2)} / ±${m.Tg.mae.toFixed(0)} ℃`,'Tgモデル(10分割, MAE)'],
    [`R² ${m.modulus.r2.toFixed(2)} / ±${m.modulus.mae.toFixed(2)} GPa`,'弾性率モデル(10分割, MAE)'],
    [`R² ${m.cte.r2.toFixed(2)} / ±${m.cte.mae.toFixed(0)} ppm/℃`,'CTEモデル(10分割, MAE)'],
  ].map(([b,s])=>`<div class="tile"><b>${b}</b><span>${s}</span></div>`).join('');
}
function renderLegend(){
  $('legend').innerHTML=`<span><i style="background:var(--near)"></i>予測・学習データに近い</span><span><i style="background:var(--far)"></i>予測・学習データから遠い</span><span><i class="tri" style="background:var(--measured)"></i>実測(既存データ)${measured().length.toLocaleString()}件</span><span><i class="ring"></i>パレート前線</span>`;
}
function render(){$('chartTitle').textContent=`Tg と${axis().label}`;$('mapImage').href=axis().png;renderTiles();renderLegend();draw();renderTable()}

function init(data){
  state.data=data;
  const options=Object.values(data.dianhydrides).sort((a,b)=>a.name.localeCompare(b.name));
  $('dianhydride').innerHTML='<option value="">すべて</option>'+options.filter(d=>data.rows.some(r=>r.d===d.key)).map(d=>`<option value="${d.key}">${esc(firstId(d.id))} ${esc(d.name.slice(0,44))}</option>`).join('');
  state.view=fullView();
  $('yAxis').addEventListener('input',()=>{
    // Measured points differ per axis, so a selected one may not exist on the new axis.
    state.selectedMeasured=null;
    if(state.selected)select(state.selected);
    else $('detail').innerHTML='<h2>構造の詳細</h2><p class="muted">点または表の行を選んでください。</p>';
    state.view=fullView();applyFilters()});
  for(const id of['direction','dianhydride','diamineType','similarity','excludeHetero','search','searchDianhydride'])$(id).addEventListener('input',applyFilters);
  $('excludeFluorine').addEventListener('input',()=>{
    // A selected polymer that is now hidden must not stay marked on the chart.
    const hidden=excludeFluorine()&&((state.selected&&hasFluorine(state.selected.p))||(state.selectedMeasured&&measuredHasFluorine(state.selectedMeasured)));
    if(hidden){state.selected=null;state.selectedMeasured=null;$('detail').innerHTML='<h2>構造の詳細</h2><p class="muted">点または表の行を選んでください。</p>'}
    applyFilters()});
  $('showMeasured').addEventListener('input',draw);
  $('paretoOnly').addEventListener('input',renderTable);
  $('resetZoom').addEventListener('click',()=>{state.view=fullView();draw()});
  $('table').tHead.addEventListener('click',e=>{const key=e.target.closest('button')?.dataset.key;if(!key)return;
    if(state.sortKey===key)state.sortDir*=-1;else{state.sortKey=key;state.sortDir=COLUMNS.find(c=>c.key===key).num?-1:1}renderTable()});
  const pick=e=>{const tr=e.target.closest('tr[data-i]');if(tr)select(data.rows[+tr.dataset.i])};
  $('table').tBodies[0].addEventListener('click',pick);
  $('table').tBodies[0].addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(e)}});
  $('download').addEventListener('click',download);
  let drag=null,moved=false,frame=0;
  const pos=e=>{const b=canvas.getBoundingClientRect();return[e.clientX-b.left,e.clientY-b.top]};
  canvas.addEventListener('pointerdown',e=>{drag=pos(e);moved=false;canvas.setPointerCapture(e.pointerId)});
  canvas.addEventListener('pointermove',e=>{
    const [mx,my]=pos(e);
    if(drag){const b=canvas.getBoundingClientRect(),v=state.view,dx=(mx-drag[0])/(b.width-PAD.l-PAD.r)*(v.x1-v.x0),dy=(my-drag[1])/(b.height-PAD.t-PAD.b)*(v.y1-v.y0);
      if(Math.abs(mx-drag[0])+Math.abs(my-drag[1])>2)moved=true;
      state.view={x0:v.x0-dx,x1:v.x1-dx,y0:v.y0+dy,y1:v.y1+dy};drag=[mx,my];$('tooltip').hidden=true;draw();return}
    if(frame)return;
    frame=requestAnimationFrame(()=>{frame=0;showTooltip(nearest(mx,my),mx,my)});
  });
  canvas.addEventListener('pointerup',e=>{
    if(!moved){const[mx,my]=pos(e),hit=nearest(mx,my);if(hit?.kind==='pred')select(hit.r);else if(hit?.kind==='measured')selectMeasured(hit.m)}
    drag=null});
  canvas.addEventListener('pointerleave',()=>{$('tooltip').hidden=true});
  canvas.addEventListener('wheel',e=>{e.preventDefault();
    const[mx,my]=pos(e),b=canvas.getBoundingClientRect(),v=state.view,k=e.deltaY>0?1.2:1/1.2;
    const fx=(mx-PAD.l)/(b.width-PAD.l-PAD.r),fy=1-(my-PAD.t)/(b.height-PAD.t-PAD.b);
    const ax=v.x0+fx*(v.x1-v.x0),ay=v.y0+fy*(v.y1-v.y0);
    state.view={x0:ax-(ax-v.x0)*k,x1:ax+(v.x1-ax)*k,y0:ay-(ay-v.y0)*k,y1:ay+(v.y1-ay)*k};draw()},{passive:false});
  addEventListener('resize',draw);
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  applyFilters();
}
function download(){
  const header=['dianhydride_id','dianhydride_name','diamine_id','diamine_name','psmiles','Tg_pred_C','modulus_pred_GPa','CTE_pred_ppm_per_C','Tg_member_std','modulus_member_std','CTE_member_std','similarity_modulus','similarity_CTE','Tg_measured_C','modulus_measured_GPa','CTE_measured_ppm_per_C','pareto','tci_url_dianhydride','tci_url_diamine'];
  const q=x=>`"${String(x??'').replace(/"/g,'""')}"`;
  const lines=currentRows().map(r=>{const v=vals(r),d=dian(r),a=diam(r);return[d.id,d.name,a.id,a.name,r.p,v[TG],v[MOD],r.cte[0],v[TG_STD],v[MOD_STD],r.cte[1],v[SIM],r.cte[2],v[TG_KNOWN],v[MOD_KNOWN],r.cte[3],state.pareto.has(r.i),d.url,a.url].map(q).join(',')});
  const url=URL.createObjectURL(new Blob(['﻿'+[header.join(','),...lines].join('\n')],{type:'text/csv'}));
  const link=document.createElement('a');link.href=url;link.download=`virtual_polyimides_film.csv`;link.click();URL.revokeObjectURL(url);
}
fetch('data.json').then(r=>r.json()).then(init).catch(error=>{document.body.insertAdjacentHTML('afterbegin',`<p class="warning">データの読み込みに失敗しました: ${esc(error.message)}</p>`)});
