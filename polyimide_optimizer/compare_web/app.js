'use strict';
// One large scatter (canvas) with an axis switch, wheel zoom and drag pan, a detail panel and a
// sortable table; the layout follows polyimide_virtual_app.
const {meta,training,designs}=window.DATA;
const byId=new Map([...training,...designs].map(r=>[r.id,r]));
const $=id=>document.getElementById(id);
const canvas=$('chart'),ctx=canvas.getContext('2d');
const PAD={l:64,r:18,t:14,b:48},HIT=14;

const PROP={
  Tg:{label:'Tg',unit:'℃',digits:0,axis:'Tg (℃)'},
  modulus:{label:'引張弾性率',unit:'GPa',digits:2,log:true,axis:'引張弾性率 (GPa, 対数軸)'},
  CTE:{label:'CTE',unit:'ppm/K',digits:1,axis:'CTE (ppm/K)'},
};
const AX={...Object.fromEntries(Object.entries(PROP).map(([k,p])=>[k,{...p,get:r=>r[k]}])),
  mapx:{axis:'構造マップ 1',get:r=>r.x,map:true},mapy:{axis:'構造マップ 2',get:r=>r.y,map:true}};
const VIEWS={tgmod:['Tg','modulus'],tgcte:['Tg','CTE'],modcte:['modulus','CTE'],map:['mapx','mapy']};
const state={view:null,box:null,selected:null,hover:null,sortKey:'rank',sortDir:1,drawn:[]};

const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmt=(p,v)=>v==null||Number.isNaN(v)?'—':v.toFixed(PROP[p].digits);
const pct=v=>v==null?'—':(100*v).toFixed(0)+'%';
const isNext=d=>d.next_ranks.length>0;
const rank=d=>isNext(d)?Math.min(...d.next_ranks):Infinity;
const isDesign=r=>r.id[0]==='d';
const kindJa=r=>r.kind==='observation'?'追加した実測':r.kind==='copolymer'?'学習データ(共重合体)':'学習データ(単独重合体)';
const axes=()=>VIEWS[$('view').value].map(k=>AX[k]);

// Target region from region_search.py, e.g. "['Tg>=350', 'modulus<=3']"; 'Tg=240-260' gives two bounds.
const REGION=(meta.region?meta.region.match(/(Tg|modulus|CTE)\s*([<>]=?\s*[0-9.]+|=\s*[0-9.]+\s*-\s*[0-9.]+)/g)||[]:[]).flatMap(t=>{
  const r=t.match(/(\w+)\s*=\s*([0-9.]+)\s*-\s*([0-9.]+)/);
  if(r)return[{prop:r[1],ge:true,bound:+r[2],text:t},{prop:r[1],ge:false,bound:+r[3],text:''}];
  const m=t.match(/(\w+)\s*([<>])=?\s*([0-9.]+)/);return[{prop:m[1],ge:m[2]==='>',bound:+m[3],text:t}];});
const isRegion=REGION.length>0;
const regionText=(src=meta.region)=>((src||'').match(/(Tg|modulus|CTE)\s*([<>]=?\s*[0-9.]+|=\s*[0-9.]+\s*-\s*[0-9.]+)/g)||[]).map(t=>{
  const p=t.match(/^\w+/)[0],P=PROP[p],r=t.match(/=\s*([0-9.]+)\s*-\s*([0-9.]+)/);
  const name=p==='modulus'?'弾性率':P.label;
  if(r)return`${name} ${r[1]}〜${r[2]} ${P.unit}`;
  const m=t.match(/([<>])=?\s*([0-9.]+)/);return`${name} ${m[1]==='>'?'≥':'≤'} ${m[2]} ${P.unit}`;}).join(' かつ ');

// ---------- what is shown ----------
function visibleDesigns(){
  const f=$('designFilter').value,q=$('search').value.trim().toLowerCase();
  if(f==='none')return[];
  return designs.filter(d=>(f==='all'||isNext(d))&&(!q||(d.acids+' '+d.diamines).toLowerCase().includes(q)));
}
const observations=()=>training.filter(r=>r.kind==='observation');
const literature=()=>$('showTraining').checked?training.filter(r=>r.kind!=='observation'):[];
const has=(r,[ax,ay])=>{const x=ax.get(r),y=ay.get(r);return x!=null&&y!=null&&(!ax.log||x>0)&&(!ay.log||y>0)};

// ---------- geometry ----------
const tr=(ax,v)=>ax.log?Math.log10(v):v;
function fullView(){
  const [ax,ay]=axes(),rows=[...literature(),...observations(),...visibleDesigns()].filter(r=>has(r,[ax,ay]));
  if(!rows.length)return{x0:0,x1:1,y0:0,y1:1};
  const xs=rows.map(r=>tr(ax,ax.get(r))),ys=rows.map(r=>tr(ay,ay.get(r)));
  const [x0,x1,y0,y1]=[Math.min(...xs),Math.max(...xs),Math.min(...ys),Math.max(...ys)];
  const px=(x1-x0||1)*.05,py=(y1-y0||1)*.05;
  return{x0:x0-px,x1:x1+px,y0:y0-py,y1:y1+py};
}
function size(){
  const box=canvas.getBoundingClientRect(),ratio=devicePixelRatio||1;
  if(canvas.width!==Math.round(box.width*ratio)||canvas.height!==Math.round(box.height*ratio)){canvas.width=Math.round(box.width*ratio);canvas.height=Math.round(box.height*ratio)}
  ctx.setTransform(ratio,0,0,ratio,0,0);return box;
}
function scales(box){
  const v=state.view,[ax,ay]=axes(),w=box.width-PAD.l-PAD.r,h=box.height-PAD.t-PAD.b;
  return{sx:x=>PAD.l+(tr(ax,x)-v.x0)/(v.x1-v.x0)*w,sy:y=>box.height-PAD.b-(tr(ay,y)-v.y0)/(v.y1-v.y0)*h,
    tx:t=>PAD.l+(t-v.x0)/(v.x1-v.x0)*w,ty:t=>box.height-PAD.b-(t-v.y0)/(v.y1-v.y0)*h};
}
function niceTicks(a,b,count){
  const step0=(b-a)/count,mag=10**Math.floor(Math.log10(step0)),step=[1,2,5,10].map(m=>m*mag).find(s=>s>=step0);
  const ticks=[];for(let t=Math.ceil(a/step)*step;t<=b+1e-9;t+=step)ticks.push(+t.toFixed(8));return ticks;
}
function logTicks(t0,t1){
  const out=[];
  for(let e=Math.floor(t0);e<=Math.ceil(t1);e++)for(const m of t1-t0>1.2?[1,2,5]:[1,1.5,2,3,5,7]){const v=m*10**e;if(Math.log10(v)>=t0&&Math.log10(v)<=t1)out.push(v)}
  return out;
}

// ---------- marks ----------
function diamond(x,y,r){ctx.beginPath();ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath()}
function star(x,y,r){ctx.beginPath();for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,q=i%2?r*.45:r;ctx[i?'lineTo':'moveTo'](x+q*Math.cos(a),y+q*Math.sin(a))}ctx.closePath()}
function dot(x,y,r){ctx.beginPath();ctx.arc(x,y,r,0,7)}

function draw(){
  const box=size(),{sx,sy,tx,ty}=scales(box),[ax,ay]=axes(),v=state.view;
  const C={line:css('--line'),muted:css('--muted'),ink:css('--ink'),surface:css('--surface'),training:css('--training'),pareto:css('--pareto'),next:css('--next')};
  const L=PAD.l,R=box.width-PAD.r,T=PAD.t,B=box.height-PAD.b;
  ctx.clearRect(0,0,box.width,box.height);
  ctx.font='12px system-ui,sans-serif';ctx.lineWidth=1;ctx.fillStyle=C.muted;ctx.strokeStyle=C.line;
  // grid and axes
  if(!ax.map){
    ctx.textAlign='center';ctx.textBaseline='top';
    for(const t of ax.log?logTicks(v.x0,v.x1):niceTicks(v.x0,v.x1,9)){const x=ax.log?sx(t):tx(t);ctx.beginPath();ctx.moveTo(x,T);ctx.lineTo(x,B);ctx.stroke();ctx.fillText(t,x,B+6)}
    ctx.textAlign='right';ctx.textBaseline='middle';
    for(const t of ay.log?logTicks(v.y0,v.y1):niceTicks(v.y0,v.y1,7)){const y=ay.log?sy(t):ty(t);ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(R,y);ctx.stroke();ctx.fillText(t,L-6,y)}
  }else{ctx.strokeRect(L,T,R-L,B-T)}
  ctx.textAlign='center';ctx.textBaseline='alphabetic';
  ctx.fillText(ax.axis,L+(R-L)/2,box.height-10);
  ctx.save();ctx.translate(16,T+(B-T)/2);ctx.rotate(-Math.PI/2);ctx.fillText(ay.axis,0,0);ctx.restore();

  ctx.save();ctx.beginPath();ctx.rect(L,T,R-L,B-T);ctx.clip();
  // target region
  if(isRegion&&!ax.map){
    const span=(a,lo,hi)=>{for(const c of REGION.filter(c=>c.prop===a.key)){if(c.ge)lo=Math.max(lo,c.bound);else hi=Math.min(hi,c.bound)}return[lo,hi]};
    const xk=VIEWS[$('view').value][0],yk=VIEWS[$('view').value][1];
    if(REGION.some(c=>c.prop===xk||c.prop===yk)){
      const [x0,x1]=span({key:xk},-Infinity,Infinity),[y0,y1]=span({key:yk},-Infinity,Infinity);
      const X0=x0===-Infinity?L:sx(x0),X1=x1===Infinity?R:sx(x1),Y0=y0===-Infinity?B:sy(y0),Y1=y1===Infinity?T:sy(y1);
      ctx.fillStyle=C.pareto;ctx.globalAlpha=.08;ctx.fillRect(X0,Y1,X1-X0,Y0-Y1);ctx.globalAlpha=1;
      ctx.strokeStyle=C.pareto;ctx.setLineDash([5,4]);ctx.strokeRect(X0,Y1,X1-X0,Y0-Y1);ctx.setLineDash([]);
      ctx.fillStyle=C.pareto;ctx.textAlign='right';ctx.textBaseline='bottom';ctx.fillText('目標領域',Math.min(X1,R)-6,Math.min(Y0,B)-5);
    }
  }
  const drawn=[];
  const xy=r=>[sx(ax.get(r)),sy(ay.get(r))];
  // literature training data
  const lit=literature().filter(r=>has(r,[ax,ay]));
  ctx.fillStyle=C.training;ctx.globalAlpha=.75;
  for(const r of lit){const[x,y]=xy(r);dot(x,y,2.6);ctx.fill();drawn.push({r,x,y,p:0})}
  ctx.globalAlpha=1;
  ctx.strokeStyle=C.ink;ctx.lineWidth=1.4;
  for(const r of lit)if(r.front){const[x,y]=xy(r);dot(x,y,5);ctx.stroke()}
  // selection: lines to the nearest training polymers
  const sel=state.selected&&byId.get(state.selected);
  if(sel&&isDesign(sel)&&$('showNeighbors').checked&&has(sel,[ax,ay])){
    const[x0,y0]=xy(sel);ctx.strokeStyle=C.muted;ctx.lineWidth=1;ctx.setLineDash([3,3]);
    for(const[id]of sel.neighbors){const n=byId.get(id);if(!has(n,[ax,ay]))continue;const[x,y]=xy(n);ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x,y);ctx.stroke()}
    ctx.setLineDash([]);
    for(const[id]of sel.neighbors){const n=byId.get(id);if(!has(n,[ax,ay]))continue;const[x,y]=xy(n);ctx.fillStyle=C.training;dot(x,y,4.5);ctx.fill();ctx.strokeStyle=C.ink;ctx.lineWidth=1.2;ctx.stroke()}
  }
  // designs: Pareto under next experiments
  const ds=visibleDesigns().filter(d=>has(d,[ax,ay]));
  ctx.lineWidth=1.5;ctx.strokeStyle=C.surface;
  for(const d of ds)if(!isNext(d)){const[x,y]=xy(d);ctx.fillStyle=C.pareto;dot(x,y,4.5);ctx.fill();ctx.stroke();drawn.push({r:d,x,y,p:2})}
  const next=ds.filter(isNext).sort((a,b)=>rank(b)-rank(a));
  for(const d of next){const[x,y]=xy(d);ctx.fillStyle=C.next;diamond(x,y,7);ctx.fill();ctx.strokeStyle=C.surface;ctx.stroke();drawn.push({r:d,x,y,p:3})}
  if($('showRank').checked){
    ctx.font='600 11px system-ui,sans-serif';ctx.textAlign='left';ctx.textBaseline='middle';ctx.lineJoin='round';
    // Labels that would overlap a placed label or another candidate are left out; zoom in to see them.
    const placed=next.map(d=>{const[x,y]=xy(d);return{x0:x-7,x1:x+7,y0:y-7,y1:y+7}});
    const hit=b=>placed.some(p=>b.x0<p.x1&&b.x1>p.x0&&b.y0<p.y1&&b.y1>p.y0);
    for(const d of [...next].reverse()){
      const[x,y]=xy(d),t='#'+rank(d),w=ctx.measureText(t).width;
      for(const[dx,dy]of[[9,-8],[9,8],[-9-w,-8],[-9-w,8]]){
        const b={x0:x+dx-1,x1:x+dx+w+1,y0:y+dy-7,y1:y+dy+7};if(hit(b))continue;
        placed.push(b);ctx.strokeStyle=C.surface;ctx.lineWidth=3;ctx.strokeText(t,x+dx,y+dy);ctx.fillStyle=C.ink;ctx.fillText(t,x+dx,y+dy);break;
      }
    }
    ctx.font='12px system-ui,sans-serif';
  }
  // in-house measurements
  for(const r of observations()){if(!has(r,[ax,ay]))continue;const[x,y]=xy(r);ctx.fillStyle=C.ink;star(x,y,8);ctx.fill();ctx.strokeStyle=C.surface;ctx.lineWidth=1;ctx.stroke();drawn.push({r,x,y,p:4})}
  // selection ring and its 90% interval
  if(sel&&has(sel,[ax,ay])){
    const[x,y]=xy(sel);ctx.strokeStyle=C.ink;
    if(isDesign(sel)&&!ax.map){
      ctx.lineWidth=1.5;
      const k=VIEWS[$('view').value];
      const lo=p=>sel[p+'_lo'],hi=p=>sel[p+'_hi'];
      const clipX=v=>Math.max(L-10,Math.min(R+10,v)),clipY=v=>Math.max(T-10,Math.min(B+10,v));
      const sxv=v=>ax.log&&v<=0?L-10:sx(v),syv=v=>ay.log&&v<=0?B+10:sy(v);
      if(lo(k[0])!=null){ctx.beginPath();ctx.moveTo(clipX(sxv(lo(k[0]))),y);ctx.lineTo(clipX(sxv(hi(k[0]))),y);ctx.stroke()}
      if(lo(k[1])!=null){ctx.beginPath();ctx.moveTo(x,clipY(syv(lo(k[1]))));ctx.lineTo(x,clipY(syv(hi(k[1]))));ctx.stroke()}
    }
    ctx.lineWidth=2.5;dot(x,y,10);ctx.stroke();
  }
  if(state.hover&&state.hover!==sel&&has(state.hover,[ax,ay])){const[x,y]=xy(state.hover);ctx.strokeStyle=C.ink;ctx.lineWidth=1.5;dot(x,y,8);ctx.stroke()}
  ctx.restore();
  state.drawn=drawn;
}

function nearest(mx,my){
  // Designs and in-house points are drawn on top, so they win over a slightly closer training point.
  let best=null,bd=HIT*HIT;
  for(const o of state.drawn){const d=(o.x-mx)**2+(o.y-my)**2-(o.p>0?40:0);if(d<bd){bd=d;best=o.r}}
  return best;
}

// ---------- tooltip ----------
function tipHtml(r){
  const vals=Object.keys(PROP).map(p=>`${PROP[p].label} ${fmt(p,r[p])}${r[p]!=null?' '+PROP[p].unit:''}`).join(' ・ ');
  if(isDesign(r)){
    const head=isNext(r)?`次の実験候補 #${r.next_ranks.join(', #')}`:'パレート解';
    const extra=[r.p_region!=null?`目標達成確率 ${pct(r.p_region)}`:'',r.p_beyond!=null?`前線を超える確率 ${pct(r.p_beyond)}`:'',`最も近い学習データとの類似度 ${r.nearest.toFixed(2)}`].filter(Boolean).join('<br>');
    return`<b>${head}(予測)</b>${esc(r.acids)}<br>${esc(r.diamines)}<br>${vals}<br>${extra}`;
  }
  return`<b>${kindJa(r)}(実測)</b>${r.label?esc(r.label)+'<br>':''}${vals}${r.front?'<br>実測の前線に乗る':''}`;
}
function showTooltip(r,mx,my){
  const t=$('tooltip');if(!r){t.hidden=true;return}
  t.innerHTML=tipHtml(r);t.hidden=false;
  const box=canvas.getBoundingClientRect();
  t.style.left=Math.min(mx+14,box.width-t.offsetWidth-4)+'px';
  t.style.top=(my-t.offsetHeight-10<4?my+16:my-t.offsetHeight-10)+'px';
}

// ---------- detail ----------
function valuesTable(r,predicted){
  return`<table class="vals">${Object.keys(PROP).map(p=>`<tr><td>${PROP[p].label}</td><td class="num"><b>${fmt(p,r[p])}</b></td><td>${PROP[p].unit}</td><td class="muted">${
    predicted?`90%区間 ${fmt(p,r[p+'_lo'])}〜${fmt(p,r[p+'_hi'])}`:r.n&&r.n[p]?`測定 ${r.n[p]}件`:''}</td></tr>`).join('')}</table>`;
}
// Monomers: dianhydrides and diamines with their mole fraction on their side.
function monomersHtml(r,compact){
  const ms=r.monomers;
  if(!ms||!ms.length)return`<p class="muted">${isDesign(r)?'モノマーの情報がありません。':'この繰り返し単位はイミド結合で酸二無水物とジアミンに分けられませんでした(アミドイミドなど)。'}</p>`;
  const side=(k,title)=>{const g=ms.filter(m=>m.side===k);if(!g.length)return'';
    return`<div class="mono-side"><div class="mono-title">${title}</div><div class="monos${compact?' compact':''}">${g.map(m=>`<figure class="mono">${m.svg?`<img src="structures/${m.svg}" alt="${esc(m.abbr||m.smiles)}" title="${esc(m.smiles)}" loading="lazy">`:''}<figcaption><b>${esc(m.abbr||'(略称なし)')}</b>${g.length>1||ms.filter(x=>x.side!==k).length>1?` ・${(+m.fraction).toFixed(2)}`:''}${!compact&&m.name&&m.name!==m.abbr?`<span>${esc(m.name)}</span>`:''}</figcaption></figure>`).join('')}</div></div>`};
  const html=side('acid','酸二無水物')+side('diamine','ジアミン');
  return compact?`<div class="mono-row">${html}</div>`:html;
}
const structs=r=>`<div class="structs">${r.svgs.map((s,i)=>`<div class="struct${r.svgs.length===1?' single':''}"><img src="structures/${s}" alt="${esc(r.units[i])}" title="${esc(r.units[i])}" loading="lazy"><div>モル分率 ${(+r.fractions[i]).toFixed(3)}</div></div>`).join('')}</div>`;
function renderDetail(){
  const box=$('detail'),r=state.selected&&byId.get(state.selected);
  if(!r){box.innerHTML='<h2>詳細</h2><p class="muted">点または表の行を選んでください。</p>';return}
  if(isDesign(r)){
    const b=[];
    if(r.pareto)b.push('<span class="badge">パレート解</span>');
    if(r.novelty)b.push(`<span class="badge">${esc(r.novelty_ja)}・組成の差 ${r.gap==null?'—':r.gap.toFixed(2)}</span>`);
    if(r.p_region!=null)b.push(`<span class="badge">目標達成確率 ${pct(r.p_region)}</span>`);
    if(r.similarity_to_training!=null)b.push(`<span class="badge">学習データとの類似度 ${r.similarity_to_training.toFixed(2)}</span>`);
    if(r.p_beyond!=null)b.push(`<span class="badge">実測の前線を超える確率 ${pct(r.p_beyond)}</span>`);
    if(r.beaten)b.push('<span class="badge warn">予測値は実測の前線に負けている</span>');
    if(r.extrapolated)b.push('<span class="badge warn">繰り返し単位の数が学習データより多い(外挿)</span>');
    const nb=r.neighbors.map(([id,s])=>{const n=byId.get(id);return`<div class="neighbor"><div class="head" data-id="${id}"><b>類似度 ${s.toFixed(2)}</b><span class="muted">${n.kind==='observation'?'追加した実測':n.kind==='copolymer'?'共重合体':'単独重合体'}${n.refs.length?' ・文献 '+esc(n.refs.join(', ')):''}</span></div>${n.label?`<div class="names">${esc(n.label)}</div>`:''}${valuesTable(n,false)}${monomersHtml(n,true)}<details><summary>繰り返し単位</summary>${structs(n)}</details></div>`}).join('');
    box.innerHTML=`<h2>${isNext(r)?`次の実験候補 #${r.next_ranks.join(', #')}`:'パレート解'}</h2>
      <div class="names">酸二無水物: ${esc(r.acids)}<br>ジアミン: ${esc(r.diamines)}</div>
      <div class="badges">${b.join('')}</div>
      <h3>予測値</h3>${valuesTable(r,true)}
      <h3>モノマー(数字は各側のモル分率)</h3>${monomersHtml(r,false)}
      <h3>繰り返し単位</h3>${structs(r)}
      <h3>構造の近い学習データ(実測値)</h3><p class="muted">「線で結ぶ」にチェックを入れると、グラフ上でも点線で結びます。</p>${nb}`;
  }else{
    const near=designs.map(d=>[d,d.neighbors.find(([id])=>id===r.id)]).filter(([,m])=>m).sort((a,b)=>b[1][1]-a[1][1]);
    box.innerHTML=`<h2>${kindJa(r)}</h2>${r.label?`<div class="names">${esc(r.label)}</div>`:''}
      ${r.refs.length?`<p class="muted">文献 ${esc(r.refs.join(', '))}</p>`:''}${r.front?'<div class="badges"><span class="badge">実測の前線に乗る</span></div>':''}
      <h3>実測値</h3>${valuesTable(r,false)}<h3>モノマー(繰り返し単位から推定)</h3>${monomersHtml(r,false)}<h3>繰り返し単位</h3>${structs(r)}
      <h3>${near.length?`この学習データを近傍に持つ設計(${near.length}件)`:'この学習データを近傍に持つ設計はありません'}</h3>
      ${near.slice(0,10).map(([d,[,s]])=>`<div class="neighbor"><div class="head" data-id="${d.id}"><b>${esc(d.acids)} + ${esc(d.diamines)}</b><span class="muted">類似度 ${s.toFixed(2)}</span></div></div>`).join('')}`;
  }
}

// ---------- table ----------
const near0=d=>byId.get(d.neighbors[0][0]);
function columns(){
  return[
    {key:'rank',label:'順位',num:1,get:d=>rank(d),show:d=>isNext(d)?d.next_ranks.map(x=>'#'+x).join(' '):'—'},
    {key:'acids',label:'酸二無水物',get:d=>d.acids,wrap:1},
    {key:'diamines',label:'ジアミン',get:d=>d.diamines,wrap:1},
    ...(isRegion?[{key:'p',label:'目標達成確率',num:1,get:d=>d.p_region,show:d=>pct(d.p_region)},
                  {key:'sim',label:'学習データとの類似度',num:1,get:d=>d.similarity_to_training,show:d=>d.similarity_to_training?.toFixed(2)??'—'}]:[]),
    ...(designs.some(d=>d.p_beyond!=null)?[{key:'pb',label:'前線を超える確率',num:1,get:d=>d.p_beyond,show:d=>pct(d.p_beyond)}]:[]),
    ...(designs.some(d=>d.novelty)&&!isRegion?[{key:'nov',label:'新規性',get:d=>d.novelty_ja},{key:'gap',label:'組成の差',num:1,get:d=>d.gap,show:d=>d.gap==null?'—':d.gap.toFixed(2)}]:[]),
    {key:'Tg',label:'Tg 予測 (℃)',num:1,get:d=>d.Tg,show:d=>fmt('Tg',d.Tg)},
    {key:'modulus',label:'弾性率 予測 (GPa)',num:1,get:d=>d.modulus,show:d=>fmt('modulus',d.modulus)},
    {key:'CTE',label:'CTE 予測 (ppm/K)',num:1,get:d=>d.CTE,show:d=>fmt('CTE',d.CTE)},
    {key:'near',label:'最近傍の類似度',num:1,get:d=>d.nearest,show:d=>d.nearest.toFixed(2)},
    {key:'nTg',label:'最近傍の実測 Tg',num:1,get:d=>near0(d).Tg,show:d=>fmt('Tg',near0(d).Tg)},
    {key:'nMod',label:'弾性率',num:1,get:d=>near0(d).modulus,show:d=>fmt('modulus',near0(d).modulus)},
    {key:'nCte',label:'CTE',num:1,get:d=>near0(d).CTE,show:d=>fmt('CTE',near0(d).CTE)},
  ];
}
function renderTable(){
  const cols=columns(),col=cols.find(c=>c.key===state.sortKey)||cols[0];
  const pool=visibleDesigns().filter(d=>!$('nextOnly').checked||isNext(d));
  const rows=[...pool].sort((a,b)=>{const x=col.get(a),y=col.get(b);if(x==null||x===Infinity)return 1;if(y==null||y===Infinity)return -1;return(x<y?-1:x>y?1:0)*state.sortDir});
  $('table').tHead.innerHTML='<tr>'+cols.map(c=>`<th class="${c.num?'num':''}" aria-sort="${col.key===c.key?(state.sortDir>0?'ascending':'descending'):'none'}"><button type="button" data-key="${c.key}">${c.label}${col.key===c.key?(state.sortDir>0?' ▲':' ▼'):''}</button></th>`).join('')+'</tr>';
  $('table').tBodies[0].innerHTML=rows.map(d=>`<tr tabindex="0" data-id="${d.id}" class="${state.selected===d.id?'selected':''}">`+cols.map((c,i)=>{
    const v=c.show?c.show(d):c.get(d);
    const key=i===0?`<i class="key${isNext(d)?' diamond':''}" style="background:var(--${isNext(d)?'next':'pareto'})"></i>`:'';
    return`<td class="${c.num?'num':c.wrap?'wrap':''}">${key}${esc(v??'—')}</td>`}).join('')+'</tr>').join('');
  $('tableTitle').textContent=`${$('nextOnly').checked?'次の実験候補':'設計一覧'}(${rows.length}件)`;
  $('tableNote').textContent='「最近傍」は組成指紋のTanimoto類似度が最も高い学習データ(実測)です。'+(meta.next_duplicates>0?` 次の実験候補の${meta.n_next}行には並び順だけ違う同じ組成があったため、まとめています。`:'');
}

// ---------- header parts ----------
function renderTiles(){
  const t=[];const add=(b,s)=>t.push(`<div class="tile"><b>${esc(b)}</b><span>${esc(s)}</span></div>`);
  const next=designs.filter(isNext);
  if(isRegion)add(regionText(),'目標領域');
  else if(meta.constraint)add(regionText(meta.constraint)||meta.constraint.replace(/[\[\]']/g,''),'最適化の制約');
  add(next.length,'次の実験候補');
  add(designs.filter(d=>d.pareto).length,'パレート解');
  if(isRegion){
    if(next.length)add(pct(Math.max(...next.map(d=>d.p_region))),'候補1件あたりの目標達成確率(最大)');
    if(meta.p_at_least_one!=null)add(pct(meta.p_at_least_one),'候補のどれか1つが達成する確率');
  }
  if(meta.n_front)add(meta.n_front,'実測の前線に乗る既存ポリマー');
  add(meta.n_training.toLocaleString(),`学習データ(Tg ${meta.n_measured.Tg} / 弾性率 ${meta.n_measured.modulus} / CTE ${meta.n_measured.CTE})`);
  $('tiles').innerHTML=t.join('');
}
function renderLegend(){
  const [ax,ay]=axes(),n=literature().filter(r=>has(r,[ax,ay])).length;
  const svg=inner=>`<svg width="14" height="14" viewBox="0 0 14 14">${inner}</svg>`;
  const items=[];
  if($('showTraining').checked)items.push(`${svg('<circle cx="7" cy="7" r="3" fill="var(--training)"/>')}学習データ(実測)${n.toLocaleString()}件`);
  if(meta.n_front&&$('showTraining').checked)items.push(`${svg('<circle cx="7" cy="7" r="5" fill="var(--training)" stroke="var(--ink)" stroke-width="1.4"/>')}実測の前線`);
  if(meta.n_observations)items.push(`${svg('<path d="M7 0.5 L8.8 5 L13.5 5.2 L9.8 8.2 L11 13 L7 10.3 L3 13 L4.2 8.2 L0.5 5.2 L5.2 5 Z" fill="var(--ink)"/>')}追加した実測`);
  if($('designFilter').value==='all')items.push(`${svg('<circle cx="7" cy="7" r="5" fill="var(--pareto)"/>')}パレート解(予測)`);
  if($('designFilter').value!=='none')items.push(`${svg('<path d="M7 0 L14 7 L7 14 L0 7 Z" fill="var(--next)"/>')}次の実験候補(予測)`);
  if(isRegion&&!ax.map)items.push(`${svg('<rect x="1" y="1" width="12" height="12" fill="var(--pareto)" fill-opacity=".12" stroke="var(--pareto)" stroke-dasharray="3 2"/>')}目標領域`);
  $('legend').innerHTML=items.map(s=>`<span>${s}</span>`).join('');
  const view=$('view').value;
  $('chartTitle').textContent=view==='map'?'構造マップ':`${PROP[VIEWS[view][0]].label} と ${PROP[VIEWS[view][1]].label}`;
  $('chartHint').textContent=view==='map'
    ?'組成で重み付けしたMorganカウント指紋のTanimoto距離をt-SNEで2次元にしたものです。近い点ほど構造が似ています。軸の値に意味はありません。'
    :'学習データは2つの物性が両方とも実測されているものだけです。点を選ぶと、設計には90%予測区間の線が出ます。「線で結ぶ」にチェックを入れると、構造の近い学習データとも点線で結びます。ホイールで拡大、ドラッグで移動できます。';
}
function renderNotes(){
  const n=['<strong>設計の値はモデルの予測値</strong>、学習データの値は実測値(文献)です。',
    '<strong>類似度</strong>は組成で重み付けしたMorganカウント指紋のTanimoto類似度です(最適化のガウス過程と同じ指紋)。低いほど学習データから遠く、予測の信頼性は下がります。',
    '90%予測区間はモデルの誤差を含む幅です。CTEのように実測の少ない物性では区間が非常に広くなります。'];
  if(isRegion)n.push('<strong>目標達成確率</strong>は、予測の誤差を正規分布と見なしたときに目標領域に入る確率です。「どれか1つが達成する確率」はモデルが平均的に正しいと仮定した楽観的な値です。');
  if(meta.n_front)n.push('<strong>実測の前線</strong>は、学習データと追加した実測のうち、最適化の目的で他に負けていないポリマーです。候補はこれを超える必要があります。');
  $('notes').innerHTML=n.map(s=>`<li>${s}</li>`).join('');
}

// ---------- interaction ----------
function select(id){
  state.selected=state.selected===id?null:id;
  renderDetail();draw();
  for(const tr of $('table').tBodies[0].rows)tr.classList.toggle('selected',tr.dataset.id===state.selected);
}
function refresh(resetView){if(resetView)state.view=fullView();renderLegend();draw();renderTable()}

function init(){
  renderTiles();renderNotes();
  $('view').addEventListener('input',()=>refresh(true));
  for(const id of['designFilter','showTraining'])$(id).addEventListener('input',()=>refresh(true));
  $('search').addEventListener('input',()=>refresh(false));
  $('showRank').addEventListener('input',draw);
  $('showNeighbors').addEventListener('input',draw);
  $('nextOnly').addEventListener('input',renderTable);
  $('resetZoom').addEventListener('click',()=>{state.view=fullView();draw()});
  $('table').tHead.addEventListener('click',e=>{const key=e.target.closest('button')?.dataset.key;if(!key)return;
    if(state.sortKey===key)state.sortDir*=-1;else{state.sortKey=key;const c=columns().find(c=>c.key===key);state.sortDir=c.num&&key!=='rank'?-1:1}renderTable()});
  const pick=e=>{const tr=e.target.closest('tr[data-id]');if(tr)select(tr.dataset.id)};
  $('table').tBodies[0].addEventListener('click',pick);
  $('table').tBodies[0].addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick(e)}});
  $('detail').addEventListener('click',e=>{const h=e.target.closest('.head[data-id]');if(h)select(h.dataset.id)});
  let drag=null,moved=false,frame=0;
  const pos=e=>{const b=canvas.getBoundingClientRect();return[e.clientX-b.left,e.clientY-b.top]};
  canvas.addEventListener('pointerdown',e=>{drag=pos(e);moved=false;canvas.setPointerCapture(e.pointerId)});
  canvas.addEventListener('pointermove',e=>{
    const[mx,my]=pos(e);
    if(drag){const b=canvas.getBoundingClientRect(),v=state.view,dx=(mx-drag[0])/(b.width-PAD.l-PAD.r)*(v.x1-v.x0),dy=(my-drag[1])/(b.height-PAD.t-PAD.b)*(v.y1-v.y0);
      if(Math.abs(mx-drag[0])+Math.abs(my-drag[1])>2)moved=true;
      state.view={x0:v.x0-dx,x1:v.x1-dx,y0:v.y0+dy,y1:v.y1+dy};drag=[mx,my];$('tooltip').hidden=true;draw();return}
    if(frame)return;
    frame=requestAnimationFrame(()=>{frame=0;const h=nearest(mx,my);if(h!==state.hover){state.hover=h;draw()}showTooltip(h,mx,my)});
  });
  canvas.addEventListener('pointerup',e=>{if(!moved){const[mx,my]=pos(e),h=nearest(mx,my);if(h)select(h.id)}drag=null});
  canvas.addEventListener('pointerleave',()=>{$('tooltip').hidden=true;state.hover=null;draw()});
  canvas.addEventListener('wheel',e=>{e.preventDefault();
    const[mx,my]=pos(e),b=canvas.getBoundingClientRect(),v=state.view,k=e.deltaY>0?1.2:1/1.2;
    const fx=(mx-PAD.l)/(b.width-PAD.l-PAD.r),fy=1-(my-PAD.t)/(b.height-PAD.t-PAD.b);
    const cx=v.x0+fx*(v.x1-v.x0),cy=v.y0+fy*(v.y1-v.y0);
    state.view={x0:cx-(cx-v.x0)*k,x1:cx+(v.x1-cx)*k,y0:cy-(cy-v.y0)*k,y1:cy+(v.y1-cy)*k};draw()},{passive:false});
  addEventListener('resize',draw);
  matchMedia('(prefers-color-scheme:dark)').addEventListener('change',draw);
  refresh(true);
  const first=designs.filter(isNext).sort((a,b)=>rank(a)-rank(b))[0];
  if(first)select(first.id);
}
init();
