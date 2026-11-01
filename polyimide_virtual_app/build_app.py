#!/usr/bin/env python3
"""Package predictions, measured data, structure drawings and the map image for the web app.

 .venv/bin/python polyimide_virtual_app/build_app.py
 .venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory polyimide_virtual_app/dist
"""
import json
from multiprocessing import Pool
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
ROOT=Path(__file__).resolve().parent
DIST=ROOT/'dist'
BASE=ROOT.parent
MODELS=BASE/'polyimide_models'
MONOMERS=BASE/'tci_polyimide/output'
SETS={'film':('Tg_film','tensile_modulus_film_clean')}
CTE='CTE_film'
NEAR_SIMILARITY=0.4
PNG_LABELS=8
FONT='/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf'
INK,MUTED,GRID,BLUE,ORANGE,GRAY='#0b0b0b','#52514e','#e4e3df','#2a78d6','#eb6834','#b9b8b2'

def read_tsv(path):
    return pd.read_csv(path,sep='\t',encoding='utf-8-sig',keep_default_na=False)

def draw(job):
    path,smiles,width,height=job
    mol=Chem.MolFromSmiles(smiles)
    drawer=rdMolDraw2D.MolDraw2DSVG(width,height)
    options=drawer.drawOptions();options.clearBackground=False;options.bondLineWidth=1.5;options.dummiesAreAttachments=False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol)
    drawer.FinishDrawing()
    svg=drawer.GetDrawingText().split('?>',1)[-1].strip()
    path.write_text(svg)

def monomers(name,prefix):
    frame=pd.read_csv(MONOMERS/name,encoding='utf-8-sig',keep_default_na=False)
    return {row.smiles:{'key':f'{prefix}{i}','id':row.tci_product_numbers,'name':row['name'],'smiles':row.smiles,'cas':row.cas_rn,
        'type':row.type_detail,'flags':row['flags'],'date':row.pubchem_deposit_date,'url':row.product_url.split(' ')[0]}
        for i,(_,row) in enumerate(frame.iterrows())}

def measured_rows(tg_target,modulus_target):
    """Same composition measured for both targets; values are the dataset medians."""
    tg,modulus=read_tsv(MODELS/f'{tg_target}_dataset.tsv'),read_tsv(MODELS/f'{modulus_target}_dataset.tsv')
    merged=tg.merge(modulus,on='chemistry_key',suffixes=('_tg','_mod'))
    return merged

def measured_entry(m,names,units):
    """Values for the plot, then composition and provenance for the detail panel."""
    components=json.loads(m.components_tg)
    ids=[units.setdefault(s,len(units)) for s in components]
    samples=json.loads(m.sample_ids_mod)+json.loads(m.sample_ids_tg)
    labels=list(dict.fromkeys(names[s] for s in samples if names.get(s)))
    return [float(m.value_tg),float(m.value_mod),m.polymer_kind_tg,int(m.measurement_count_tg),int(m.measurement_count_mod),
            {'units':ids,'fractions':json.loads(m.mole_fractions_tg),'sequence':m.sequence_type_tg,'names':labels[:3],
             'refsTg':json.loads(m.reference_nos_tg),'refsY':json.loads(m.reference_nos_mod),
             'tgRange':[float(m.value_min_tg),float(m.value_max_tg)],'yRange':[float(m.value_min_mod),float(m.value_max_mod)]}]

def known_lookup(target):
    frame=read_tsv(MODELS/f'{target}_dataset.tsv')
    frame=frame[frame.polymer_kind=='homopolymer']
    return {json.loads(row.components)[0]:float(row.value) for _,row in frame.iterrows()}

def metrics():
    result={}
    for target in (*SETS['film'],CTE):
        report=json.loads((MODELS/target/'oof10_results.json').read_text())
        result[target]={'r2':report['ensemble_oof']['R2'],'mae':report['ensemble_oof']['MAE'],'n':report['rows']}
    return result

def pareto_mask(x,y):
    mask=np.zeros(len(x),dtype=bool);best=-np.inf
    for i in np.argsort(-x,kind='stable'):
        if y[i]>best:mask[i]=True;best=y[i]
    return mask

def render_png(frame,measured,column,similarity,ylabel,title,log_axis,measured_label,path):
    """Static map toward high Tg and low y. Labels only Pareto points near training data."""
    font=FontProperties(fname=FONT)
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11,7.4),dpi=170)
    x,y=frame.Tg_film_pred.to_numpy(),frame[column].to_numpy()
    near=frame[['Tg_film_max_similarity',similarity]].min(axis=1).to_numpy()>=NEAR_SIMILARITY
    ax.scatter(x[~near],y[~near],s=7,c=GRAY,alpha=.45,linewidths=0,label=f'予測・学習データから遠い(類似度{NEAR_SIMILARITY}未満): {(~near).sum():,}件',rasterized=True)
    ax.scatter(x[near],y[near],s=9,c=BLUE,alpha=.65,linewidths=0,label=f'予測・学習データに近い(類似度{NEAR_SIMILARITY}以上): {near.sum():,}件',rasterized=True)
    mx,my=measured.value_tg.astype(float),measured.value_mod.astype(float)
    ax.scatter(mx,my,s=24,marker='^',c=ORANGE,edgecolors='white',linewidths=.5,label=f'{measured_label}: {len(measured):,}件')
    near_frame=frame[near];mask=pareto_mask(near_frame.Tg_film_pred.to_numpy(),-near_frame[column].to_numpy())
    top=near_frame[mask].sort_values('Tg_film_pred');ax.plot(top.Tg_film_pred,top[column],color=INK,lw=1,zorder=4,drawstyle='steps-pre',label='パレート前線(学習データに近い予測)')
    ax.scatter(top.Tg_film_pred,top[column],s=30,facecolors='none',edgecolors=INK,linewidths=1.1,zorder=5)
    shown=top.iloc[np.unique(np.linspace(0,len(top)-1,min(PNG_LABELS,len(top))).astype(int))]
    for _,row in shown.iterrows():
        ax.annotate(f'{row.dianhydride_id.split(";")[0]} + {row.diamine_id.split(";")[0]}',(row.Tg_film_pred,row[column]),xytext=(5,6),textcoords='offset points',fontsize=8,color=INK,fontproperties=font)
    ax.set_yscale('log' if log_axis else 'linear');ax.grid(color=GRID,lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel('Tg (℃)',fontproperties=font,fontsize=12,color=MUTED);ax.set_ylabel(ylabel,fontproperties=font,fontsize=12,color=MUTED)
    ax.set_title(title,fontproperties=font,fontsize=14,loc='left',color=INK)
    ax.legend(prop=font,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.09),ncols=2,markerscale=1.4)
    fig.text(.01,.01,'製品番号はPubChem登録情報。現在の販売状況は未確認。予測は構造のみに基づく参考値。',fontproperties=font,fontsize=9,color=MUTED)
    fig.tight_layout(rect=(0,.03,1,1))
    fig.savefig(DIST/'assets'/path);plt.close(fig)

def main():
    (DIST/'assets').mkdir(parents=True,exist_ok=True)
    structures=DIST/'structures';structures.mkdir(exist_ok=True)
    frame=read_tsv(ROOT/'virtual_predictions.tsv')
    numeric=[c for c in frame if c.endswith(('_pred','_member_std','_max_similarity'))]
    frame[numeric]=frame[numeric].apply(pd.to_numeric)
    dianhydrides,diamines=monomers('tci_dianhydrides.csv','d'),monomers('tci_diamines.csv','a')
    known={t:known_lookup(t) for pair in SETS.values() for t in pair}
    known[CTE]=known_lookup(CTE)
    rows=[]
    for i,r in enumerate(frame.itertuples()):
        row={'i':i,'d':dianhydrides[r.dianhydride_smiles]['key'],'a':diamines[r.diamine_smiles]['key'],'p':r.psmiles,'alt':bool(r.alt_psmiles),'mw':float(r.repeat_unit_mw)}
        for name,(tg,modulus) in SETS.items():
            row[name]=[round(float(getattr(r,tg+'_pred')),1),round(float(getattr(r,modulus+'_pred')),3),
                       round(float(getattr(r,tg+'_member_std')),1),round(float(getattr(r,modulus+'_member_std')),3),
                       round(min(getattr(r,tg+'_max_similarity'),getattr(r,modulus+'_max_similarity')),3),
                       known[tg].get(r.psmiles),known[modulus].get(r.psmiles)]
        row['cte']=[round(float(r.CTE_film_pred),1),round(float(r.CTE_film_member_std),1),round(min(r.Tg_film_max_similarity,r.CTE_film_max_similarity),3),known[CTE].get(r.psmiles)]
        rows.append(row)
    measured={}
    pairs={}
    samples=pd.read_csv(BASE/'polyimide_tg_modulus/polyimide_samples.tsv',sep='\t',keep_default_na=False,low_memory=False)
    names=dict(zip(samples.sample_id,samples.reference_polymer_name))
    units={}
    for name,(tg,modulus) in SETS.items():
        merged=measured_rows(tg,modulus);pairs[name]=merged
        measured[name]=[measured_entry(m,names,units) for m in merged.itertuples()]
    measured['cte']=[measured_entry(m,names,units) for m in measured_rows('Tg_film',CTE).itertuples()]
    summary=metrics()
    payload={'near':NEAR_SIMILARITY,'metrics':{name:{'Tg':summary[t],'modulus':summary[m],'cte':summary[CTE]} for name,(t,m) in SETS.items()},
             'dianhydrides':{m['key']:m for m in dianhydrides.values()},'diamines':{m['key']:m for m in diamines.values()},'rows':rows,'measured':measured,'unitSmiles':list(units)}
    (DIST/'data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    jobs=[(structures/f"{m['key']}.svg",m['smiles'],300,220) for m in [*dianhydrides.values(),*diamines.values()]]
    jobs+=[(structures/f'p{i}.svg',s,560,300) for i,s in enumerate(frame.psmiles)]
    jobs+=[(structures/f'u{i}.svg',s,560,300) for s,i in units.items()]
    with Pool() as pool:pool.map(draw,jobs,chunksize=100)
    render_png(frame,pairs['film'],'tensile_modulus_film_clean_pred','tensile_modulus_film_clean_max_similarity','引張弾性率 (GPa, 対数軸)','高Tg・低弾性に向けた仮想ポリイミドと既存実測値(フィルムモデル)',True,'実測(フィルム, 監査済み弾性率)','virtual_map.png')
    render_png(frame,measured_rows('Tg_film',CTE),'CTE_film_pred','CTE_film_max_similarity','CTE (ppm/℃)','高Tg・低CTEに向けた仮想ポリイミドと既存実測値(フィルムモデル)',False,'実測(フィルム, CTE)','virtual_map_cte.png')
    print(len(rows),'polymers;',{k:len(v) for k,v in measured.items()},'measured;',round(sum(j[0].stat().st_size for j in jobs)/1e6,1),'MB structures;',round((DIST/'data.json').stat().st_size/1e6,1),'MB data')

if __name__=='__main__':main()
