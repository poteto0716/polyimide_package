#!/usr/bin/env python3
"""Find structures associated with lower film modulus and higher Tg.

A: fraction-weighted substructure regression on measured film data, with family-cluster
   bootstrap intervals, repeated without references whose values look suspect.
B: the same regression on the virtual predictions, as a model cross-check.
C: monomer effects in the virtual predictions, adjusted for the partner monomer.

 .venv/bin/python polyimide_virtual_app/analyze_structure_effects.py
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
MODELS=ROOT.parent/'polyimide_models'
OUT=ROOT/'structure_effects'
SUSPECT_REFS=set(json.loads((MODELS/'review_references.json').read_text())['suspect'])
N_BOOT=1000
MIN_ROWS=8
NEAR=.4
SEED=20261031
FRAGMENTS={
 'ペンダントフェニル':'[cH1]1[cH1][cH1][cH1][cH1]c1-!@[#6,#7]',
 'tert-ブチル':'[CH3]C([CH3])([CH3])c',
 'トリメチルシリル':'[Si]([CH3])([CH3])[CH3]',
 'CF3(芳香環上)':'c-C(F)(F)F',
 '6F(ヘキサフルオロイソプロピリデン)':'C(C(F)(F)F)(C(F)(F)F)(c)',
 'イソプロピリデン':'[CH3]C([CH3])(c)c',
 'フルオレン(カルド)':'C1(c)(c)c2ccccc2-c2ccccc21',
 'アダマンタン':'C1C2CC3CC1CC(C2)C3',
 'スピロ炭素':'[CX4;x4;R2]',
 '芳香環メチル':'[CH3]c',
 'ジアリールエーテル':'c-[OX2]-c',
 'ジアリールケトン':'c-C(=O)-c',
 'スルホン':'c-S(=O)(=O)-c',
 'スルフィド':'c-[SX2]-c',
 'エステル':'c-C(=O)-O-c',
 'アミド':'C(=O)-[NH1]',
 'アルキル鎖(C3以上)':'[CH2][CH2][CH2]',
 '脂環':'[CX4;R;!$(C(F))]',
 'ビフェニル結合':'[c;R1]-!@[c;R1]',
 'トリアリールアミン':'[NX3;R0;H0](c)(c)c',
 'ピリジン・ジアジン環':'[n;r6]',
 'オキサジアゾール':'c1nnco1',
 'キノキサリン':'c1cnc2ccccc2n1',
 'シロキサン':'[Si]-O-[Si]',
 'メタ置換ベンゼン':'[!#1]-!@c1[cH]c(-!@[!#1])[cH][cH][cH]1',
 'パラ置換ベンゼン':'[!#1]-!@c1[cH][cH]c(-!@[!#1])[cH][cH]1',
 'ハロゲン(Cl, Br)':'[Cl,Br]',
}
QUERIES={k:Chem.MolFromSmarts(v) for k,v in FRAGMENTS.items()}
assert all(QUERIES.values())
NAMES=list(QUERIES)
IMIDE=Chem.MolFromSmarts('[#7;R]([#6;R](=O))[#6;R](=O)')

def unit_features(smiles,cache={}):
    """Fragment presence plus imide density, the covariate for repeat-unit length."""
    if smiles not in cache:
        mol=Chem.MolFromSmiles(smiles)
        heavy=sum(a.GetAtomicNum()>0 for a in mol.GetAtoms())
        cache[smiles]=np.array([float(mol.HasSubstructMatch(QUERIES[n])) for n in NAMES]+[len(mol.GetSubstructMatches(IMIDE))/heavy*10])
    return cache[smiles]

def design(components,fractions):
    rows=[]
    for c,f in zip(components,fractions):
        c,f=json.loads(c),np.array(json.loads(f),dtype=float)
        rows.append(sum(unit_features(s)*w for s,w in zip(c,f/f.sum())))
    return np.array(rows)

def fit(X,y):
    A=np.c_[np.ones(len(X)),X]
    return np.linalg.lstsq(A,y,rcond=None)[0][1:]

def cluster_interval(X,y,groups,rng):
    """Resample whole structure families so related polymers do not inflate confidence."""
    labels,inverse=np.unique(groups,return_inverse=True)
    members=[np.where(inverse==i)[0] for i in range(len(labels))]
    boots=[]
    for _ in range(N_BOOT):
        take=np.concatenate([members[i] for i in rng.integers(0,len(labels),len(labels))])
        boots.append(fit(X[take],y[take]))
    return np.percentile(boots,[2.5,97.5],axis=0)

def load(target,drop_suspect):
    d=pd.read_csv(MODELS/f'{target}_dataset.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
    if drop_suspect:d=d[[not(set(json.loads(r))&SUSPECT_REFS) for r in d.reference_nos]]
    return d

def measured_effects(drop_suspect,rng):
    tg,mo=load('Tg_film',drop_suspect),load('tensile_modulus_film_clean',drop_suspect)
    Xt,Xm=design(tg.components,tg.mole_fractions),design(mo.components,mo.mole_fractions)
    yt,ym=tg.value.astype(float).to_numpy(),np.log10(mo.value.astype(float).to_numpy())
    bt,bm=fit(Xt,yt),fit(Xm,ym)
    ct,cm=cluster_interval(Xt,yt,tg.family_group.to_numpy(),rng),cluster_interval(Xm,ym,mo.family_group.to_numpy(),rng)
    rows=[]
    for i,n in enumerate(NAMES):
        rows.append({'fragment':n,'Tg_rows':int((Xt[:,i]>0).sum()),'modulus_rows':int((Xm[:,i]>0).sum()),
            'dTg_C':bt[i],'dTg_lo':ct[0,i],'dTg_hi':ct[1,i],
            'modulus_pct':100*(10**bm[i]-1),'modulus_lo':100*(10**cm[0,i]-1),'modulus_hi':100*(10**cm[1,i]-1)})
    return pd.DataFrame(rows),len(tg),len(mo)

def main():
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    full,nt,nm=measured_effects(False,rng)
    clean,nt2,nm2=measured_effects(True,rng)
    virtual=pd.read_csv(ROOT/'virtual_predictions.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
    T,G='Tg_film_pred','tensile_modulus_film_clean_pred'
    for c in(T,G,'Tg_film_max_similarity','tensile_modulus_film_clean_max_similarity'):virtual[c]=virtual[c].astype(float)
    virtual['similarity']=virtual[['Tg_film_max_similarity','tensile_modulus_film_clean_max_similarity']].min(axis=1)
    Xv=np.array([unit_features(s) for s in virtual.psmiles])
    bT,bG=fit(Xv,virtual[T].to_numpy()),fit(Xv,np.log10(virtual[G].to_numpy()))
    table=full.merge(clean,on='fragment',suffixes=('','_clean'))
    table['virtual_dTg_C']=bT[:len(NAMES)];table['virtual_modulus_pct']=100*(10**bG[:len(NAMES)]-1);table['virtual_rows']=(Xv[:,:len(NAMES)]>0).sum(axis=0)
    table=table[(table.Tg_rows>=MIN_ROWS)&(table.modulus_rows>=MIN_ROWS)]
    table.to_csv(OUT/'fragment_effects.tsv',sep='\t',index=False,encoding='utf-8-sig')
    print('rows full Tg/G',nt,nm,'clean',nt2,nm2)
    show=['fragment','Tg_rows','modulus_rows','dTg_C','dTg_lo','dTg_hi','modulus_pct','modulus_lo','modulus_hi','dTg_C_clean','modulus_pct_clean','modulus_lo_clean','modulus_hi_clean','virtual_dTg_C','virtual_modulus_pct']
    print(table[show].round(1).sort_values('modulus_pct').to_string(index=False))
    # Monomer effects: mean over the partner monomers, near-training rows only, partner-adjusted.
    near=virtual[virtual.similarity>=NEAR].copy();near['logG']=np.log10(near[G])
    results=[]
    for column,partner,label,name in(('diamine_id','dianhydride_id','diamine','diamine_name'),('dianhydride_id','diamine_id','dianhydride','dianhydride_name')):
        for y in(T,'logG'):
            near[y+'_adj_'+label]=near[y]-near.groupby(partner)[y].transform('mean')
        effect=near.groupby(column).agg(name=(name,'first'),n=(T,'size'),dTg_C=(T+'_adj_'+label,'mean'),dlogG=('logG_adj_'+label,'mean'),similarity=('similarity','mean'),Tg_mean=(T,'mean'),G_mean=(G,'mean')).reset_index().rename(columns={column:'id'})
        effect['type']=label;effect['modulus_pct']=100*(10**effect.dlogG-1)
        results.append(effect)
    monomers=pd.concat(results)
    monomers.to_csv(OUT/'monomer_effects.tsv',sep='\t',index=False,encoding='utf-8-sig')
    for label,min_n in(('diamine',10),('dianhydride',20)):
        m=monomers[(monomers.type==label)&(monomers.n>=min_n)]
        hit=m[(m.dTg_C>0)&(m.modulus_pct<0)].copy()
        hit['score']=hit.dTg_C/m.dTg_C.std()-hit.modulus_pct/m.modulus_pct.std()
        print('\n',label,'evaluated',len(m),'quadrant',len(hit),'corr',round(m.dTg_C.corr(m.modulus_pct),2))
        print(hit.sort_values('score',ascending=False)[['id','name','n','dTg_C','modulus_pct','similarity','Tg_mean','G_mean']].head(14).round(1).to_string(index=False))
    print('\nTg-log modulus correlation, measured pairs and virtual')
    tg,mo=load('Tg_film',False),load('tensile_modulus_film_clean',False)
    pair=tg.merge(mo,on='chemistry_key');print(round(np.corrcoef(pair.value_x.astype(float),np.log10(pair.value_y.astype(float)))[0,1],2),len(pair),round(np.corrcoef(virtual[T],np.log10(virtual[G]))[0,1],2))

if __name__=='__main__':main()
