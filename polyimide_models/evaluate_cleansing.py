#!/usr/bin/env python3
"""Compare modulus cleansing rules: only training labels differ, test rows are shared.

Each variant trains on its own cleansed rows inside the same family-grouped folds. All variants
are scored on the same held-out rows, against the raw labels and against the strictest labels.
 .venv/bin/python polyimide_models/evaluate_cleansing.py
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from train_models import make_model,forward,inverse,SEED
ROOT=Path(__file__).resolve().parent
TARGET='tensile_modulus_film_range'
RULES={'raw':[],
 'method':['non_tensile_method','crosslinked'],
 'physical':['non_tensile_method','crosslinked','secant_inconsistent','strength_implausible'],
 'physical+outlier':['non_tensile_method','crosslinked','secant_inconsistent','strength_implausible','structure_outlier'],
 'all':['non_tensile_method','crosslinked','secant_inconsistent','strength_implausible','structure_outlier','low_modulus_review']}
SEEDS=[SEED+10,SEED+11,SEED+12]
N_BOOT=2000

def labels(audit,rows,flags):
    kept=audit[~audit[flags].any(axis=1)] if flags else audit
    median=kept.groupby('row_id').modulus.median()
    return np.array([median.get(r,np.nan) for r in rows])

def score(y,p):
    e=p-y
    return {'MAE':np.abs(e).mean(),'RMSE':np.sqrt((e**2).mean()),'R2':1-(e**2).sum()/((y-y.mean())**2).sum(),'log10RMSE':np.sqrt(((np.log10(np.maximum(p,1e-3))-np.log10(y))**2).mean())}

def main():
    ds=json.loads((ROOT/'datasets.json').read_text())[TARGET]
    arrays=np.load(ROOT/'features.npz')
    selection=json.loads((ROOT/TARGET/'frozen_selection.json').read_text())
    members=selection['members'];weights=np.array([m['weight'] for m in members])
    X={m['feature_set']:arrays[TARGET+'__'+m['feature_set']] for m in members}
    audit=pd.read_csv(ROOT/'tensile_modulus_film_audit.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
    for f in RULES['all']:audit[f]=audit[f].astype(str)=='True'
    rows=[r['row_id'] for r in ds];groups=np.array([r['family_group'] for r in ds])
    y={name:labels(audit,rows,flags) for name,flags in RULES.items()}
    strict=y['all'];raw=y['raw']
    assert np.isfinite(raw).all()
    print({name:{'rows':int(np.isfinite(v).sum()),'changed':int((np.isfinite(v)&(v!=raw)).sum())} for name,v in y.items()},flush=True)
    results=[];predictions={name:[] for name in RULES}
    for seed in SEEDS:
        preds={name:np.full(len(ds),np.nan) for name in RULES}
        for fold,(train,test) in enumerate(GroupKFold(n_splits=10,shuffle=True,random_state=seed).split(raw,groups=groups)):
            for name in RULES:
                use=train[np.isfinite(y[name][train])]
                stack=[]
                for m in members:
                    model=make_model(m['model'],m['variant'])
                    model.fit(X[m['feature_set']][use],forward(y[name][use],m['target_transform']))
                    stack.append(inverse(model.predict(X[m['feature_set']][test]),m['target_transform'],TARGET))
                preds[name][test]=np.array(stack).T@weights
            print('seed',seed,'fold',fold+1,flush=True)
        for name in RULES:
            predictions[name].append(preds[name])
            common=np.isfinite(strict)
            results.append({'seed':seed,'variant':name,'test':'same_rows_raw_labels',**score(raw[common],preds[name][common])})
            results.append({'seed':seed,'variant':name,'test':'same_rows_strict_labels',**score(strict[common],preds[name][common])})
            own=np.isfinite(y[name])
            results.append({'seed':seed,'variant':name,'test':'own_labels_not_comparable','n':int(own.sum()),**score(y[name][own],preds[name][own])})
    table=pd.DataFrame(results);table.to_csv(ROOT/'tensile_modulus_film_cleansing_comparison.tsv',sep='\t',index=False,encoding='utf-8-sig')
    print(table.groupby(['test','variant'],sort=False)[['MAE','RMSE','R2','log10RMSE']].mean().round(3).to_string())
    # Paired family bootstrap of the squared-error and absolute-error change against raw training.
    rng=np.random.default_rng(SEED);common=np.where(np.isfinite(strict))[0]
    families=pd.Series(groups[common]).groupby(groups[common]).groups
    base=np.mean(predictions['raw'],axis=0)
    labels_family=[common[list(v)] for v in families.values()]
    for name in RULES:
        if name=='raw':continue
        p=np.mean(predictions[name],axis=0);boots=[]
        for _ in range(N_BOOT):
            take=np.concatenate([labels_family[i] for i in rng.integers(0,len(labels_family),len(labels_family))])
            boots.append([np.abs(p[take]-strict[take]).mean()-np.abs(base[take]-strict[take]).mean(),np.sqrt(((p[take]-strict[take])**2).mean())-np.sqrt(((base[take]-strict[take])**2).mean())])
        lo,hi=np.percentile(boots,[2.5,97.5],axis=0)
        print(name,'vs raw on strict labels: dMAE [%.3f,%.3f] dRMSE [%.3f,%.3f]'%(lo[0],hi[0],lo[1],hi[1]))
    # Rough ceiling: scatter among repeat measurements of one structure after the strictest rules.
    kept=audit[~audit[RULES['all']].any(axis=1)];log=np.log10(kept.modulus)
    groups_rep=kept.assign(log=log).groupby('row_id').log
    size=groups_rep.size();rep=size[size>1].index
    pooled=((size[rep]-1)*groups_rep.var(ddof=1)[rep]).sum()/(size[rep]-1).sum()
    print('replicate structures',len(rep),'within log10 var',round(pooled,4),'sd',round(pooled**.5,3),'between-row log10 var',round(np.nanvar(np.log10(strict)),4),'single-measurement rows',int((size==1).sum()))

if __name__=='__main__':main()
