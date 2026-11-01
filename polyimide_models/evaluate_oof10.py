#!/usr/bin/env python3
"""Pooled 10-fold group OOF assessment of previously selected fixed models."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from train_models import make_model,forward,inverse,metrics,candidate_id,SEED
ROOT=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--target',required=True,choices=['Tg','Tg_film','tensile_modulus','tensile_modulus_film','tensile_modulus_film_range','tensile_modulus_film_clean','CTE_film']);args=ap.parse_args();t=args.target
 ds=json.loads((ROOT/'datasets.json').read_text())[t];a=np.load(ROOT/'features.npz');s=json.loads((ROOT/t/'frozen_selection.json').read_text());members=s['members'];weights=np.array([m['weight'] for m in members]);y=np.array([r['value'] for r in ds]);g=np.array([r['family_group'] for r in ds]);pred=np.full((len(ds),5),np.nan);folds=np.full(len(ds),-1);baseline=np.full(len(ds),np.nan);fm=[]
 X={m['feature_set']:a[t+'__'+m['feature_set']] for m in members}
 for fold,(tr,va) in enumerate(GroupKFold(n_splits=10,shuffle=True,random_state=SEED+10).split(y,groups=g),1):
  tu=set(c for i in tr for c in ds[i]['components']);vu=set(c for i in va for c in ds[i]['components']);assert tu.isdisjoint(vu)
  assert (folds[va]==-1).all();folds[va]=fold;baseline[va]=np.median(y[tr])
  for j,m in enumerate(members):
   est=make_model(m['model'],m['variant']);est.fit(X[m['feature_set']][tr],forward(y[tr],m['target_transform']));pred[va,j]=inverse(est.predict(X[m['feature_set']][va]),m['target_transform'],t)
  met=metrics(y[va],pred[va]@weights);fm.append({'fold':fold,'train_rows':len(tr),'validation_rows':len(va),'validation_families':len(set(g[va])),'shared_repeat_units':0,**met});print(t,'fold',fold,'n',len(va),'MAE',met['MAE'],'R2',met['R2'],flush=True)
 assert np.isfinite(pred).all() and (folds>0).all();ens=pred@weights;best_index=next(i for i,m in enumerate(members) if m['candidate_id']==s['best_single_candidate']);best=pred[:,best_index]
 rows=[]
 for i,r in enumerate(ds):
  rows.append({'row_id':r['row_id'],'family_group':r['family_group'],'fold':int(folds[i]),'polymer_kind':r['polymer_kind'],'observed':y[i],'best_single_prediction':best[i],'ensemble_prediction':ens[i],'ensemble_error':ens[i]-y[i],'components_json':json.dumps(r['components']),'mole_fractions_json':json.dumps(r['mole_fractions']),'sequence_type':r['sequence_type'],'shapes_json':json.dumps(r['shapes']),'reference_nos_json':json.dumps(r['reference_nos']),**{f'member_{j+1}_prediction':pred[i,j] for j in range(5)}})
 out=ROOT/t
 pd.DataFrame(rows).to_csv(out/'oof10_predictions.tsv',sep='\t',index=False,encoding='utf-8-sig');pd.DataFrame(fm).to_csv(out/'oof10_folds.tsv',sep='\t',index=False,encoding='utf-8-sig')
 result={'target':t,'unit':'C' if t.startswith('Tg') else 'ppm_per_C' if t.startswith('CTE') else 'GPa','n_splits':10,'rows':len(ds),'families':len(set(g)),'shared_repeat_units':0,'seed':SEED+10,'model_selection':'Previously selected models, hyperparameters and weights fixed; no selection within these folds.','caveat':'Non-nested assessment: prior feature/model selection used part of this same dataset. All predictions are out-of-fold for fitting, but selection bias can remain. Not equivalent to the untouched holdout.','aggregation':'Metrics calculated once on all concatenated OOF predictions, not an average of fold R2.','best_single_oof':metrics(y,best),'ensemble_oof':metrics(y,ens),'baseline_oof':metrics(y,baseline),'folds':fm,'selection':s}
 (out/'oof10_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('RESULT',t,json.dumps(result['ensemble_oof']),flush=True)
if __name__=='__main__':main()
