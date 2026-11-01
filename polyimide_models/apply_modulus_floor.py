"""Reaggregate modulus labels after exclusions; preserve Tg and existing families."""
from pathlib import Path
import json,collections
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'polyimide_tg_modulus'
data=json.loads((ROOT/'datasets.json').read_text());m=pd.read_csv(SRC/'polyimide_measurements.tsv',sep='\t',keep_default_na=False).set_index('measurement_id');s=pd.read_csv(SRC/'polyimide_samples.tsv',sep='\t',keep_default_na=False).set_index('sample_id')
a=np.load(ROOT/'features.npz');arrays={k:a[k] for k in a.files};a.close();summary={}
for target in ['tensile_modulus','tensile_modulus_film']:
 old=data[target];new=[];indices=[]
 for i,r in enumerate(old):
  ids=[m_id for m_id in r['measurement_ids'] if m_id in m.index]
  if not ids:continue
  ms=m.loc[ids];values=pd.to_numeric(ms.value_min).to_numpy();assert (values>.01).all()
  v=dict(r);v.update(value=float(np.median(values)),measurement_count=len(ids),value_min=float(values.min()),value_max=float(values.max()),value_std=float(values.std()),measurement_ids=ids,sample_ids=list(dict.fromkeys(ms.sample_id)),reference_nos=list(dict.fromkeys(ms.reference_no)),registry_pids=list(dict.fromkeys(ms.registry_pid)),shapes=list(dict.fromkeys(s.loc[sid,'sample_shape_raw'] for sid in ms.sample_id)))
  new.append(v);indices.append(i)
 data[target]=new
 for key in list(arrays):
  if key.startswith(target+'__'):arrays[key]=arrays[key][indices]
 pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in new]).to_csv(ROOT/(target+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
 summary[target]={'before_rows':len(old),'after_rows':len(new),'before_measurements':sum(r['measurement_count'] for r in old),'after_measurements':sum(r['measurement_count'] for r in new)}
 # Invalidate generated fit/selection/OOF artifacts after the labels change.
 path=ROOT/target
 for f in path.iterdir():
  if f.is_file() and f.suffix in ['.json','.tsv','.joblib','.npy','.log']:f.unlink()
(ROOT/'datasets.json').write_text(json.dumps(data,ensure_ascii=False));np.savez_compressed(ROOT/'features_floor.npz',**arrays);(ROOT/'features_floor.npz').replace(ROOT/'features.npz')
(ROOT/'modulus_floor_update.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
