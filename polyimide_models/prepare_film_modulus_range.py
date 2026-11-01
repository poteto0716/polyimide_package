"""Add film modulus target limited per measurement to 0.01 < GPa < 10; other targets are untouched."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from modeling import short_hash
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'polyimide_tg_modulus'
BASE='tensile_modulus_film';TARGET='tensile_modulus_film_range';LOW=.01;HIGH=10.
data=json.loads((ROOT/'datasets.json').read_text())
m=pd.read_csv(SRC/'polyimide_measurements.tsv',sep='\t',keep_default_na=False).set_index('measurement_id')
rows=[];indices=[];removed=[]
for i,r in enumerate(data[BASE]):
    values=pd.to_numeric(m.loc[r['measurement_ids'],'value_min']).to_numpy()
    keep=(values>LOW)&(values<HIGH)
    ids=[x for x,k in zip(r['measurement_ids'],keep) if k]
    removed+=[{'row_id':r['row_id'],'measurement_id':x,'value_GPa':float(v),'row_dropped':not keep.any()} for x,v,k in zip(r['measurement_ids'],values,keep) if not k]
    if not ids:continue
    kept=m.loc[ids];values=values[keep]
    v=dict(r);v.update(target=TARGET,row_id=TARGET+'_'+short_hash(r['chemistry_key']),value=float(np.median(values)),measurement_count=len(ids),
        value_min=float(values.min()),value_max=float(values.max()),value_std=float(values.std()),measurement_ids=ids,
        sample_ids=list(dict.fromkeys(kept.sample_id)),reference_nos=list(dict.fromkeys(kept.reference_no)))
    rows.append(v);indices.append(i)
data[TARGET]=rows
a=np.load(ROOT/'features.npz');arrays={k:a[k] for k in a.files};a.close()
for key in list(arrays):
    if key.startswith(BASE+'__'):arrays[key.replace(BASE+'__',TARGET+'__')]=arrays[key][indices]
(ROOT/'datasets.json').write_text(json.dumps(data,ensure_ascii=False))
np.savez_compressed(ROOT/'features_range.npz',**arrays);(ROOT/'features_range.npz').replace(ROOT/'features.npz')
pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows]).to_csv(ROOT/(TARGET+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
pd.DataFrame(removed).to_csv(ROOT/(TARGET+'_removed_measurements.tsv'),sep='\t',index=False,encoding='utf-8-sig')
print(BASE,len(data[BASE]),'rows ->',TARGET,len(rows),'rows;',len(removed),'measurements removed;',sum(x['row_dropped'] for x in removed),'of them in dropped rows; value range',min(r['value'] for r in rows),max(r['value'] for r in rows))
