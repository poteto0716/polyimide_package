"""Add film modulus target without measurements that audit_film_modulus.py flags as non-tensile or physically inconsistent."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from modeling import short_hash
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'polyimide_tg_modulus'
BASE='tensile_modulus_film_range';TARGET='tensile_modulus_film_clean'
RULES=['non_tensile_method','crosslinked','secant_inconsistent','strength_implausible','low_modulus_review']
data=json.loads((ROOT/'datasets.json').read_text())
audit=pd.read_csv(ROOT/'tensile_modulus_film_audit.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
removed=set(audit[(audit[RULES].astype(str)=='True').any(axis=1)].measurement_id)
m=pd.read_csv(SRC/'polyimide_measurements.tsv',sep='\t',keep_default_na=False,low_memory=False).set_index('measurement_id')
rows=[];indices=[]
for i,r in enumerate(data[BASE]):
    ids=[x for x in r['measurement_ids'] if x not in removed]
    if not ids:continue
    kept=m.loc[ids];values=pd.to_numeric(kept.value_min).to_numpy()
    v=dict(r);v.update(target=TARGET,row_id=TARGET+'_'+short_hash(r['chemistry_key']),value=float(np.median(values)),measurement_count=len(ids),
        value_min=float(values.min()),value_max=float(values.max()),value_std=float(values.std()),measurement_ids=ids,
        sample_ids=list(dict.fromkeys(kept.sample_id)),reference_nos=list(dict.fromkeys(kept.reference_no)))
    rows.append(v);indices.append(i)
data[TARGET]=rows
a=np.load(ROOT/'features.npz');arrays={k:a[k] for k in a.files};a.close()
for key in list(arrays):
    if key.startswith(BASE+'__'):arrays[key.replace(BASE+'__',TARGET+'__')]=arrays[key][indices]
(ROOT/'datasets.json').write_text(json.dumps(data,ensure_ascii=False))
np.savez_compressed(ROOT/'features_clean.npz',**arrays);(ROOT/'features_clean.npz').replace(ROOT/'features.npz')
pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows]).to_csv(ROOT/(TARGET+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
print(BASE,len(data[BASE]),'rows ->',TARGET,len(rows),'rows;',len(removed),'measurements removed')
