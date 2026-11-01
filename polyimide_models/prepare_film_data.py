"""Build a separate film/sheet/membrane modulus domain before label aggregation."""
import json,collections,argparse
from pathlib import Path
import numpy as np,pandas as pd
from modeling import short_hash
ROOT=Path(__file__).resolve().parent;SRC=ROOT.parent/'polyimide_tg_modulus'
ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['Tg','tensile_modulus'],default='tensile_modulus');args=ap.parse_args()
base_target=args.target;target=base_target+'_film'
data=json.loads((ROOT/'datasets.json').read_text());allrows=data[base_target]
measurement_map={mid:i for i,r in enumerate(allrows) for mid in r['measurement_ids']}
m=pd.read_csv(SRC/'polyimide_measurements.tsv',sep='\t',keep_default_na=False)
s=pd.read_csv(SRC/'polyimide_samples.tsv',sep='\t',keep_default_na=False).set_index('sample_id')
film_shapes={'Film','Sheet','Membrane','membrane','Foil'}
raw=collections.defaultdict(list)
for _,r in m[m.property==base_target].iterrows():
    if r.measurement_id in measurement_map and s.loc[r.sample_id,'sample_shape_raw'] in film_shapes:
        raw[measurement_map[r.measurement_id]].append(r)
rows=[];indices=[]
for i,rs in sorted(raw.items()):
    base=dict(allrows[i]);values=np.array([float(r.value_min) for r in rs])
    base.update({'target':target,'row_id':target+'_'+short_hash(base['chemistry_key']),
                 'value':float(np.median(values)),'measurement_count':len(rs),'value_min':float(values.min()),
                 'value_max':float(values.max()),'value_std':float(values.std()),
                 'measurement_ids':[r.measurement_id for r in rs],'sample_ids':list(dict.fromkeys(r.sample_id for r in rs)),
                 'reference_nos':list(dict.fromkeys(r.reference_no for r in rs)),
                 'shapes':list(dict.fromkeys(s.loc[r.sample_id,'sample_shape_raw'] for r in rs)),
                 'registry_pids':list(dict.fromkeys(r.registry_pid for r in rs))})
    rows.append(base);indices.append(i)
data[target]=rows
(ROOT/'datasets.json').write_text(json.dumps(data,ensure_ascii=False))
arrays=np.load(ROOT/'features.npz');new={k:arrays[k] for k in arrays.files}
for k in arrays.files:
    if k.startswith(base_target+'__'):new[k.replace(base_target+'__',target+'__')]=arrays[k][indices]
arrays.close();np.savez_compressed(ROOT/'features_updated.npz',**new);(ROOT/'features_updated.npz').replace(ROOT/'features.npz')
pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in rows]).to_csv(ROOT/(target+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
print('Film-only rows',len(rows),'measurements',sum(r['measurement_count'] for r in rows),'families',len(set(r['family_group'] for r in rows)),flush=True)
