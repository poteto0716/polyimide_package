#!/usr/bin/env python3
"""Add a neat-film in-plane CTE target from the source TSV; other targets are untouched.

Source rows repeat one property for each combination of other properties, so records are
deduplicated by property ID. Most records state no temperature range, so this target is
the usual reported glassy-state TMA value, not a strict room-temperature CTE.
 .venv/bin/python polyimide_models/prepare_film_cte.py
"""
import collections,json
from pathlib import Path
import numpy as np,pandas as pd
from modeling import FeatureBuilder,FEATURE_SETS,chemistry_key,short_hash
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;SRC=BASE/'polyimide_tg_modulus'
SOURCE=BASE/'data_20240322.tsv';SOURCE_ENCODING='cp932'
TARGET='CTE_film'
SIDP,SID,REF,PID,LOW,HIGH,QUALIFIER,UNIT,METHOD,T_LOW,T_HIGH=0,1,2,318,319,320,321,322,327,329,330  # 0-based
FILM_SHAPES={'Film','Sheet','Membrane','membrane','Foil'}
THERMOMECHANICAL=('tma','thermomechanical','dilatometer','thermal analy')
GLASSY_MAX_PPM=150.   # a glassy polyimide does not expand this fast; such values suggest a rubbery-range or unit problem
QUALIFIERS_KEPT=('','ca')

def read_records(samples):
    records={}
    with SOURCE.open(encoding=SOURCE_ENCODING,errors='replace',newline='') as handle:
        handle.readline()
        for line in handle:
            p=line.rstrip('\r\n').split('\t')
            if p[SID] in samples.index and p[LOW].strip():
                records.setdefault(p[PID],{'property_id':p[PID],'sample_id':p[SID],'reference_no':p[REF],'low':p[LOW],'high':p[HIGH],
                    'qualifier':p[QUALIFIER].strip(),'unit':p[UNIT].strip(),'method':p[METHOD].strip(),'t_low':p[T_LOW].strip(),'t_high':p[T_HIGH].strip()})
    return pd.DataFrame(records.values())

def exclusion(r,sample,components):
    low,high=pd.to_numeric(r.low,errors='coerce'),pd.to_numeric(r.high,errors='coerce')
    fractions=sample.composition_mole_fractions_json
    if not sample.smiles_complete:return 'incomplete_smiles'
    if sample.sample_shape_raw not in FILM_SHAPES:return 'not_film'
    if str(sample.shape_construction_raw).strip() in('crosslinked','hyper-branched'):return 'not_linear'
    if r.qualifier not in QUALIFIERS_KEPT:return 'censored_value'
    if r.unit!='1/C':return 'unit_unknown'
    if not np.isfinite(low) or low!=high:return 'range_or_missing_value'
    if not fractions or components is None or len(components)!=len(json.loads(fractions)):return 'component_alignment'
    if r.t_low or r.t_high:return 'stated_range_excludes_room_temperature'
    if r.method and not any(k in r.method.lower() for k in THERMOMECHANICAL):return 'not_thermomechanical'
    if low*1e6>GLASSY_MAX_PPM:return 'above_glassy_limit'
    return ''

def main():
    samples=pd.read_csv(SRC/'polyimide_samples.tsv',sep='\t',keep_default_na=False,low_memory=False).set_index('sample_id')
    parts=pd.read_csv(SRC/'polyimide_components.tsv',sep='\t',keep_default_na=False)
    by_sample={k:v.sort_values('component_order') for k,v in parts.groupby('sample_id')}
    records=read_records(samples)
    rows=[];audit=[]
    for r in records.itertuples():
        sample=samples.loc[r.sample_id];components=by_sample.get(r.sample_id)
        reason=exclusion(r,sample,components)
        audit.append({'property_id':r.property_id,'sample_id':r.sample_id,'reference_no':r.reference_no,'polymer':sample.reference_polymer_name,
            'CTE_ppm_per_C':float(pd.to_numeric(r.low,errors='coerce'))*1e6,'method':r.method,'stated_range':f'{r.t_low}-{r.t_high}' if r.t_low else '','shape':sample.sample_shape_raw,'reason':reason or 'included'})
        if reason:continue
        pairs=collections.defaultdict(float)
        for smiles,f in zip(components.smiles.tolist(),json.loads(sample.composition_mole_fractions_json)):
            if f>0:pairs[smiles]+=f
        pairs=sorted(pairs.items());units=[p[0] for p in pairs];fractions=[p[1] for p in pairs]
        rows.append({'key':chemistry_key(units,fractions,sample.sequence_type),'components':units,'mole_fractions':fractions,'sequence_type':sample.sequence_type,
            'polymer_kind':sample.polymer_kind,'value':float(r.low)*1e6,'property_id':r.property_id,'sample_id':r.sample_id,'reference_no':r.reference_no,'shape':sample.sample_shape_raw})
    frame=pd.DataFrame(rows)
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    for units in frame.components:
        for smiles in units:
            a,b=find(units[0]),find(smiles)
            if a!=b:parents[max(a,b)]=min(a,b)
    builder=FeatureBuilder();grouped=[]
    for key,g in frame.groupby('key',sort=False):
        first=g.iloc[0];values=g.value.to_numpy()
        grouped.append({'target':TARGET,'chemistry_key':key,'components':first.components,'mole_fractions':first.mole_fractions,'sequence_type':first.sequence_type,
            'value':float(np.median(values)),'polymer_kind':first.polymer_kind,'row_id':TARGET+'_'+short_hash(key),'measurement_count':len(g),'value_min':float(values.min()),
            'value_max':float(values.max()),'value_std':float(values.std()),'sample_ids':list(dict.fromkeys(g.sample_id)),'measurement_ids':g.property_id.tolist(),
            'reference_nos':list(dict.fromkeys(g.reference_no)),'shapes':list(dict.fromkeys(g['shape'])),'family_group':'F_'+short_hash(find(first.components[0]))})
    built=[builder.build(r['components'],r['mole_fractions'],r['sequence_type']) for r in grouped]
    data=json.loads((ROOT/'datasets.json').read_text());data[TARGET]=grouped
    a=np.load(ROOT/'features.npz');arrays={k:a[k] for k in a.files};a.close()
    for name in FEATURE_SETS:arrays[TARGET+'__'+name]=np.array([b[name] for b in built])
    (ROOT/'datasets.json').write_text(json.dumps(data,ensure_ascii=False))
    np.savez_compressed(ROOT/'features_cte.npz',**arrays);(ROOT/'features_cte.npz').replace(ROOT/'features.npz')
    pd.DataFrame([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()} for r in grouped]).to_csv(ROOT/(TARGET+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
    audit=pd.DataFrame(audit);audit.to_csv(ROOT/(TARGET+'_audit.tsv'),sep='\t',index=False,encoding='utf-8-sig')
    print(len(audit),'unique records',audit.reason.value_counts().to_dict())
    values=np.array([r['value'] for r in grouped])
    print(len(grouped),'rows',len({r['family_group'] for r in grouped}),'families',len(frame.reference_no.unique()),'references; ppm',np.percentile(values,[0,25,50,75,100]).round(1).tolist(),'single-measurement',sum(r['measurement_count']==1 for r in grouped))
    print(audit[audit.reason=='above_glassy_limit'][['reference_no','polymer','CTE_ppm_per_C']].to_string(index=False)[:900])

if __name__=='__main__':main()
