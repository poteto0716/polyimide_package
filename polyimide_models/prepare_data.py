import collections,json
from pathlib import Path
import numpy as np,pandas as pd
from modeling import FeatureBuilder,FEATURE_SETS,chemistry_key,short_hash
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'polyimide_tg_modulus'

def main():
    m=pd.read_csv(SRC/'polyimide_measurements.tsv',sep='\t',keep_default_na=False)
    s=pd.read_csv(SRC/'polyimide_samples.tsv',sep='\t',keep_default_na=False).set_index('sample_id')
    c=pd.read_csv(SRC/'polyimide_components.tsv',sep='\t',keep_default_na=False)
    bysid={k:v.sort_values('component_order') for k,v in c.groupby('sample_id')}
    builder=FeatureBuilder();output={};all_rows=[];audit=[]
    for target in ['Tg','tensile_modulus']:
        rows=[]
        for _,r in m[m.property==target].iterrows():
            reason=''
            lo=pd.to_numeric(r.value_min,errors='coerce');hi=pd.to_numeric(r.value_max,errors='coerce')
            if r.material_group!='neat_polymer':reason='modified_or_composite'
            elif not r.smiles_complete:reason='incomplete_smiles'
            elif r.qualifier not in ['','ca']:reason='censored_or_other_qualifier'
            elif not np.isfinite(lo) or not np.isfinite(hi) or lo!=hi:reason='nonpoint_or_missing_value'
            elif not r.composition_mole_fractions_json:reason='unknown_or_unaligned_composition'
            elif target=='tensile_modulus' and lo<=.01:reason='tensile_modulus_at_or_below_0.01_GPa'
            fractions=json.loads(r.composition_mole_fractions_json) if r.composition_mole_fractions_json else []
            comp=bysid.get(r.sample_id)
            if not reason and (comp is None or len(comp)!=len(fractions)):reason='component_alignment'
            # Do not combine high-temperature and ambient tensile measurements.
            temp=pd.to_numeric(r.temperature_min_raw,errors='coerce')
            if target=='tensile_modulus' and not reason:
                if r.temperature_unit_raw=='K' and np.isfinite(temp):temp-=273.15
                if np.isfinite(temp) and r.temperature_unit_raw not in ['C','K']:reason='temperature_unit_unknown'
                elif np.isfinite(temp) and not 15<=temp<=35:reason='nonambient_temperature'
            audit.append({'measurement_id':r.measurement_id,'target':target,'sample_id':r.sample_id,'included':not bool(reason),'reason':reason or 'included'})
            if reason:continue
            pairs=collections.defaultdict(float)
            for smi,f in zip(comp.smiles.tolist(),fractions):
                if f>0:pairs[smi]+=f
            pairs=sorted(pairs.items());components=[p[0] for p in pairs];fs=[p[1] for p in pairs]
            key=chemistry_key(components,fs,r.sequence_type)
            rows.append({'target':target,'chemistry_key':key,'components':components,'mole_fractions':fs,'sequence_type':r.sequence_type,
                         'value':float(lo),'sample_id':r.sample_id,'measurement_id':r.measurement_id,'reference_no':r.reference_no,
                         'polymer_kind':r.polymer_kind,'registry_pid':r.registry_pid,'shape':s.loc[r.sample_id,'sample_shape_raw'],
                         'temperature_C':float(temp) if np.isfinite(temp) else None,'qualifier':r.qualifier})
        grouped=[]
        for key,g in pd.DataFrame(rows).groupby('chemistry_key',sort=False):
            d=g.iloc[0].to_dict();values=g.value.to_numpy()
            d.update({'row_id':target+'_'+short_hash(key),'value':float(np.median(values)),
                      'measurement_count':len(g),'value_min':float(values.min()),'value_max':float(values.max()),
                      'value_std':float(values.std()),'sample_ids':list(dict.fromkeys(g.sample_id)),
                      'measurement_ids':g.measurement_id.tolist(),'reference_nos':list(dict.fromkeys(g.reference_no)),
                      'shapes':list(dict.fromkeys(g['shape'])),'registry_pids':list(dict.fromkeys(g.registry_pid))})
            for n in ['sample_id','measurement_id','reference_no','shape','registry_pid','temperature_C','qualifier']:d.pop(n,None)
            grouped.append(d)
        output[target]=grouped;all_rows.extend(grouped)
    # Connected families: any shared repeat unit keeps homopolymers and their
    # copolymers in the same train/validation/test group, across mol ratios.
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for r in all_rows:
        for smi in r['components']:union(r['components'][0],smi)
    matrices={}
    for target,rows in output.items():
        for r in rows:r['family_group']='F_'+short_hash(find(r['components'][0]))
        frames=[]
        for r in rows:
            f=builder.build(r['components'],r['mole_fractions'],r['sequence_type']);frames.append(f)
        for name in FEATURE_SETS:matrices[target+'__'+name]=np.array([f[name] for f in frames])
        serial=[]
        for r in rows:serial.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in r.items()})
        pd.DataFrame(serial).to_csv(ROOT/(target+'_dataset.tsv'),sep='\t',index=False,encoding='utf-8-sig')
        counts=collections.Counter(r['family_group'] for r in rows)
        print(target,'rows',len(rows),'families',len(counts),'largest groups',counts.most_common(5),flush=True)
    np.savez_compressed(ROOT/'features.npz',**matrices)
    (ROOT/'datasets.json').write_text(json.dumps(output,ensure_ascii=False))
    (ROOT/'feature_names.json').write_text(json.dumps({n:builder.names(n) for n in FEATURE_SETS},ensure_ascii=False,indent=2))
    pd.DataFrame(audit).to_csv(ROOT/'data_selection_audit.tsv',sep='\t',index=False,encoding='utf-8-sig')
    print('features',{n:len(builder.names(n)) for n in FEATURE_SETS},flush=True)
if __name__=='__main__':main()
