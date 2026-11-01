#!/usr/bin/env python3
"""Flag film modulus measurements using the source TSV; no model output is consulted.

Writes tensile_modulus_film_audit.tsv with one row per measurement of the
tensile_modulus_film_range target and the rules that flag it.
 .venv/bin/python polyimide_models/audit_film_modulus.py
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
SOURCE=BASE/'data_20240322.tsv'
SOURCE_ENCODING='cp932'
SIDP,STRENGTH,ELONGATION,SHAPE=0,650,658,16  # 0-based columns: SIDP, stress at break, elongation at break, construction
TARGET='tensile_modulus_film_range'
NON_TENSILE=('dma','dmta','nano','indent','method by ref','dynamic')
SECANT_RATIO_MAX=1.5   # strength/strain above the initial slope by this factor cannot come from one curve
STRENGTH_MIN_GPA=.01   # with MODULUS_LOW_GPA: both values far too low suggests a whole-record unit error
MODULUS_LOW_GPA=.5     # a low strength alone is not enough, since some papers only have the strength unit wrong
MAD_Z_MAX=3.5
MAD_MIN_MEASUREMENTS=4
OUTLIER_MIN_FACTOR=2.  # tightly reproduced structures must not flag ordinary values
LOW_MODULUS_REVIEW=frozenset(json.loads((ROOT/'review_references.json').read_text())['low_modulus'])  # uniformly low values from one paper

def source_columns(sidps):
    found={}
    with SOURCE.open(encoding=SOURCE_ENCODING,errors='replace',newline='') as handle:
        handle.readline()
        for line in handle:
            parts=line.rstrip('\r\n').split('\t')
            if parts[SIDP] in sidps:found[parts[SIDP]]=(parts[STRENGTH],parts[ELONGATION],parts[SHAPE])
    return found

def main():
    measurements=pd.read_csv(BASE/'polyimide_tg_modulus/polyimide_measurements.tsv',sep='\t',keep_default_na=False,low_memory=False)
    dataset=json.loads((ROOT/'datasets.json').read_text())[TARGET]
    owner={m:r['row_id'] for r in dataset for m in r['measurement_ids']}
    audit=measurements[measurements.measurement_id.isin(owner)].copy()
    audit['row_id']=audit.measurement_id.map(owner)
    audit['modulus']=audit.value_min.astype(float)
    audit['sidp']=audit.source_sidps.map(lambda s:json.loads(s)[0])
    source=source_columns(set(audit.sidp))
    assert set(audit.sidp)<=set(source)
    audit['strength']=pd.to_numeric(audit.sidp.map(lambda s:source[s][0]),errors='coerce')
    audit['elongation']=pd.to_numeric(audit.sidp.map(lambda s:source[s][1]),errors='coerce')
    audit['construction']=audit.sidp.map(lambda s:source[s][2])
    audit['secant_ratio']=audit.strength/(audit.elongation/100)/audit.modulus
    method=audit.measurement_method.str.lower()
    audit['non_tensile_method']=method.map(lambda m:any(k in m for k in NON_TENSILE))
    audit['crosslinked']=audit.construction.str.lower().str.contains('crosslinked')
    audit['secant_inconsistent']=audit.secant_ratio>SECANT_RATIO_MAX
    audit['strength_implausible']=(audit.strength<STRENGTH_MIN_GPA)&(audit.modulus<MODULUS_LOW_GPA)
    log=np.log10(audit.modulus)
    median=log.groupby(audit.row_id).transform('median');mad=(log-median).abs().groupby(audit.row_id).transform('median')
    count=log.groupby(audit.row_id).transform('size')
    audit['structure_outlier']=(count>=MAD_MIN_MEASUREMENTS)&(mad>0)&((log-median).abs()/(1.4826*mad)>MAD_Z_MAX)&((log-median).abs()>np.log10(OUTLIER_MIN_FACTOR))
    audit['low_modulus_review']=audit.reference_no.astype(str).str.zfill(7).isin(LOW_MODULUS_REVIEW)
    flags=['non_tensile_method','crosslinked','secant_inconsistent','strength_implausible','structure_outlier','low_modulus_review']
    keep=['measurement_id','row_id','sample_id','reference_no','modulus','strength','elongation','secant_ratio','measurement_method','construction']+flags
    audit[keep].to_csv(ROOT/'tensile_modulus_film_audit.tsv',sep='\t',index=False,encoding='utf-8-sig')
    print(len(audit),'measurements;',{f:int(audit[f].sum()) for f in flags},'any',int(audit[flags].any(axis=1).sum()))
    print(sorted(audit[audit.non_tensile_method].measurement_method.str[:50].unique()))
    print(audit[audit.structure_outlier|audit.strength_implausible|audit.secant_inconsistent][['reference_no','modulus','strength','elongation','secant_ratio','secant_inconsistent','strength_implausible','structure_outlier']].round(3).to_string(index=False))

if __name__=='__main__':main()
