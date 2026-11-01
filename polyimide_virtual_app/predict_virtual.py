#!/usr/bin/env python3
"""Predict Tg and tensile modulus for every virtual TCI-monomer polyimide homopolymer.

Uses the production ensembles in polyimide_models and adds, per target, the highest Morgan
Tanimoto similarity to a repeat unit in that target's training data.

 .venv/bin/python polyimide_virtual_app/predict_virtual.py
"""
import json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent
MODELS=ROOT.parent/'polyimide_models'
sys.path.insert(0,str(MODELS))
TARGETS=['Tg_film','tensile_modulus_film_clean','CTE_film']
BATCH=500
GENERATOR=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def fingerprint(smiles):
    return GENERATOR.GetFingerprint(Chem.MolFromSmiles(smiles))

def training_units(target):
    data=pd.read_csv(MODELS/f'{target}_dataset.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
    return sorted({s for row in data.components for s in json.loads(row)})

def main():
    virtual=pd.read_csv(ROOT.parent/'tci_polyimide/output/tci_polyimide_psmiles.csv',encoding='utf-8-sig',keep_default_na=False)
    records=[{'components':[s],'mole_fractions':[1.0],'sequence_type':'not_applicable'} for s in virtual.psmiles]
    fingerprints=[fingerprint(s) for s in virtual.psmiles]
    builder=None
    for target in TARGETS:
        model=joblib.load(MODELS/target/'ensemble_production.joblib')
        # Sharing the builder reuses its per-structure descriptor cache across targets.
        builder=builder or model.builder
        model.builder=builder
        values,spreads=[],[]
        for start in range(0,len(records),BATCH):
            value,members=model.predict(records[start:start+BATCH],return_members=True)
            values.append(value);spreads.append(np.std(members,axis=1))
            print(target,start+BATCH,flush=True)
        virtual[target+'_pred']=np.concatenate(values)
        virtual[target+'_member_std']=np.concatenate(spreads)
        train=[fingerprint(s) for s in training_units(target)]
        virtual[target+'_max_similarity']=[max(DataStructs.BulkTanimotoSimilarity(f,train)) for f in fingerprints]
    virtual.to_csv(ROOT/'virtual_predictions.tsv',sep='\t',index=False,encoding='utf-8-sig')
    print(virtual.filter(regex='pred|similarity').describe().T.to_string())

if __name__=='__main__':main()
