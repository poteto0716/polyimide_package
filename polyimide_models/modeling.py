"""Reusable, permutation-invariant polymer composition features and ensembles."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import Descriptors,MACCSkeys,rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
SEQUENCES=['block','random','statistical','alternating','graft','periodic','unknown']
DESCRIPTORS=[(n,f) for n,f in Descriptors._descList if n!='Ipc']
CHEM_NAMES=['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds','RingCount','NumAromaticRings','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAliphaticRings','NumSaturatedRings','BertzCT','BalabanJ','HallKierAlpha','Kappa1','Kappa2','Kappa3']
META_NAMES=['n_components','max_fraction','min_fraction','fraction_std','entropy','simpson_diversity','effective_components']+['sequence_'+s for s in SEQUENCES]+['sequence_unspecified']
FEATURE_SETS=['chemistry','rdkit','maccs','morgan_r2','morgan_r3','rdkit_path','hybrid']
class FeatureBuilder:
    def __init__(self):
        self.cache={}
        self.r2=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        self.r3=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=1024)
        self.path=rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=1024)
    def __getstate__(self):
        # RDKit fingerprint generators cannot be pickled; rebuild deterministic
        # generators on load rather than serializing Boost.Python objects.
        return {}
    def __setstate__(self,state):
        self.__init__()
    def unit(self,s):
        if s in self.cache:return self.cache[s]
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError('Invalid SMILES: '+s)
        vals=[]
        for _,f in DESCRIPTORS:
            try:
                v=float(f(mol));vals.append(v if np.isfinite(v) and abs(v)<1e20 else np.nan)
            except Exception:vals.append(np.nan)
        d=dict(zip([n for n,_ in DESCRIPTORS],vals))
        atoms=[a for a in mol.GetAtoms() if a.GetAtomicNum()>0];n=max(1,len(atoms))
        chem=[d.get(k,np.nan) for k in CHEM_NAMES]
        chem += [sum(a.GetAtomicNum()==z for a in atoms)/n for z in [6,7,8,9,14,15,16,17,35]]
        chem += [sum(a.GetIsAromatic() for a in atoms)/n,
                 len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7;R]([#6;R](=O))[#6;R](=O)')))/n,
                 sum(a.GetAtomicNum()==0 for a in mol.GetAtoms())]
        def arr(fp):
            a=np.zeros(fp.GetNumBits(),dtype=np.float32);DataStructs.ConvertToNumpyArray(fp,a);return a
        result={'chemistry':np.array(chem,dtype=np.float64),'rdkit':np.array(vals,dtype=np.float64),
                'maccs':arr(MACCSkeys.GenMACCSKeys(mol)), 'morgan_r2':arr(self.r2.GetFingerprint(mol)),
                'morgan_r3':arr(self.r3.GetFingerprint(mol)), 'rdkit_path':arr(self.path.GetFingerprint(mol))}
        self.cache[s]=result;return result
    def build(self,components,fractions,sequence='unknown'):
        if len(components)!=len(fractions) or not components:raise ValueError('Components and fractions must align.')
        x=np.asarray(fractions,dtype=float)
        if np.any(x<0) or not np.isfinite(x).all() or x.sum()<=0:raise ValueError('Invalid mole fractions.')
        pairs={}
        for s,fraction in zip(components,x):
            if fraction==0:continue
            mol=Chem.MolFromSmiles(s)
            if mol is None:raise ValueError('Invalid SMILES: '+s)
            canonical=Chem.MolToSmiles(mol)
            pairs[canonical]=pairs.get(canonical,0)+fraction
        components=sorted(pairs)
        x=np.array([pairs[s] for s in components]);x=x/x.sum()
        if len(components)==1:sequence='not_applicable'
        us=[self.unit(s) for s in components]
        meta=np.array([len(x),x.max(),x.min(),x.std(),-np.sum(x*np.log(x)),1-np.sum(x*x),1/np.sum(x*x)]+
                      [float(s in sequence.split('+')) for s in SEQUENCES]+[float('unspecified' in sequence)],dtype=float)
        result={}
        for name in FEATURE_SETS[:-1]:
            v=np.array([u[name] for u in us]);mean=np.sum(x[:,None]*v,axis=0)
            if name in ['chemistry','rdkit','maccs']:
                std=np.sqrt(np.maximum(0,np.sum(x[:,None]*(v-mean)**2,axis=0)))
                result[name]=np.r_[mean,std,meta].astype(np.float32)
            else:result[name]=np.r_[mean,meta].astype(np.float32)
        result['hybrid']=np.r_[result['rdkit'][:-len(meta)],result['morgan_r2'][:-len(meta)],meta].astype(np.float32)
        return result
    def names(self,name):
        chem=CHEM_NAMES+['atom_fraction_'+s for s in ['C','N','O','F','Si','P','S','Cl','Br']]+['aromatic_atom_fraction','imide_N_density','attachment_count']
        if name=='chemistry':base=chem
        elif name=='rdkit':base=[n for n,_ in DESCRIPTORS]
        elif name=='maccs':base=['MACCS_'+str(i) for i in range(167)]
        elif name in ['morgan_r2','morgan_r3','rdkit_path']:return [name+'_bit_'+str(i) for i in range(1024)]+META_NAMES
        elif name=='hybrid':return self.names('rdkit')[:-len(META_NAMES)]+self.names('morgan_r2')[:-len(META_NAMES)]+META_NAMES
        return ['mean_'+s for s in base]+['std_'+s for s in base]+META_NAMES

def chemistry_key(components,fractions,sequence):
    pairs={}
    for s,f in zip(components,fractions):pairs[s]=pairs.get(s,0)+f
    norm=sum(pairs.values());pairs=sorted((s,round(f/norm,10)) for s,f in pairs.items() if f>0)
    return json.dumps([pairs,sequence if len(pairs)>1 else 'homopolymer'],separators=(',',':'))

def short_hash(s):return hashlib.sha256(s.encode()).hexdigest()[:16]

class PolymerEnsemble:
    def __init__(self,members,weights,target,metadata=None):
        self.members=members;self.weights=np.array(weights);self.target=target;self.metadata=metadata or {}
        self.builder=FeatureBuilder()
    def predict(self,records,return_members=False):
        features=[self.builder.build(r['components'],r['mole_fractions'],r.get('sequence_type','unknown')) for r in records]
        predictions=[]
        for member in self.members:
            X=np.array([f[member['feature_set']] for f in features])
            y=member['estimator'].predict(X)
            if member['target_transform']=='log10':y=np.power(10,np.clip(y,-8,8))
            if self.target.startswith('tensile_modulus'):y=np.maximum(y,0)
            predictions.append(y)
        preds=np.array(predictions).T
        answer=preds@self.weights
        return (answer,preds) if return_members else answer
