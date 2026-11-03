#!/usr/bin/env python3
"""Soft segments for the optimizers: long, flexible monomers such as an aminopropyl-terminated
siloxane diamine of molecular weight 3200.

Nothing like them is in the training data, and their repeat units are far outside what the property
models were trained on, so the models are not asked about them. Instead a design is split into
 - its hard part: the same design with the soft monomers removed and the other fractions on that side
   renormalized (ODPA / TPE-R for ODPA / MESi3200:TPE-R = 0.12:0.88), predicted by the usual
   posterior (ensemble + GP), and
 - a soft shift: slope_t * w, where w is the weight fraction of soft segment in the polymer and the
   slope (per target, in model space: deg C for Tg, log10 GPa for modulus, ppm/K for CTE) is learned
   from measured soft-segment polymers by Bayesian linear regression on
   measured - hard-part prediction = slope * w + error.
Segmented siloxane-imide copolymers phase-separate, so the hard phase keeps much of its Tg while the
soft phase pulls the modulus down; a slope per property captures that to first order. One measured
polymer pins the slopes roughly; each further one narrows them. Without any, the slopes keep their
wide prior and every soft design is a guess (the scripts warn).

Soft monomers come from soft_monomers.csv (columns: abbreviation, side, name, smiles, recipe, note).
A recipe builds the SMILES, e.g. 'siloxane_diamine:mw=3200:groups=C|CC' (methyl/ethyl siloxane,
aminopropyl ends, chain length closest to that molecular weight).
"""
import json
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from mobo_copolyimide import TARGETS,UNITS,composition_fingerprint,short_name,to_model_space

HERE=Path(__file__).resolve().parent
SOFT_FILE=HERE/'soft_monomers.csv'
WATER=18.015
DEFAULT_PRIOR_SD={'Tg':300.,'modulus':3.,'CTE':300.}  # slope sd per unit weight fraction, model space

def siloxane_diamine_smiles(mw,groups=('C','CC'),spacer='CCC'):
    """H2N-spacer-[Si(R1)(R2)-O]n-Si(R1)(R2)-spacer-NH2 with n chosen to come closest to mw."""
    def build(n):
        si=f'[Si]({groups[0]})({groups[1]})'
        return f'N{spacer}{si}'+f'O{si}'*n+f'{spacer[::-1]}N'
    best=min(range(1,400),key=lambda n:abs(Descriptors.MolWt(Chem.MolFromSmiles(build(n)))-mw))
    return build(best)

def from_recipe(recipe):
    kind,*parts=recipe.split(':');options=dict(p.split('=',1) for p in parts)
    if kind=='siloxane_diamine':
        return siloxane_diamine_smiles(float(options['mw']),tuple(options.get('groups','C|C').split('|')),options.get('spacer','CCC'))
    raise SystemExit(f'Unknown soft-monomer recipe: {recipe}')

def read_soft_monomers(path=SOFT_FILE):
    if not Path(path).exists():return []
    rows=pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False).to_dict('records')
    out=[]
    for r in rows:
        smiles=r.get('smiles','').strip() or from_recipe(r['recipe'].strip())
        if Chem.MolFromSmiles(smiles) is None:raise SystemExit(f'Bad SMILES for soft monomer {r.get("abbreviation")}')
        out.append(dict(abbreviation=r['abbreviation'].strip(),side=r['side'].strip(),name=r.get('name','').strip(),smiles=smiles,
                        mw=Descriptors.MolWt(Chem.MolFromSmiles(smiles))))
    return out

def register(acids,diamines,soft_rows):
    """Add soft monomers to the candidate sides; return (soft acid indices, soft diamine indices)."""
    soft=({},{})
    for r in soft_rows:
        side,bucket=(acids,soft[0]) if r['side']=='acid' else (diamines,soft[1])
        index=side.add(r['smiles'],r['name'],'','',r['abbreviation'],'')
        bucket[index]=r
    return set(soft[0]),set(soft[1])

class SoftSurrogate:
    """Surrogate.evaluate for designs that may hold soft monomers; plain designs go to the base."""
    def __init__(self,base,space,soft_acids,soft_diamines,prior_sd=None):
        self.base=base;self.space=space;self.targets=base.targets;self.gps=base.gps;self.models=base.models
        self.soft=(set(soft_acids),set(soft_diamines));self.memo={};self.mixed=base.mixed;self.molwt={}
        prior_sd={**DEFAULT_PRIOR_SD,**(prior_sd or {})}
        self.prior=np.array([prior_sd[short_name(t)]**2 for t in self.targets])
        self.slope=np.zeros(len(self.targets));self.slope_var=self.prior.copy();self.soft_rows=0
    # --- composition ---
    def mass(self,side,index):
        key=(side,index)
        if key not in self.molwt:
            monomers=(self.space.acids if side==0 else self.space.diamines).monomers
            self.molwt[key]=Descriptors.MolWt(Chem.MolFromSmiles(monomers[index]['smiles']))
        return self.molwt[key]
    def is_soft(self,design):
        return any(s in self.soft[k] for k in (0,1) for s in design[k][0])
    def soft_weight(self,design):
        """Weight fraction of soft segment in the polyimide (two waters lost per imide pair)."""
        total=-2*WATER;soft=0.
        for k in (0,1):
            for s,u in zip(*design[k]):
                m=self.mass(k,s)*u/UNITS;total+=m
                if s in self.soft[k]:soft+=m-(2*1.008*u/UNITS if k==1 else 0.)
        return soft/total
    def hard_design(self,design):
        """The design without soft monomers, fractions renormalized per side; None if a side is all soft."""
        out=[]
        for k in (0,1):
            pairs=[(s,u) for s,u in zip(*design[k]) if s not in self.soft[k]]
            if not pairs:return None
            total=sum(u for _,u in pairs);raw=[u*UNITS/total for _,u in pairs]
            units=[int(x) for x in raw];left=UNITS-sum(units)
            for i in sorted(range(len(raw)),key=lambda i:raw[i]-units[i],reverse=True)[:left]:units[i]+=1
            out.append((tuple(s for s,_ in pairs),tuple(units)))
        return tuple(out)
    # --- learning ---
    def add_observations(self,items):
        """items: (design, {target: value}). Plain ones update the base; soft ones fit the slopes."""
        plain=[(d,v) for d,v in items if not self.is_soft(d)];soft=[(d,v) for d,v in items if self.is_soft(d)]
        if plain:self.base.add_observations(plain)
        self.memo.clear()
        if not soft:return
        hard=[self.hard_design(d) for d,_ in soft]
        if any(h is None for h in hard):raise SystemExit('A measured soft-segment polymer needs at least one non-soft monomer on each side.')
        mean,sd=self.base.evaluate(hard)
        w=np.array([self.soft_weight(d) for d,_ in soft])
        for k,target in enumerate(self.targets):
            name=short_name(target);rows=[i for i,(_,v) in enumerate(soft) if target in v]
            if not rows:continue
            y=np.array([to_model_space(target,np.array(soft[i][1][target])) for i in rows],dtype=float)
            r=y-mean[rows,k];s2=sd[rows,k]**2+self.gps[target].noise;x=w[rows]
            precision=1/self.prior[k]+np.sum(x*x/s2)
            self.slope_var[k]=1/precision;self.slope[k]=self.slope_var[k]*np.sum(x*r/s2)
            print(f'soft slope {name}: {self.slope[k]:.3g} +/- {np.sqrt(self.slope_var[k]):.3g} per unit weight fraction (model space), {len(rows)} polymers',flush=True)
        self.soft_rows=len(soft)
    def slopes(self):
        return {short_name(t):dict(slope=float(self.slope[k]),sd=float(np.sqrt(self.slope_var[k]))) for k,t in enumerate(self.targets)}
    # --- prediction ---
    def evaluate(self,designs):
        missing=[d for d in dict.fromkeys(designs) if d not in self.memo]
        if missing:
            plain=[d for d in missing if not self.is_soft(d)]
            if plain:
                m,s=self.base.evaluate(plain)
                for d,a,b in zip(plain,m,s):self.memo[d]=(a,b)
            soft=[d for d in missing if self.is_soft(d)]
            if soft:
                hard=[self.hard_design(d) for d in soft]
                ok=[i for i,h in enumerate(hard) if h is not None]
                m,s=self.base.evaluate([hard[i] for i in ok]) if ok else (np.zeros((0,len(self.targets))),)*2
                for j,i in enumerate(ok):
                    w=self.soft_weight(soft[i])
                    self.memo[soft[i]]=(m[j]+self.slope*w,np.sqrt(s[j]**2+w*w*self.slope_var))
                for i,h in enumerate(hard):
                    if h is None:self.memo[soft[i]]=(np.full(len(self.targets),np.nan),np.full(len(self.targets),1e6))
        return np.array([self.memo[d][0] for d in designs]),np.array([self.memo[d][1] for d in designs])
    def hard_prediction(self,designs):
        hard=[self.hard_design(d) if self.is_soft(d) else d for d in designs]
        return self.base.evaluate(hard)
    def correlation(self,designs,k):
        """Posterior correlation of target k between designs: GP covariance of the hard parts plus the
        shared slope uncertainty (w_i w_j var), which ties all soft designs together."""
        hard=[self.hard_design(d) if self.is_soft(d) else d for d in designs]
        X=np.array([composition_fingerprint(*self.space.composition(h)) for h in hard])
        target=self.targets[k];C=self.gps[target].covariance(X)
        d=np.sqrt(np.maximum(np.diag(C),1e-12));R=C/np.outer(d,d)
        _,s=self.base.evaluate(hard);hs=s[:,k]
        w=np.array([self.soft_weight(x) if self.is_soft(x) else 0. for x in designs])
        cov=np.outer(hs,hs)*R+np.outer(w,w)*self.slope_var[k]
        t=np.sqrt(np.maximum(np.diag(cov),1e-12))
        return cov/np.outer(t,t)
