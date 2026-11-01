#!/usr/bin/env python3
"""Multi-objective Bayesian optimization of copolyimide monomers and compositions, using NSGA-II.

A design is a set of dianhydrides and a set of diamines with mole fractions, each set summing to
exactly 1.0. Fractions are integers in units of 1/10000 until written, so the sums are exact. A
repeat unit A_i D_j gets the fraction x_i * y_j (random copolymer).

Posterior for each property: the production ensemble is the prior mean, and a Gaussian process with
a Tanimoto kernel on composition-weighted Morgan counts models its error. The GP's scale comes from
the ensemble's out-of-fold errors, so its uncertainty grows away from the training data. Training
rows carry zero residual because the ensemble already fits them; measured results passed with
--observations carry their real residuals and shift the mean.

Training copolymers have at most TRAINED_UNITS repeat units. For larger designs the ensemble is
extrapolating, so the gap between it and a mixing rule over its own homopolymer predictions (Fox for
Tg, log rule for modulus, linear for CTE) is added to the uncertainty, and the row is flagged.

 1. NSGA-II on the posterior mean gives the predicted Pareto set.
 2. NSGA-II on mean + kappa * sd, seeded with step 1, explores uncertain regions.
 3. Expected hypervolume improvement picks a batch of experiments from everything evaluated, with
    a kriging-believer update so the batch does not repeat itself.

Examples:
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --constraint 'Tg>=300'
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --fix-acid 'PMDA=0.5' --fix-diamine ODA
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --fix-acid 'PMDA=0.2-0.6' --n-diamines 3
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --n-acids 2 --n-diamines 2 --acid-candidates 'PMDA;BPDA;6FDA;ODPA'
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --observations measured.csv
 .venv/bin/python polyimide_optimizer/mobo_copolyimide.py --check-calibration

Fix syntax: NAME (species fixed, fraction free), NAME=0.5 (fraction fixed), NAME=0.2-0.6 (range).
NAME is an abbreviation, a TCI name, product number, CAS number, or SMILES.
Candidates: acid_candidates.csv and diamine_candidates.csv (use=False skips a row), or the
--acid-candidates and --diamine-candidates lists, which take priority for that run.
--observations CSV columns: acids and diamines ('PMDA=0.5;BPDA=0.5') and any of Tg, modulus, CTE.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,re,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from scipy.stats import norm,qmc,spearmanr
RDLogger.DisableLog('rdApp.*')
ROOT=Path(__file__).resolve().parent.parent
MODELS=ROOT/'polyimide_models'
TCI=ROOT/'tci_polyimide'
sys.path[:0]=[str(MODELS),str(TCI)]
from polyimide_psmiles import make_polyimide,MonomerError,parent_mol

UNITS=10000
TARGETS={'Tg':'Tg_film','modulus':'tensile_modulus_film_clean','CTE':'CTE_film'}
LOG_TARGETS={'tensile_modulus_film_clean'}
Z90=1.645
KERNEL_POWERS=[1,2,3,5,8,16]
SIGNAL_SHARES=[.1,.3,.5,.7,.9]
EHVI_POINTS=1024
TRAINED_UNITS=3
BATCH_POOL=300
FP_GENERATOR=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

def canonical(smiles):
    return Chem.MolToSmiles(parent_mol(smiles)[0])

class Side:
    """Candidate monomers for one side of the imide, and the rules for choosing among them."""
    def __init__(self,kind,table,min_kinds,max_kinds,min_fraction,step):
        self.kind=kind;self.min_kinds=min_kinds;self.max_kinds=max_kinds
        self.min_units=round(min_fraction*UNITS);self.step=round(step*UNITS)
        if self.min_units<=0 or self.step<=0:raise SystemExit('--min-fraction and --step must be positive.')
        self.monomers=[];self.by_smiles={}
        self.allowed=None
        for row in table:
            self.add(row['smiles'],row.get('name',''),row.get('tci_product_numbers',''),row.get('cas_rn',''),row.get('abbreviation',''),row.get('other_abbreviations',''))
        self.fixed=[]  # (monomer index, lower units, upper units)
    def restrict(self,names):
        """Limit the free search to the named monomers; fixed and observed ones can still be named."""
        self.allowed={self.resolve(n) for n in names if n.strip()}
        self.free=[]
    def add(self,smiles,name='',numbers='',cas='',abbreviation='',others=''):
        key=canonical(smiles)
        if key not in self.by_smiles:
            self.by_smiles[key]=len(self.monomers)
            self.monomers.append(dict(smiles=key,name=abbreviation or name or key,full_name=name or key,numbers=numbers,cas=cas,
                                      keys={k.casefold() for k in [abbreviation,*others.split(';')] if k}))
        return self.by_smiles[key]
    def resolve(self,text):
        """Find a monomer by abbreviation, name, product number, CAS or SMILES; add a new SMILES."""
        wanted=text.strip();folded=wanted.casefold()
        exact=[i for i,m in enumerate(self.monomers) if folded in m['keys'] or folded==m['full_name'].casefold() or folded==m['cas'].casefold() or folded in m['numbers'].casefold().split(';')]
        if len(exact)==1:return exact[0]
        partial=[i for i,m in enumerate(self.monomers) if folded in m['full_name'].casefold()]
        if len(partial)==1:return partial[0]
        if len(partial)>1:
            raise SystemExit(f'"{text}" matches several {self.kind}s: '+'; '.join(self.monomers[i]['name'] for i in partial[:8]))
        if Chem.MolFromSmiles(wanted) is None:raise SystemExit(f'Unknown {self.kind}: {text}')
        return self.add(wanted)
    def set_fixed(self,specs):
        pool=len(self.monomers);seen=set()
        for spec in specs:
            index,low,high=self.parse_fix(spec)
            if index in seen:raise SystemExit(f'{self.kind} fixed twice: {spec}')
            seen.add(index);self.fixed.append((index,low,high))
        lows=sum(f[1] for f in self.fixed);highs=sum(f[2] for f in self.fixed)
        if lows>UNITS:raise SystemExit(f'Fixed {self.kind} fractions exceed 1.0.')
        self.free=[i for i in range(pool) if i not in seen and (self.allowed is None or i in self.allowed)]
        room=self.max_kinds-len(self.fixed)
        if room<0:raise SystemExit(f'More fixed {self.kind}s than --max-{self.kind}s allows.')
        self.free_max=min(room,(UNITS-lows)//self.min_units,len(self.free))
        self.free_min=max(self.min_kinds-len(self.fixed),1 if highs<UNITS else 0,0)
        if self.free_min>self.free_max:
            raise SystemExit(f'No valid {self.kind} composition: fixed fractions total {highs/UNITS:.4g} at most, and {self.free_max} more kinds fit.')
    def parse_fix(self,spec):
        """Split NAME=a or NAME=a-b, treating a trailing '=1' of a kekulized SMILES as part of the name."""
        match=re.fullmatch(r'(.+?)=([0-9.]+)(?:-([0-9.]+))?',spec.strip())
        looks_like_smiles=match and Chem.MolFromSmiles(match.group(1)) is None and Chem.MolFromSmiles(spec.strip()) is not None
        if not match or looks_like_smiles:
            return self.resolve(spec),self.min_units,UNITS
        low=float(match.group(2));high=float(match.group(3)) if match.group(3) else low
        if not 0<low<=high<=1:raise SystemExit(f'Fractions must satisfy 0 < low <= high <= 1: {spec}')
        return self.resolve(match.group(1)),round(low*UNITS),round(high*UNITS)
    def repair(self,species,weights):
        """Turn raw weights into integer fractions that respect every bound and sum to UNITS."""
        lows=[f[1] for f in self.fixed]+[self.min_units]*(len(species)-len(self.fixed))
        highs=[f[2] for f in self.fixed]+[UNITS]*(len(species)-len(self.fixed))
        units=list(lows);left=UNITS-sum(units)
        total=sum(weights) or 1.
        goal=[UNITS*w/total for w in weights]
        while left>0:
            open_slots=[i for i in range(len(units)) if units[i]<highs[i]]
            if not open_slots:raise ValueError(f'Cannot complete {self.kind} composition')
            i=max(open_slots,key=lambda k:goal[k]-units[k])
            add=min(self.step,highs[i]-units[i],left)
            units[i]+=add;left-=add
        return tuple(species),tuple(units)
    def random(self,rng):
        count=int(rng.integers(self.free_min,self.free_max+1))
        chosen=[self.free[i] for i in rng.choice(len(self.free),count,replace=False)] if count else []
        return self.repair([f[0] for f in self.fixed]+chosen,rng.dirichlet(np.ones(len(self.fixed)+count)).tolist())
    def crossover(self,first,second,rng):
        keep=len(self.fixed)
        table=dict(zip(*second));table.update({s:(table.get(s,u)+u)/2 for s,u in zip(*first)})
        pool=[s for s in table if s not in first[0][:keep]]
        count=int(np.clip(rng.choice([len(first[0]),len(second[0])])-keep,self.free_min,self.free_max))
        count=min(count,len(pool))
        chosen=[pool[i] for i in rng.choice(len(pool),count,replace=False)] if count else []
        while len(chosen)<self.free_min:
            extra=self.free[int(rng.integers(len(self.free)))]
            if extra not in chosen:chosen.append(extra)
        species=list(first[0][:keep])+chosen
        return self.repair(species,[max(table.get(s,self.min_units),1) for s in species])
    def mutate(self,gene,rng,rate):
        species=list(gene[0]);weights=[float(u) for u in gene[1]];keep=len(self.fixed)
        if len(species)>keep and rng.random()<rate:
            options=[i for i in self.free if i not in species]
            if options:species[int(rng.integers(keep,len(species)))]=options[int(rng.integers(len(options)))]
        if rng.random()<rate/2 and len(species)-keep<self.free_max:
            options=[i for i in self.free if i not in species]
            if options:
                species.append(options[int(rng.integers(len(options)))]);weights.append(float(np.mean(weights)))
        if rng.random()<rate/2 and len(species)-keep>self.free_min:
            drop=int(rng.integers(keep,len(species)));del species[drop];del weights[drop]
        if rng.random()<rate:weights=[w*float(rng.lognormal(0,.5)) for w in weights]
        return self.repair(species,weights)
    def label(self,gene,field='name'):
        return ' / '.join(self.monomers[s][field] for s in gene[0])

class Space:
    def __init__(self,acids,diamines):
        self.acids=acids;self.diamines=diamines;self.pairs={}
    def repeat_unit(self,acid,diamine):
        key=acid,diamine
        if key not in self.pairs:
            psmiles,alt,_=make_polyimide(self.acids.monomers[acid]['smiles'],self.diamines.monomers[diamine]['smiles'])
            self.pairs[key]=(psmiles,bool(alt))
        return self.pairs[key]
    def composition(self,design):
        """Repeat units with fractions x_i * y_j; the exact products sum to 1."""
        (a,x),(d,y)=design
        units=[self.repeat_unit(i,j)[0] for i in a for j in d]
        fractions=[p*q/UNITS**2 for p in x for q in y]
        return units,fractions
    def ambiguous(self,design):
        return any(self.repeat_unit(i,j)[1] for i in design[0][0] for j in design[1][0])
    def random(self,rng):
        return self.acids.random(rng),self.diamines.random(rng)
    def child(self,first,second,rng,rate):
        a=self.acids.mutate(self.acids.crossover(first[0],second[0],rng),rng,rate)
        d=self.diamines.mutate(self.diamines.crossover(first[1],second[1],rng),rng,rate)
        return a,d

def unit_fingerprint(smiles,cache={}):
    if smiles not in cache:
        counts=FP_GENERATOR.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
        cache[smiles]=counts.astype(np.float32)
    return cache[smiles]

def composition_fingerprint(units,fractions):
    return sum(f*unit_fingerprint(s) for s,f in zip(units,fractions))

def tanimoto(A,B):
    product=A@B.T
    return product/((A*A).sum(1)[:,None]+(B*B).sum(1)[None]-product+1e-9)

class ResidualGP:
    """Zero-mean GP on the ensemble's error, in model space (log10 for modulus)."""
    def __init__(self,X,residuals):
        self.X=X;similarity=tanimoto(X,X);total=float(np.mean(residuals**2));best=-np.inf
        for power in KERNEL_POWERS:
            # One eigendecomposition per power makes every signal share cheap to score.
            values,vectors=np.linalg.eigh(similarity**power);projected=vectors.T@residuals
            for share in SIGNAL_SHARES:
                spectrum=share*total*np.maximum(values,0)+(1-share)*total+1e-8
                likelihood=-.5*np.sum(projected**2/spectrum)-.5*np.log(spectrum).sum()
                if likelihood>best:best,self.power,self.signal,self.noise=likelihood,power,share*total,(1-share)*total
        self.condition(X,np.zeros(len(X)))
    def condition(self,X,residuals):
        self.X=X
        K=self.signal*tanimoto(X,X)**self.power+(self.noise+1e-8)*np.eye(len(X))
        self.inverse=np.linalg.inv(K);self.alpha=self.inverse@residuals
    def add_observations(self,X,residuals):
        self.condition(np.vstack([self.X,X]),np.r_[np.zeros(len(self.X)),residuals])
    def cross(self,A):
        return self.signal*tanimoto(A,self.X)**self.power
    def predict(self,A):
        """Mean and latent sd; the sd of a new measurement also includes self.noise."""
        K=self.cross(A)
        variance=self.signal-np.einsum('ij,ij->i',K@self.inverse,K)
        return K@self.alpha,np.sqrt(np.maximum(variance,1e-10))
    def covariance(self,A):
        K=self.cross(A)
        return self.signal*tanimoto(A,A)**self.power-K@self.inverse@K.T

def to_model_space(target,values):
    return np.log10(np.maximum(values,1e-3)) if target in LOG_TARGETS else values
def from_model_space(target,values):
    return 10**values if target in LOG_TARGETS else values

def training_set(target,data):
    """Composition fingerprints and out-of-fold ensemble errors for one target."""
    oof=pd.read_csv(MODELS/target/'oof10_predictions.tsv',sep='\t',encoding='utf-8-sig',keep_default_na=False)
    by_id={r['row_id']:r for r in data[target]}
    rows=[by_id[i] for i in oof.row_id]
    X=np.array([composition_fingerprint(r['components'],np.array(r['mole_fractions'])/sum(r['mole_fractions'])) for r in rows])
    residuals=to_model_space(target,oof.observed.to_numpy(float))-to_model_space(target,oof.ensemble_prediction.to_numpy(float))
    return X,residuals,oof.fold.to_numpy(int)

def mixing_rule(target,values,fractions,weights):
    """Combine homopolymer values in model space: Fox for Tg, linear for log modulus and CTE."""
    if target.startswith('Tg'):
        masses=fractions*weights;masses=masses/masses.sum()
        return 1/np.sum(masses/np.maximum(values+273.15,1.))-273.15
    return float(np.sum(fractions*values))

class Surrogate:
    def __init__(self,space,targets,sequence):
        data=json.loads((MODELS/'datasets.json').read_text());self.space=space;self.targets=targets;self.sequence=sequence
        self.models={};self.gps={};self.memo={};self.homopolymers={};self.mixed={};builder=None
        for target in targets:
            model=joblib.load(MODELS/target/'ensemble_production.joblib')
            builder=builder or model.builder;model.builder=builder;self.models[target]=model
            X,residuals,_=training_set(target,data)
            self.gps[target]=ResidualGP(X,residuals)
            gp=self.gps[target]
            print(f'{target}: GP power {gp.power}, sd far from data {np.sqrt(gp.signal+gp.noise):.3g} (model space), {len(X)} rows',flush=True)
    def prior(self,compositions):
        records=[dict(components=u,mole_fractions=f,sequence_type=self.sequence) for u,f in compositions]
        return {t:to_model_space(t,self.models[t].predict(records)) for t in self.targets}
    def predict_homopolymers(self,smiles):
        """Predict every new repeat unit in one batch; the models are slow one row at a time."""
        missing=[s for s in dict.fromkeys(smiles) if s not in self.homopolymers]
        if not missing:return
        prior=self.prior([([s],[1.]) for s in missing])
        for i,s in enumerate(missing):
            self.homopolymers[s]=({t:float(prior[t][i]) for t in self.targets},Descriptors.MolWt(Chem.MolFromSmiles(s)))
    def mixing_estimate(self,units,fractions):
        """Mixing-rule value per target, in model space."""
        known=[self.homopolymers[s] for s in units];x=np.array(fractions);w=np.array([k[1] for k in known])
        return {t:mixing_rule(t,np.array([k[0][t] for k in known]),x,w) for t in self.targets}
    def add_observations(self,items):
        """items: (design, {target: measured value}). Prior means are subtracted to get residuals."""
        for target in self.targets:
            rows=[(d,v[target]) for d,v in items if target in v]
            if not rows:continue
            compositions=[self.space.composition(d) for d,_ in rows]
            prior=self.prior(compositions)[target]
            residuals=to_model_space(target,np.array([v for _,v in rows],dtype=float))-prior
            X=np.array([composition_fingerprint(*c) for c in compositions])
            self.gps[target].add_observations(X,residuals)
            print(f'{target}: {len(rows)} observations, mean residual {residuals.mean():.3g} (model space)',flush=True)
        self.memo.clear()
    def evaluate(self,designs):
        """Return posterior mean and latent sd arrays (designs x targets), with memoization."""
        missing=list(dict.fromkeys(d for d in designs if d not in self.memo))
        if missing:
            compositions=[self.space.composition(d) for d in missing]
            prior=self.prior(compositions);X=np.array([composition_fingerprint(*c) for c in compositions])
            parts=[self.gps[t].predict(X) for t in self.targets]
            self.predict_homopolymers([s for u,_ in compositions if len(u)>TRAINED_UNITS for s in u])
            for i,d in enumerate(missing):
                gap=np.zeros(len(self.targets));units,fractions=compositions[i]
                if len(units)>TRAINED_UNITS:
                    mix=self.mixing_estimate(units,fractions);self.mixed[d]=mix
                    gap=np.array([prior[t][i]-mix[t] for t in self.targets])
                self.memo[d]=(np.array([prior[t][i]+parts[k][0][i] for k,t in enumerate(self.targets)]),
                              np.sqrt(np.array([parts[k][1][i]**2 for k in range(len(self.targets))])+gap**2))
        return np.array([self.memo[d][0] for d in designs]),np.array([self.memo[d][1] for d in designs])

class Problem:
    """Objectives and constraints turned into 'maximize' arrays in model space."""
    def __init__(self,objectives,constraints):
        self.objectives=objectives;self.constraints=constraints  # (target,sign), (target,sign,bound)
        self.penalty=lambda designs:np.zeros(len(designs))
        self.targets=list(dict.fromkeys([t for t,_ in objectives]+[c[0] for c in constraints]))
        self.index=[self.targets.index(t) for t,_ in objectives]
        self.signs=np.array([s for _,s in objectives],dtype=float)
    def utilities(self,mean,sd,kappa=0.):
        return (mean[:,self.index]*self.signs+kappa*sd[:,self.index]) if len(mean) else np.empty((0,len(self.index)))
    def violation(self,mean):
        total=np.zeros(len(mean))
        for target,sign,bound in self.constraints:
            total+=np.maximum(0,sign*(bound-mean[:,self.targets.index(target)]))
        return total
    def feasibility(self,mean,sd):
        probability=np.ones(len(mean))
        for target,sign,bound in self.constraints:
            k=self.targets.index(target)
            probability*=norm.cdf(sign*(mean[:,k]-bound)/sd[:,k])
        return probability

def dominance_ranks(values,violation):
    """Non-domination rank (maximize); infeasible designs rank after all feasible ones by violation."""
    ranks=np.full(len(values),-1);feasible=np.where(violation<=0)[0];level=0;remaining=feasible
    while len(remaining):
        v=values[remaining]
        dominated=np.array([np.any(np.all(v>=row,axis=1)&np.any(v>row,axis=1)) for row in v])
        ranks[remaining[~dominated]]=level;remaining=remaining[dominated];level+=1
    for i in np.where(violation>0)[0][np.argsort(violation[violation>0],kind='stable')]:
        ranks[i]=level;level+=1
    return ranks

def crowding(values):
    distance=np.zeros(len(values))
    for k in range(values.shape[1]):
        order=np.argsort(values[:,k]);span=values[order[-1],k]-values[order[0],k]
        distance[order[[0,-1]]]=np.inf
        if span>0:distance[order[1:-1]]+=(values[order[2:],k]-values[order[:-2],k])/span
    return distance

def select_survivors(values,violation,size):
    ranks=dominance_ranks(values,violation);chosen=[];rank_order=np.zeros(len(values));crowd=np.zeros(len(values))
    for level in np.unique(ranks):
        members=np.where(ranks==level)[0];crowd[members]=crowding(values[members])
        if len(chosen)+len(members)<=size:chosen+=members.tolist()
        else:
            chosen+=members[np.argsort(-crowd[members],kind='stable')][:size-len(chosen)].tolist();break
    return np.array(chosen),ranks,crowd

def hypervolume(front,box_low,box_high,points):
    """Share of the box, as a Monte Carlo estimate, that the front dominates."""
    if not len(front):return 0.
    grid=box_low+points*(box_high-box_low)
    return float(np.any(np.all(front[None]>=grid[:,None],axis=2),axis=1).mean())

def nsga2(space,surrogate,problem,kappa,rng,population,generations,mutation,seeds,label):
    designs=list(dict.fromkeys(seeds))[:population]
    while len(designs)<population:
        d=space.random(rng)
        if d not in designs:designs.append(d)
    for generation in range(generations+1):
        mean,sd=surrogate.evaluate(designs)
        values=problem.utilities(mean,sd,kappa);violation=problem.violation(mean)+problem.penalty(designs)
        chosen,ranks,crowd=select_survivors(values,violation,population)
        designs=[designs[i] for i in chosen];ranks=ranks[chosen];crowd=crowd[chosen]
        if generation%10==0 or generation==generations:
            print(f'{label} gen {generation}: {int((ranks==0).sum())} on front, {int((violation[chosen]<=0).sum())} feasible, {len(surrogate.memo)} evaluated',flush=True)
        if generation==generations:break
        def tournament():
            i,j=rng.integers(len(designs),size=2)
            return designs[i] if (ranks[i],-crowd[i])<=(ranks[j],-crowd[j]) else designs[j]
        children=[];known=set(designs);attempts=0
        while len(children)<population and attempts<population*20:
            attempts+=1
            child=space.child(tournament(),tournament(),rng,mutation)
            if child not in known:known.add(child);children.append(child)
        designs=designs+children
    return designs

def pareto_mask(values,violation):
    ranks=dominance_ranks(values,violation)
    return (ranks==0)&(violation<=0)

def select_batch(surrogate,problem,designs,mean,sd,mask,size,box_margin=.1):
    """Greedy batch by expected hypervolume improvement with kriging-believer variance updates."""
    values=problem.utilities(mean,sd);latent=sd[:,problem.index];feasible=problem.feasibility(mean,sd)
    front=values[mask] if mask.any() else values[:0]
    reference=front.min(0) if len(front) else values.min(0)
    span=(front.max(0) if len(front) else values.max(0))-reference
    low=reference-box_margin*(span+1e-6);high=np.maximum((values+3*latent).max(0),low+1e-6)
    grid=low+qmc.Sobol(values.shape[1],seed=0).random(EHVI_POINTS)*(high-low)
    if len(front):grid=grid[~np.any(np.all(front[None]>=grid[:,None],axis=2),axis=1)]
    def ehvi(mu,sigma,points):
        score=np.zeros(len(mu))
        for start in range(0,len(mu),256):
            z=(mu[start:start+256,None]-points[None])/np.maximum(sigma[start:start+256,None],1e-6)
            score[start:start+256]=norm.cdf(z).prod(2).mean(1)
        return score
    first=ehvi(values,latent,grid)*feasible
    pool=np.argsort(-first)[:BATCH_POOL];mu=values[pool];sigma2=latent[pool]**2;picked=[];gains=[]
    X=np.array([composition_fingerprint(*surrogate.space.composition(designs[i])) for i in pool])
    covariances=[surrogate.gps[problem.targets[k]].covariance(X) for k in problem.index]
    noises=[surrogate.gps[problem.targets[k]].noise for k in problem.index]
    for _ in range(min(size,len(pool))):
        if not len(grid):break
        score=ehvi(mu,np.sqrt(np.maximum(sigma2,1e-10)),grid)*feasible[pool]
        score[picked]=-1
        best=int(np.argmax(score));picked.append(best);gains.append(float(score[best]))
        # Believe the mean at the picked point: the front grows and nearby variance shrinks.
        grid=grid[~np.all(mu[best]>=grid,axis=1)]
        for k in range(mu.shape[1]):
            c=covariances[k];update=np.outer(c[:,best],c[:,best])/(c[best,best]+noises[k])
            covariances[k]=c-update;sigma2[:,k]=np.diag(covariances[k])
    return [int(pool[i]) for i in picked],gains

def table(space,surrogate,problem,designs,mean,sd,training_max):
    rows=[]
    for i,design in enumerate(designs):
        (a,x),(d,y)=design
        if sum(x)!=UNITS or sum(y)!=UNITS:raise AssertionError(f'Fractions do not sum to 1: {design}')
        units,fractions=space.composition(design)
        row=dict(acids=space.acids.label(design[0]),acid_fractions=' / '.join(f'{u/UNITS:.4g}' for u in x),
                 diamines=space.diamines.label(design[1]),diamine_fractions=' / '.join(f'{u/UNITS:.4g}' for u in y),
                 acid_sum=sum(x)/UNITS,diamine_sum=sum(y)/UNITS,n_repeat_units=len(units))
        for k,target in enumerate(problem.targets):
            name=short_name(target);gp=surrogate.gps[target]
            total=np.sqrt(sd[i,k]**2+gp.noise)
            row[name+'_pred']=from_model_space(target,mean[i,k])
            row[name+'_lo90']=from_model_space(target,mean[i,k]-Z90*total)
            row[name+'_hi90']=from_model_space(target,mean[i,k]+Z90*total)
            if design in surrogate.mixed:row[name+'_mixing_rule']=from_model_space(target,surrogate.mixed[design][target])
        row['composition_extrapolated']=len(units)>TRAINED_UNITS
        row['min_similarity_to_training']=training_max(units)
        row['regioisomer_ambiguous']=space.ambiguous(design)
        row['acid_full_names']=space.acids.label(design[0],'full_name');row['diamine_full_names']=space.diamines.label(design[1],'full_name')
        row['acid_smiles']=space.acids.label(design[0],'smiles');row['diamine_smiles']=space.diamines.label(design[1],'smiles')
        row['repeat_units_json']=json.dumps(units);row['repeat_unit_fractions_json']=json.dumps(fractions)
        rows.append(row)
    return pd.DataFrame(rows)

def short_name(target):
    return next(k for k,v in TARGETS.items() if v==target)

def parse_objective(text):
    name,_,direction=text.partition(':')
    if name not in TARGETS or direction not in('max','min'):raise SystemExit(f'Objective must look like Tg:max, modulus:max or CTE:min: {text}')
    return TARGETS[name],1 if direction=='max' else -1

def parse_constraint(text):
    match=re.fullmatch(r'(\w+)\s*([<>])=?\s*([0-9.]+)',text.strip())
    if not match or match.group(1) not in TARGETS:raise SystemExit(f"Constraint must look like 'Tg>=300' or 'CTE<=40': {text}")
    target=TARGETS[match.group(1)]
    return target,1 if match.group(2)=='>' else -1,float(to_model_space(target,np.array(float(match.group(3)))))

def read_monomers(path,fluorine_free):
    rows=pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False).to_dict('records')
    rows=[r for r in rows if str(r.get('polyimide_ok','True'))=='True' and str(r.get('use','True')).strip().lower() not in('false','0','')]
    if fluorine_free:rows=[r for r in rows if not any(a.GetAtomicNum()==9 for a in parent_mol(r['smiles'])[0].GetAtoms())]
    return rows

def parse_design_side(side,text):
    pairs=[side.parse_fix(p) for p in text.split(';') if p.strip()]
    if any(low!=high for _,low,high in pairs):raise SystemExit(f'Observed fractions must be exact: {text}')
    if sum(low for _,low,_ in pairs)!=UNITS:raise SystemExit(f'Observed fractions must sum to 1.0: {text}')
    return tuple(i for i,_,_ in pairs),tuple(low for _,low,_ in pairs)

def read_observations(path,space):
    items=[]
    for row in pd.read_csv(path,encoding='utf-8-sig',keep_default_na=False).to_dict('records'):
        design=parse_design_side(space.acids,row['acids']),parse_design_side(space.diamines,row['diamines'])
        values={TARGETS[k]:float(row[k]) for k in TARGETS if k in row and str(row[k]).strip()!=''}
        items.append((design,values))
    return items

def check_calibration():
    """Hold out each OOF fold from the GP and score its uncertainty on the held-out errors."""
    data=json.loads((MODELS/'datasets.json').read_text());results={}
    for name,target in TARGETS.items():
        X,residuals,folds=training_set(target,data);gp=ResidualGP(X,residuals);z=np.zeros(len(X));sds=np.zeros(len(X))
        for fold in np.unique(folds):
            held=folds==fold;gp.condition(X[~held],np.zeros((~held).sum()))
            sds[held]=np.sqrt(gp.predict(X[held])[1]**2+gp.noise);z[held]=residuals[held]/sds[held]
        result=dict(rows=len(X),power=gp.power,coverage_90=float((np.abs(z)<Z90).mean()),
                    sd_vs_error_spearman=float(spearmanr(sds,np.abs(residuals)).statistic),
                    sd_low=float(sds.min()),sd_high=float(sds.max()),rmse=float(np.sqrt((residuals**2).mean())))
        results[target]=result;print(target,json.dumps(result),flush=True)
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--objective',action='append',help='Tg:max, modulus:max or CTE:min. Repeat. Default: all three.')
    ap.add_argument('--constraint',action='append',default=[],help="e.g. 'Tg>=300' or 'CTE<=40'. Repeat.")
    ap.add_argument('--fix-acid',action='append',default=[]);ap.add_argument('--fix-diamine',action='append',default=[])
    ap.add_argument('--n-acids',type=int,help='Exact number of dianhydrides. Overrides --min/--max-acids.')
    ap.add_argument('--n-diamines',type=int,help='Exact number of diamines. Overrides --min/--max-diamines.')
    ap.add_argument('--acid-candidates',help="Search only these dianhydrides, separated by ';'.")
    ap.add_argument('--diamine-candidates',help="Search only these diamines, separated by ';'.")
    ap.add_argument('--min-acids',type=int,default=1);ap.add_argument('--max-acids',type=int,default=2)
    ap.add_argument('--min-diamines',type=int,default=1);ap.add_argument('--max-diamines',type=int,default=2)
    ap.add_argument('--min-fraction',type=float,default=.05);ap.add_argument('--step',type=float,default=.05)
    ap.add_argument('--acid-file',type=Path,default=ROOT/'polyimide_optimizer/acid_candidates.csv',help='Candidate list. Rows with use=False are skipped.')
    ap.add_argument('--diamine-file',type=Path,default=ROOT/'polyimide_optimizer/diamine_candidates.csv',help='Candidate list. Rows with use=False are skipped.')
    ap.add_argument('--exclude-fluorine',action='store_true')
    ap.add_argument('--min-similarity',type=float,default=0.,help='Reject designs whose least familiar repeat unit is below this Tanimoto similarity to training data.')
    ap.add_argument('--sequence',default='random')
    ap.add_argument('--observations',type=Path)
    ap.add_argument('--population',type=int,default=120);ap.add_argument('--generations',type=int,default=60)
    ap.add_argument('--explore-generations',type=int,default=30);ap.add_argument('--mutation',type=float,default=.3)
    ap.add_argument('--kappa',type=float,default=1.);ap.add_argument('--batch',type=int,default=10)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--output-dir',type=Path,default=ROOT/'polyimide_optimizer/output')
    ap.add_argument('--check-calibration',action='store_true')
    args=ap.parse_args()
    if args.check_calibration:
        check_calibration();return
    start=time.time();rng=np.random.default_rng(args.seed)
    objectives=[parse_objective(o) for o in args.objective or ['Tg:max','modulus:max','CTE:min']]
    constraints=[parse_constraint(c) for c in args.constraint]
    problem=Problem(objectives,constraints)
    if args.n_acids:args.min_acids=args.max_acids=args.n_acids
    if args.n_diamines:args.min_diamines=args.max_diamines=args.n_diamines
    acids=Side('acid',read_monomers(args.acid_file,args.exclude_fluorine),args.min_acids,args.max_acids,args.min_fraction,args.step)
    diamines=Side('diamine',read_monomers(args.diamine_file,args.exclude_fluorine),args.min_diamines,args.max_diamines,args.min_fraction,args.step)
    if args.acid_candidates:acids.restrict(args.acid_candidates.split(';'))
    if args.diamine_candidates:diamines.restrict(args.diamine_candidates.split(';'))
    acids.set_fixed(args.fix_acid);diamines.set_fixed(args.fix_diamine)
    space=Space(acids,diamines)
    for side,other,spec in [(acids,diamines,args.fix_acid),(diamines,acids,args.fix_diamine)]:
        for index,_,_ in side.fixed:
            # A fixed monomer given as new SMILES must be able to form an imide.
            probe=other.monomers[other.free[0] if other.free else other.fixed[0][0]]['smiles']
            pair=(side.monomers[index]['smiles'],probe) if side is acids else (probe,side.monomers[index]['smiles'])
            try:make_polyimide(*pair)
            except MonomerError as error:raise SystemExit(str(error))
    surrogate=Surrogate(space,problem.targets,args.sequence)
    observed=[]
    if args.observations:
        observed=read_observations(args.observations,space);surrogate.add_observations(observed)
    train_fps={t:np.array([unit_fingerprint(s) for s in sorted({s for r in json.loads((MODELS/'datasets.json').read_text())[t] for s in r['components']})]) for t in problem.targets}
    similarity_cache={}
    def least_similar(units):
        """Lowest, over repeat units and targets, of the best similarity to that target's training units."""
        lowest=1.
        for s in units:
            if s not in similarity_cache:
                f=unit_fingerprint(s)[None];similarity_cache[s]=min(float(tanimoto(f,x).max()) for x in train_fps.values())
            lowest=min(lowest,similarity_cache[s])
        return lowest
    if args.min_similarity>0:
        problem.penalty=lambda ds:np.array([float(least_similar(space.composition(d)[0])<args.min_similarity) for d in ds])
    exploit=nsga2(space,surrogate,problem,0.,rng,args.population,args.generations,args.mutation,[],'mean')
    nsga2(space,surrogate,problem,args.kappa,rng,args.population,args.explore_generations,args.mutation,exploit,'explore')
    designs=list(surrogate.memo)
    mean,sd=surrogate.evaluate(designs)
    values=problem.utilities(mean,sd);violation=problem.violation(mean)+problem.penalty(designs)
    mask=pareto_mask(values,violation)
    frame=table(space,surrogate,problem,[designs[i] for i in np.where(mask)[0]],mean[mask],sd[mask],least_similar)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame.sort_values(short_name(objectives[0][0])+'_pred',ascending=objectives[0][1]<0).to_csv(args.output_dir/'pareto_predicted.csv',index=False,encoding='utf-8-sig')
    known={d for d,_ in observed}
    blocked=problem.penalty(designs)
    open_designs=[i for i,d in enumerate(designs) if d not in known and blocked[i]==0]
    picked,gains=select_batch(surrogate,problem,[designs[i] for i in open_designs],mean[open_designs],sd[open_designs],mask[open_designs],args.batch)
    chosen=[open_designs[i] for i in picked]
    batch=table(space,surrogate,problem,[designs[i] for i in chosen],mean[chosen],sd[chosen],least_similar)
    batch.insert(0,'batch_rank',range(1,len(batch)+1));batch.insert(1,'ehvi',gains)
    batch.to_csv(args.output_dir/'next_experiments.csv',index=False,encoding='utf-8-sig')
    summary=dict(arguments={k:str(v) for k,v in vars(args).items()},evaluated=len(designs),pareto=int(mask.sum()),feasible=int((violation<=0).sum()),seconds=round(time.time()-start))
    (args.output_dir/'run_summary.json').write_text(json.dumps(summary,indent=1,ensure_ascii=False))
    print(json.dumps({k:summary[k] for k in('evaluated','pareto','feasible','seconds')}),'->',args.output_dir)

if __name__=='__main__':main()
