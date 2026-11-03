#!/usr/bin/env python3
"""Propose copolyimides that are far from the training compositions and most likely to land in a
target region of properties, such as Tg >= 350 °C and tensile modulus <= 3 GPa.

The posterior is the one in mobo_copolyimide.py: the production ensemble as prior mean, a Gaussian
process on its error (Tanimoto kernel on composition-weighted Morgan counts), and extra uncertainty
for designs with more repeat units than any training copolymer. P(region) is the probability that
the design's true properties satisfy every --region condition, treating the properties as
independent.

Distance from the training data is the Tanimoto similarity between the design's composition
fingerprint and that of its most similar training (or observed) polymer, the same similarity the GP
kernel uses. Training polymers have a nearest neighbour at 0.88 or more in 95 % of cases; a random
design is at 0.78 in the median. At --max-similarity 0.8 (default) the GP term is close to its prior,
so P(region) rests on the ensemble and its out-of-fold error: it is an honest but rough number.

 1. NSGA-II on two objectives: maximize log P(region), minimize similarity to training. Designs above
    --max-similarity count as infeasible. The result is the trade-off between being likely and being
    new, written to region_candidates.csv.
 2. A batch for the lab: greedy, maximizing the probability that at least one design in the batch
    lands in the region. In the joint samples similar designs share part of their error
    (--shared-error, default 0.5, on top of the GP correlation), so the batch spreads out. At most
    --max-per-system designs per monomer set and --max-monomer-uses designs per monomer.
    P(at least one) still rests on the model being right on average; read it as optimistic.

Soft segments (long siloxane diamines and the like, listed in soft_monomers.csv) are predicted as
hard part + slope x soft weight fraction, with the slopes learned from measured soft-segment polymers
in --observations; see soft_segment.py. They enter a search only when fixed (--fix-diamine
'MESi3200=0.05-0.3') or named in a candidate list. Distance is measured to the training data only,
so designs may stay close to your own measurements.

Examples:
 .venv/bin/python polyimide_optimizer/region_search.py --region 'Tg>=350' --region 'modulus<=3'
 .venv/bin/python polyimide_optimizer/region_search.py --region 'Tg=240-260' --region 'modulus<=1.0' \
     --fix-diamine 'MESi3200=0.05-0.3' --observations polyimide_optimizer/examples/observations_siloxane_example.csv
 .venv/bin/python polyimide_optimizer/region_search.py --region 'Tg>=300' --region 'CTE<=20' --max-similarity 0.7
 .venv/bin/python polyimide_optimizer/region_search.py --region 'Tg>=350' --region 'modulus<=3' --fix-diamine ODA --observations measured.csv

Outputs (in --output-dir, default polyimide_optimizer/output_region): region_candidates.csv,
region_batch.csv, run_summary.json.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import argparse,json,re,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import mobo_copolyimide as mobo
import soft_segment
from mobo_copolyimide import (ROOT,TARGETS,UNITS,Known,MonomerError,Side,Space,Surrogate,composition_fingerprint,
                              make_polyimide,nsga2,parse_constraint,read_monomers,read_observations,short_name,table,tanimoto)

LOG_FLOOR=-12.   # log10 P below this is treated as equal; keeps NSGA-II's crowding meaningful
POOL=600;POOL_PER_SYSTEM=5;SAMPLES=4000

class RegionProblem(mobo.Problem):
    """Objectives: log10 P(region) up, similarity to training down. Infeasible above max_similarity."""
    def __init__(self,region,space,known,max_similarity,invalid=None):
        super().__init__([],region)
        self.targets=list(dict.fromkeys(self.targets+list(TARGETS.values())))  # predict all three for the tables
        self.space=space;self.known=known;self.max_similarity=max_similarity;self.cache={}
        self.invalid=invalid or (lambda design:False)
        self.penalty=lambda designs:np.maximum(0,self.similarity(designs)-max_similarity)+np.array([float(self.invalid(d)) for d in designs])
        # Conditions on one property make an interval: Tg>=240 and Tg<=260 -> [240, 260].
        self.intervals={}
        for target,sign,bound in region:
            lo,hi=self.intervals.get(target,(-np.inf,np.inf))
            self.intervals[target]=(max(lo,bound),hi) if sign>0 else (lo,min(hi,bound))
        training=[i for i,r in enumerate(known.rows) if r['source']=='training']
        self.training_X=known.X[training]
    def similarity(self,designs):
        """Similarity to the nearest training polymer. Own --observations do not count: designs may
        stay close to what was just measured, but must be far from the literature data."""
        missing=[d for d in dict.fromkeys(designs) if d not in self.cache]
        if missing:
            X=np.array([composition_fingerprint(*self.space.composition(d)) for d in missing],np.float32)
            s=tanimoto(X,self.training_X).max(1)
            self.cache.update(zip(missing,s.tolist()))
        return np.array([self.cache[d] for d in designs])
    def target_probability(self,mean,sd,target):
        k=self.targets.index(target);lo,hi=self.intervals[target];s=np.maximum(sd[:,k],1e-9)
        return np.clip(norm.cdf((hi-mean[:,k])/s)-norm.cdf((lo-mean[:,k])/s),0,1)
    def log_probability(self,mean,sd):
        total=np.zeros(len(mean))
        for target in self.intervals:
            total+=np.log10(np.maximum(self.target_probability(mean,sd,target),1e-300))
        return np.maximum(np.nan_to_num(total,nan=LOG_FLOOR),LOG_FLOOR)
    def score(self,designs,mean,sd,kappa=0.):
        return np.column_stack([self.log_probability(mean,sd),-self.similarity(designs)])
    def violation(self,mean):
        return np.zeros(len(mean))  # the region is the objective here, not a constraint

def region_success(problem,draws):
    """draws: samples x designs x targets (model space) -> samples x designs, all conditions met."""
    ok=np.ones(draws.shape[:2],bool)
    for target,(lo,hi) in problem.intervals.items():
        v=draws[:,:,problem.targets.index(target)];ok&=(v>=lo)&(v<=hi)
    return ok

def select_batch(surrogate,problem,designs,mean,sd,systems,size,max_per_system,max_monomer_uses,shared,fixed,rng):
    """Greedy batch maximizing P(at least one design lands in the region) under joint samples.

    Errors are correlated by (1 - shared) * GP correlation + shared * Tanimoto similarity between the
    designs: far from the data the GP correlation vanishes, but designs that share monomers still
    share the ensemble's systematic error. A monomer (other than fixed ones) is used at most
    max_monomer_uses times, so the batch does not bet on one diamine."""
    logp=problem.log_probability(mean,sd)
    pool=[];count={}
    for i in np.argsort(-logp):
        if count.get(systems[i],0)<POOL_PER_SYSTEM:
            pool.append(int(i));count[systems[i]]=count.get(systems[i],0)+1
            if len(pool)==POOL:break
    if not pool:return [],[],[]
    X=np.array([composition_fingerprint(*surrogate.space.composition(designs[i])) for i in pool])
    draws=np.zeros((SAMPLES,len(pool),len(problem.targets)))
    for k,target in enumerate(problem.targets):
        if target not in problem.intervals:continue
        # GP correlation between designs, scaled to each design's full sd (which also holds the
        # extrapolation term), so close designs succeed or fail together. A soft-segment surrogate
        # also ties designs through the shared uncertainty of its slope.
        if hasattr(surrogate,'correlation'):G=surrogate.correlation([designs[i] for i in pool],k)
        else:
            C=surrogate.gps[target].covariance(X);d=np.sqrt(np.maximum(np.diag(C),1e-12));G=C/np.outer(d,d)
        R=(1-shared)*G+shared*tanimoto(X,X);np.fill_diagonal(R,1.);s=sd[pool,k]
        # Posterior covariances can be a hair off positive definite (near-identical designs, round-off);
        # clip negative eigenvalues instead of a Cholesky that would fail.
        w,V=np.linalg.eigh((np.outer(s,s)*R+(np.outer(s,s)*R).T)/2)
        L=V*np.sqrt(np.clip(w,0,None))
        draws[:,:,k]=mean[pool,k][None]+rng.standard_normal((SAMPLES,len(pool)))@L.T
    success=region_success(problem,draws)
    covered=np.zeros(SAMPLES,bool);picked=[];gains=[];cumulative=[];used={};uses={}
    monomers=lambda system:[('a',m) for m in system[0] if m not in fixed[0]]+[('d',m) for m in system[1] if m not in fixed[1]]
    for _ in range(min(size,len(pool))):
        score=(covered[:,None]|success).mean(0)
        for j,i in enumerate(pool):
            if j in picked or used.get(systems[i],0)>=max_per_system or any(uses.get(m,0)>=max_monomer_uses for m in monomers(systems[i])):score[j]=-1
        best=int(np.argmax(score))
        if score[best]<0:break
        gains.append(float(score[best]-covered.mean()));covered|=success[:,best]
        picked.append(best);cumulative.append(float(covered.mean()));used[systems[pool[best]]]=used.get(systems[pool[best]],0)+1
        for m in monomers(systems[pool[best]]):uses[m]=uses.get(m,0)+1
    return [pool[j] for j in picked],gains,cumulative

def describe(space,surrogate,problem,known,designs,mean,sd,tolerance):
    frame=table(space,surrogate,problem,designs,mean,sd,lambda units:np.nan,known,np.zeros((0,len(problem.index))),tolerance)
    frame=frame.drop(columns=['dominated_by_measured_front','p_beyond_measured_front','min_similarity_to_training'],errors='ignore')
    logp=problem.log_probability(mean,sd)
    frame.insert(0,'p_region',10**logp)
    for j,target in enumerate(problem.intervals):
        frame.insert(1+j,f'p_{short_name(target)}',problem.target_probability(mean,sd,target))
    frame.insert(1+len(problem.intervals),'similarity_to_training',problem.similarity(designs))
    if isinstance(surrogate,soft_segment.SoftSurrogate):
        # The soft-segment share and what the hard part alone is predicted to do.
        at=frame.columns.get_loc('n_repeat_units')+1
        frame.insert(at,'soft_weight_fraction',[surrogate.soft_weight(d) if surrogate.is_soft(d) else 0. for d in designs])
        hm,_=surrogate.hard_prediction(designs)
        for k,target in enumerate(problem.targets):
            frame.insert(at+1+k,f'hard_part_{short_name(target)}_pred',mobo.from_model_space(target,hm[:,k]))
    return frame

def write_observations(folder,space,observed):
    """Resolved measured polymers for the comparison page (compare_with_training.py)."""
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for design,values in observed:
        units,fractions=space.composition(design)
        rows.append(dict(acids=space.acids.label(design[0])+' = '+' / '.join(f'{u/UNITS:g}' for u in design[0][1]),
                         diamines=space.diamines.label(design[1])+' = '+' / '.join(f'{u/UNITS:g}' for u in design[1][1]),
                         units_json=json.dumps(units),fractions_json=json.dumps(fractions),
                         **{short_name(t):v for t,v in values.items()}))
    pd.DataFrame(rows).to_csv(folder/'observations_resolved.csv',index=False,encoding='utf-8-sig')

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--region',action='append',required=True,
                    help="A condition such as 'Tg>=350', 'modulus<=3' (GPa), 'CTE<=20', or a range 'Tg=240-260'. Repeat; all must hold.")
    ap.add_argument('--soft-file',type=Path,default=soft_segment.SOFT_FILE,help='Soft monomers (long siloxane diamines and the like); see soft_segment.py.')
    ap.add_argument('--soft-prior-sd',default='',help="Prior sd of the soft slopes per unit weight fraction, e.g. 'Tg=300,modulus=3,CTE=300' (model space; modulus in log10 GPa).")
    ap.add_argument('--max-similarity',type=float,default=.8,help='Most Tanimoto similarity to the nearest training polymer (default 0.8).')
    ap.add_argument('--fix-acid',action='append',default=[]);ap.add_argument('--fix-diamine',action='append',default=[])
    ap.add_argument('--n-acids',type=int);ap.add_argument('--n-diamines',type=int)
    ap.add_argument('--acid-candidates');ap.add_argument('--diamine-candidates')
    ap.add_argument('--min-acids',type=int,default=1);ap.add_argument('--max-acids',type=int,default=2)
    ap.add_argument('--min-diamines',type=int,default=1);ap.add_argument('--max-diamines',type=int,default=2)
    ap.add_argument('--min-fraction',type=float,default=.1);ap.add_argument('--step',type=float,default=.05)
    ap.add_argument('--acid-file',type=Path,default=HERE/'acid_candidates.csv');ap.add_argument('--diamine-file',type=Path,default=HERE/'diamine_candidates.csv')
    ap.add_argument('--exclude-fluorine',action='store_true')
    ap.add_argument('--sequence',default='random')
    ap.add_argument('--observations',type=Path,help='Measured results (same CSV as mobo_copolyimide.py); they update the GP and count as known compositions.')
    ap.add_argument('--population',type=int,default=150);ap.add_argument('--generations',type=int,default=80);ap.add_argument('--mutation',type=float,default=.3)
    ap.add_argument('--batch',type=int,default=10);ap.add_argument('--max-per-system',type=int,default=1)
    ap.add_argument('--max-monomer-uses',type=int,default=2,help='Most designs in a batch that use one monomer; fixed monomers are exempt (default 2).')
    ap.add_argument('--shared-error',type=float,default=.5,help='Share of the prediction error that similar designs have in common, for P(at least one) (default 0.5; 0 trusts the GP alone).')
    ap.add_argument('--remeasure-tolerance',type=float,default=.1,help='Only for the novelty column; see mobo_copolyimide.py.')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--output-dir',type=Path,default=HERE/'output_region')
    args=ap.parse_args()
    start=time.time();rng=np.random.default_rng(args.seed)
    region=[]
    for text in args.region:
        span=re.fullmatch(r'\s*(\w+)\s*=\s*([0-9.]+)\s*-\s*([0-9.]+)\s*',text)
        region+=[parse_constraint(f'{span.group(1)}>={span.group(2)}'),parse_constraint(f'{span.group(1)}<={span.group(3)}')] if span else [parse_constraint(text)]
    if args.n_acids:args.min_acids=args.max_acids=args.n_acids
    if args.n_diamines:args.min_diamines=args.max_diamines=args.n_diamines
    acids=Side('acid',read_monomers(args.acid_file,args.exclude_fluorine),args.min_acids,args.max_acids,args.min_fraction,args.step)
    diamines=Side('diamine',read_monomers(args.diamine_file,args.exclude_fluorine),args.min_diamines,args.max_diamines,args.min_fraction,args.step)
    soft_acids,soft_diamines=soft_segment.register(acids,diamines,soft_segment.read_soft_monomers(args.soft_file))
    if args.acid_candidates:acids.restrict(args.acid_candidates.split(';'))
    if args.diamine_candidates:diamines.restrict(args.diamine_candidates.split(';'))
    # Soft monomers join a search only when fixed or named in a candidate list: the hard monomers
    # around them are what the search is for.
    for side,soft in [(acids,soft_acids),(diamines,soft_diamines)]:
        if side.allowed is None and soft:side.allowed=set(range(len(side.monomers)))-soft
    acids.set_fixed(args.fix_acid);diamines.set_fixed(args.fix_diamine)
    space=Space(acids,diamines)
    for side,other in [(acids,diamines),(diamines,acids)]:
        for index,_,_ in side.fixed:
            probe=other.monomers[other.free[0] if other.free else other.fixed[0][0]]['smiles']
            pair=(side.monomers[index]['smiles'],probe) if side is acids else (probe,side.monomers[index]['smiles'])
            try:make_polyimide(*pair)
            except MonomerError as error:raise SystemExit(str(error))
    known=Known()
    probe=RegionProblem(region,space,known,args.max_similarity)
    surrogate=Surrogate(space,probe.targets,args.sequence)
    soft_used=bool(soft_acids or soft_diamines)
    if soft_used:
        prior={k:float(v) for k,v in (p.split('=') for p in args.soft_prior_sd.split(',') if p.strip())}
        surrogate=soft_segment.SoftSurrogate(surrogate,space,soft_acids,soft_diamines,prior)
    problem=RegionProblem(region,space,known,args.max_similarity,
                          invalid=(lambda d:surrogate.is_soft(d) and surrogate.hard_design(d) is None) if soft_used else None)
    observed=[]
    if args.observations:
        observed=read_observations(args.observations,space);surrogate.add_observations(observed);known.add_observations(observed,space)
        write_observations(args.output_dir,space,observed)
    soft_in_search=any(f[0] in soft_acids for f in acids.fixed)+any(f[0] in soft_diamines for f in diamines.fixed)+any(i in soft_acids for i in acids.free)+any(i in soft_diamines for i in diamines.free)
    if soft_in_search and not getattr(surrogate,'soft_rows',0):
        print('WARNING: soft monomers are in the search but no measured soft-segment polymer was given (--observations). '
              'Their effect is only the wide prior; the probabilities for them mean little.',flush=True)
    nsga2(space,surrogate,problem,0.,rng,args.population,args.generations,args.mutation,[],'region')
    designs=list(surrogate.memo)
    mean,sd=surrogate.evaluate(designs)
    values=problem.score(designs,mean,sd);far=problem.penalty(designs)<=0
    mask=mobo.pareto_mask(values,np.where(far,0.,1.))
    order=np.where(mask)[0]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    frame=describe(space,surrogate,problem,known,[designs[i] for i in order],mean[order],sd[order],args.remeasure_tolerance)
    frame.sort_values('p_region',ascending=False).to_csv(args.output_dir/'region_candidates.csv',index=False,encoding='utf-8-sig')

    observed_designs={d for d,_ in observed}
    open_designs=[i for i,d in enumerate(designs) if far[i] and d not in observed_designs]
    systems=[space.system(designs[i]) for i in open_designs]
    picked,gains,cumulative=select_batch(surrogate,problem,[designs[i] for i in open_designs],mean[open_designs],sd[open_designs],systems,args.batch,args.max_per_system,args.max_monomer_uses,args.shared_error,
                                            ({f[0] for f in acids.fixed},{f[0] for f in diamines.fixed}),rng)
    chosen=[open_designs[i] for i in picked]
    batch=describe(space,surrogate,problem,known,[designs[i] for i in chosen],mean[chosen],sd[chosen],args.remeasure_tolerance)
    batch.insert(0,'batch_rank',range(1,len(batch)+1));batch.insert(1,'p_at_least_one_so_far',cumulative);batch.insert(2,'gain',gains)
    batch.to_csv(args.output_dir/'region_batch.csv',index=False,encoding='utf-8-sig')
    p_far=10**problem.log_probability(mean[far],sd[far]) if far.any() else np.zeros(0)
    summary=dict(arguments={k:str(v) for k,v in vars(args).items()},evaluated=len(designs),far_enough=int(far.sum()),trade_off=int(mask.sum()),
                 best_p_region_far=float(p_far.max()) if len(p_far) else 0.,batch=len(batch),
                 p_at_least_one=cumulative[-1] if cumulative else 0.,seconds=round(time.time()-start))
    if soft_used:summary['soft_slopes']=surrogate.slopes();summary['soft_polymers_measured']=surrogate.soft_rows
    (args.output_dir/'run_summary.json').write_text(json.dumps(summary,indent=1,ensure_ascii=False))
    print(json.dumps({k:summary[k] for k in('evaluated','far_enough','trade_off','best_p_region_far','batch','p_at_least_one','seconds')}),'->',args.output_dir)

if __name__=='__main__':main()
