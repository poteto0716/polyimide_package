#!/usr/bin/env python3
"""Checks for mobo_copolyimide.py that need no trained model.
 .venv/bin/python polyimide_optimizer/test_mobo.py
"""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mobo_copolyimide import UNITS,Side,Space,dominance_ranks,read_monomers,mixing_rule,unit_key,dominated_by,parse_design_side

def sides(fix_acid=(),fix_diamine=(),max_kinds=2,min_kinds=1,min_fraction=.05,step=.05):
    a=Side('acid',read_monomers(HERE/'acid_candidates.csv',False),min_kinds,max_kinds,min_fraction,step)
    d=Side('diamine',read_monomers(HERE/'diamine_candidates.csv',False),min_kinds,max_kinds,min_fraction,step)
    a.set_fixed(list(fix_acid));d.set_fixed(list(fix_diamine))
    return Space(a,d)

def check(space,rng,rounds=1500):
    """Random designs and GA children must obey every rule on each side."""
    population=[space.random(rng) for _ in range(40)]
    for _ in range(rounds):
        i,j=rng.integers(len(population),size=2)
        child=space.child(population[i],population[j],rng,.5);population[int(rng.integers(len(population)))]=child
        for side,(species,units) in zip((space.acids,space.diamines),child):
            assert sum(units)==UNITS,child
            assert len(set(species))==len(species),child
            assert len(species)==len(units)
            assert side.min_kinds<=len(species)<=side.max_kinds,child
            for (index,low,high),s,u in zip(side.fixed,species,units):
                assert s==index and low<=u<=high,child
            assert all(u>=side.min_units for s,u in zip(species,units) if s not in [f[0] for f in side.fixed]),child
    return population

def main():
    rng=np.random.default_rng(0)
    pop=check(sides(),rng);print('free search ok',len({d for d in pop}))
    pop=check(sides(fix_acid=['PMDA=0.5'],fix_diamine=['ODA']),rng)
    assert all(d[0][1][0]==5000 and len(d[0][0])==2 for d in pop);print('fixed acid fraction and diamine species ok')
    pop=check(sides(fix_acid=['PMDA=0.2-0.6','BPDA=0.1'],max_kinds=3,min_kinds=3),rng);print('range with exact kinds ok')
    pop=check(sides(fix_acid=['PMDA=0.33','6FDA=0.67'],fix_diamine=['ODA=1']),rng)
    assert all(d[0][1]==(3300,6700) and d[1][1]==(UNITS,) for d in pop);print('fully fixed design ok')
    assert all(sum(p*q for p in d[0][1] for q in d[1][1])==UNITS**2 for d in pop)
    pop=check(sides(max_kinds=3,min_kinds=3),rng)
    units,fractions=Space.composition(sides(max_kinds=3,min_kinds=3),pop[0])
    assert len(units)==9 and abs(sum(fractions)-1)<1e-12;print('3 x 3 gives 9 repeat units summing to 1')
    for bad in [dict(fix_acid=['PMDA=0.7','BPDA=0.7']),dict(fix_acid=['PMDA=0.3','BPDA=0.3']),dict(fix_acid=['nosuchmonomer'])]:
        try:sides(**bad);raise AssertionError(bad)
        except SystemExit as error:print('rejected:',error)
    probe=sides()
    for text,full in [('PMDA','Pyromellitic dianhydride'),('s-BPDA',"5,5'-Bi-2-benzofuran-1,1',3,3'-tetrone"),('BPDA',"5,5'-Bi-2-benzofuran-1,1',3,3'-tetrone"),('6fda',"4,4'-(Hexafluoroisopropylidene)diphthalic anhydride")]:
        assert probe.acids.monomers[probe.acids.resolve(text)]['full_name']==full,text
    for text,full in [('ODA',"4,4'-Diaminodiphenyl Ether"),('PDA','P-Phenylenediamine'),('TFMB',"(1,1'-Biphenyl)-4,4'-diamine, 2,2'-bis(trifluoromethyl)-")]:
        assert probe.diamines.monomers[probe.diamines.resolve(text)]['full_name']==full,text
    print('abbreviations resolve')
    limited=Side('diamine',read_monomers(HERE/'diamine_candidates.csv',False),2,2,.05,.05)
    limited.restrict(['p-PDA','TFMB']);limited.set_fixed(['ODA']);oda=limited.resolve('ODA')
    assert limited.free==sorted(limited.resolve(n) for n in ['p-PDA','TFMB'])
    for _ in range(200):
        species,units=limited.random(np.random.default_rng(_))
        assert species[0]==oda and len(species)==2 and species[1] in limited.free and sum(units)==UNITS
    print('fixed monomer stays outside a restricted candidate list')
    # One composition, one design, whatever order the monomers come in.
    space=sides()
    for _ in range(300):
        d=space.random(np.random.default_rng(_))
        for side,(species,units) in zip((space.acids,space.diamines),d):
            assert list(species)==sorted(species),d
            assert side.canonical(species[::-1],units[::-1])==(species,units)
    a=space.acids
    assert parse_design_side(a,'PMDA=0.3;s-BPDA=0.7')==parse_design_side(a,'s-BPDA=0.7;PMDA=0.3')
    fixed=sides(fix_acid=['s-BPDA'])
    d=fixed.acids.repair([fixed.acids.resolve('s-BPDA'),fixed.acids.resolve('PMDA')],[1,1])
    assert d[0][0]==fixed.acids.resolve('s-BPDA');print('compositions are canonical')
    # The same repeat unit cut at two places gets one key.
    a1='*c1ccc(-c2ccc(N3C(=O)c4ccc(C(c5ccc6c(c5)C(=O)N(*)C6=O)(C(F)(F)F)C(F)(F)F)cc4C3=O)cc2C(F)(F)F)c(C(F)(F)F)c1'
    a2='*c1ccc2c(c1)C(=O)N(c1ccc(-c3ccc(N4C(=O)c5ccc(C(*)(C(F)(F)F)C(F)(F)F)cc5C4=O)cc3C(F)(F)F)c(C(F)(F)F)c1)C2=O'
    assert unit_key(a1)==unit_key(a2) and unit_key(a1)!=unit_key('*c1ccc(-n2c(=O)c3cc4c(=O)n(*)c(=O)c4cc3c2=O)cc1');print('repeat-unit keys ignore the cut')
    # Composition gap: a measured film plus 5 % of something else is 0.05 away.
    from mobo_copolyimide import Known
    k=Known.__new__(Known);k.rows=[];k.X=np.zeros((0,2048),np.float32)
    pp,bp,new='*c1ccc(-n2c(=O)c3cc4c(=O)n(*)c(=O)c4cc3c2=O)cc1','*c1ccc(N2C(=O)c3ccc(-c4ccc5c(c4)C(=O)N(*)C5=O)cc3C2=O)cc1','*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1'
    k.extend([dict(units=[pp,bp],fractions=[.5,.5],sequence='random',source='training',values={'Tg':450.})])
    kind,gap,_=k.novelty([pp,bp,new],[.475,.475,.05],.1);assert kind=='measured' and abs(gap-.05)<1e-9,(kind,gap)
    kind,gap,_=k.novelty([pp,bp],[.8,.2],.1);assert kind=='known_system' and abs(gap-.3)<1e-9
    assert k.novelty([pp,new],[.5,.5],.1)[0]=='new_units' and k.novelty([new],[1.],.1)[1]==1.
    print('composition gap ok')
    front=np.array([[2.,2.],[3.,0.]])
    assert dominated_by(front,np.array([[1.,1.],[2.,2.],[2.5,2.5],[3.,0.]])).tolist()==[True,False,False,False]
    values=np.array([[1.,1.],[2.,2.],[3.,0.],[0.,0.]])
    assert dominance_ranks(values,np.zeros(4)).tolist()==[1,0,0,2]
    assert dominance_ranks(values,np.array([0,1.,0,0]))[1]==2
    tg=mixing_rule('Tg_film',np.array([100.,300.]),np.array([.5,.5]),np.array([1.,1.]))
    assert abs(tg-(1/(.5/373.15+.5/573.15)-273.15))<1e-9
    print('all passed')

if __name__=='__main__':main()
