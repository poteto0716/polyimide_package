#!/usr/bin/env python3
"""Checks for mobo_copolyimide.py that need no trained model.
 .venv/bin/python polyimide_optimizer/test_mobo.py
"""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mobo_copolyimide import UNITS,Side,Space,dominance_ranks,read_monomers,mixing_rule

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
    values=np.array([[1.,1.],[2.,2.],[3.,0.],[0.,0.]])
    assert dominance_ranks(values,np.zeros(4)).tolist()==[1,0,0,2]
    assert dominance_ranks(values,np.array([0,1.,0,0]))[1]==2
    tg=mixing_rule('Tg_film',np.array([100.,300.]),np.array([.5,.5]),np.array([1.,1.]))
    assert abs(tg-(1/(.5/373.15+.5/573.15)-273.15))<1e-9
    print('all passed')

if __name__=='__main__':main()
