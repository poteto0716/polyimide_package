#!/usr/bin/env python3
"""Compare the Bayesian-optimization designs with the training data, as a local web page.

Reads output/pareto_predicted.csv and output/next_experiments.csv from mobo_copolyimide.py, and
polyimide_models/datasets.json (decrypted training data). Writes output/compare/index.html: one large scatter (Tg/modulus/CTE pairs or the structure map) with
zoom and pan, a detail panel and a sortable table, laid out like polyimide_virtual_app. It shows
 - predicted Tg, modulus and CTE of the designs over measured training pairs,
 - a structure map (t-SNE on Tanimoto distance of composition-weighted Morgan counts, the same
   fingerprint the optimizer's GP uses),
 - for each design, its monomers and repeat units, and the nearest training polymers with their measured
   values, monomers and structures. Monomers of measured polymers come from splitting the repeat units
   at the imide bonds (about 9 in 10 units split; amide-imides and the like do not).

The page holds measured values, so it is written under output/, which is not committed.

 .venv/bin/python polyimide_optimizer/compare_with_training.py
 .venv/bin/python polyimide_optimizer/compare_with_training.py --run-dir other_output --neighbors 8
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.manifold import TSNE
RDLogger.DisableLog('rdApp.*')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from mobo_copolyimide import MODELS,TARGETS,composition_fingerprint,tanimoto

PROPS=list(TARGETS)  # Tg, modulus, CTE

# ---------- monomers ----------
ANHYDRIDE=Chem.MolFromSmarts('[#6](=O)[#8][#6](=O)');AMINE=Chem.MolFromSmarts('[NX3;H2]')
def split_unit(smiles):
    """Repeat unit -> (dianhydrides, diamines) as canonical SMILES lists, or None.
    The two * ends are joined first (the cut may sit anywhere in the unit), then every imide ring is
    opened: N-C(=O) bonds are removed, the two carbonyls get an O bridge and the N becomes NH2."""
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:return None
    stars=[a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
    if len(stars)!=2 or any(s.GetDegree()!=1 for s in stars):return None
    a,b=(s.GetNeighbors()[0].GetIdx() for s in stars)
    if a==b or mol.GetBondBetweenAtoms(a,b):return None
    rw=Chem.RWMol(mol);rw.AddBond(a,b,Chem.BondType.SINGLE)
    for s in sorted((s.GetIdx() for s in stars),reverse=True):rw.RemoveAtom(s)
    m=rw.GetMol()
    try:Chem.SanitizeMol(m);Chem.Kekulize(m,clearAromaticFlags=True)
    except Exception:return None
    carbonyl=lambda x:x.GetSymbol()=='C' and any(bd.GetBondType()==Chem.BondType.DOUBLE and bd.GetOtherAtom(x).GetSymbol()=='O' for bd in x.GetBonds())
    imides=[]
    for x in m.GetAtoms():
        if x.GetSymbol()!='N' or not x.IsInRing():continue
        cs=[n.GetIdx() for n in x.GetNeighbors() if carbonyl(n) and m.GetBondBetweenAtoms(x.GetIdx(),n.GetIdx()).IsInRing()]
        if len(cs)==2:imides.append((x.GetIdx(),cs))
    if not imides:return None
    rw=Chem.RWMol(m)
    for n,(c1,c2) in imides:
        rw.RemoveBond(n,c1);rw.RemoveBond(n,c2)
        o=rw.AddAtom(Chem.Atom(8));rw.AddBond(c1,o,Chem.BondType.SINGLE);rw.AddBond(c2,o,Chem.BondType.SINGLE)
        rw.GetAtomWithIdx(n).SetNoImplicit(False);rw.GetAtomWithIdx(n).SetNumExplicitHs(0)
    m=rw.GetMol()
    try:Chem.SanitizeMol(m)
    except Exception:return None
    acids,diamines=[],[]
    for f in Chem.GetMolFrags(m,asMols=True):
        s=Chem.MolToSmiles(f)
        if f.HasSubstructMatch(ANHYDRIDE):acids.append(s)
        elif f.HasSubstructMatch(AMINE):diamines.append(s)
        else:return None
    if not acids or not diamines:return None
    return acids,diamines

def canon(smiles):
    m=Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else smiles

def monomer_names():
    """Canonical SMILES -> (abbreviation, name) from the candidate lists and soft_monomers.csv."""
    out={}
    for file in ['acid_candidates.csv','diamine_candidates.csv']:
        for r in pd.read_csv(HERE/file,encoding='utf-8-sig',keep_default_na=False).to_dict('records'):
            if r.get('smiles'):out.setdefault(canon(r['smiles']),(r['abbreviation'],r['name']))
    try:
        import soft_segment
        for r in soft_segment.read_soft_monomers():out.setdefault(canon(r['smiles']),(r['abbreviation'],r['name']))
    except Exception:pass
    return out

def monomers_from_units(units,fractions,names,cache={}):
    """Monomers of a measured polymer from its repeat units; None if a unit cannot be split."""
    sides={'acid':{},'diamine':{}}
    for u,f in zip(units,fractions):
        if u not in cache:cache[u]=split_unit(u)
        parts=cache[u]
        if parts is None:return None
        for side,mons in zip(('acid','diamine'),parts):
            for m in mons:sides[side][m]=sides[side].get(m,0.)+float(f)/len(mons)
    out=[]
    for side,mons in sides.items():
        total=sum(mons.values())
        for smi,f in sorted(mons.items(),key=lambda kv:-kv[1]):
            abbr,name=names.get(smi,('',''))
            out.append(dict(side=side,abbr=abbr,name=name,smiles=smi,fraction=round(f/total,3)))
    return out

def monomers_from_row(r):
    """Monomers of a design, in the order of the CSV row."""
    out=[]
    for side,prefix in [('acid','acid'),('diamine','diamine')]:
        abbrs=getattr(r,prefix+'s').split(' / ');fracs=str(getattr(r,prefix+'_fractions')).split(' / ')
        smiles=str(getattr(r,prefix+'_smiles','')).split(' / ');full=str(getattr(r,prefix+'_full_names','')).split(' / ')
        for i,a in enumerate(abbrs):
            out.append(dict(side=side,abbr=a,name=full[i] if len(full)==len(abbrs) else '',smiles=smiles[i] if len(smiles)==len(abbrs) else '',fraction=float(fracs[i])))
    return out

def monomer_label(mons):
    if not mons:return ''
    side=lambda k:' / '.join(m['abbr'] or '?' for m in mons if m['side']==k)
    return f"{side('acid')} + {side('diamine')}"

def canonical_design(row):
    """Same composition regardless of the order the monomers are listed in."""
    side=lambda names,fracs:tuple(sorted(zip(names.split(' / '),map(float,fracs.split(' / ')))))
    return side(row.acids,row.acid_fractions),side(row.diamines,row.diamine_fractions)

def opt_float(v):
    return None if v is None or pd.isna(v) else float(v)

def label(side):
    return ' / '.join(n for n,_ in side)+' = '+' / '.join(f'{f:g}' for _,f in side)

def load_training():
    data=json.loads((MODELS/'datasets.json').read_text())
    rows={}
    for prop,target in TARGETS.items():
        for r in data[target]:
            row=rows.setdefault(r['chemistry_key'],dict(units=r['components'],fractions=r['mole_fractions'],kind=r['polymer_kind'],refs=set(),n={}))
            row[prop]=r['value'];row['n'][prop]=r['measurement_count'];row['refs'].update(str(x) for x in r.get('reference_nos',[]))
    out=[]
    for i,(key,r) in enumerate(rows.items()):
        out.append(dict(id=f't{i}',units=r['units'],fractions=r['fractions'],kind=r['kind'],refs=sorted(r['refs'])[:6],n=r['n'],**{p:r.get(p) for p in PROPS}))
    return out

def load_designs(run_dir):
    # mobo_copolyimide.py writes pareto/next; region_search.py writes region_candidates/region_batch.
    region=(run_dir/'region_batch.csv').exists()
    pareto=pd.read_csv(run_dir/('region_candidates.csv' if region else 'pareto_predicted.csv'),encoding='utf-8-sig')
    nxt=pd.read_csv(run_dir/('region_batch.csv' if region else 'next_experiments.csv'),encoding='utf-8-sig')
    designs={}
    for frame,source in [(pareto,'pareto'),(nxt,'next')]:
        for r in frame.itertuples(index=False):
            key=canonical_design(r)
            d=designs.get(key)
            if d is None:
                d=designs[key]=dict(acids=label(key[0]),diamines=label(key[1]),pareto=False,next_ranks=[],monomers=monomers_from_row(r),
                    units=json.loads(r.repeat_units_json),fractions=json.loads(r.repeat_unit_fractions_json),
                    extrapolated=bool(r.composition_extrapolated),min_unit_similarity=opt_float(getattr(r,'min_similarity_to_training',None)),
                    p_region=opt_float(getattr(r,'p_region',None)),soft_w=opt_float(getattr(r,'soft_weight_fraction',None)),
                    hard={p:opt_float(getattr(r,f'hard_part_{p}_pred',None)) for p in PROPS},similarity_to_training=opt_float(getattr(r,'similarity_to_training',None)),
                    novelty=getattr(r,'novelty',None),novelty_ja=getattr(r,'novelty_ja',None),
                    gap=opt_float(getattr(r,'composition_gap_to_measured',None)),p_beyond=opt_float(getattr(r,'p_beyond_measured_front',None)),
                    beaten=bool(getattr(r,'dominated_by_measured_front',False)),
                    **{f'{p}{s}':float(getattr(r,f'{p}_{c}')) for p in PROPS for s,c in [('','pred'),('_lo','lo90'),('_hi','hi90')]})
            if source=='pareto':d['pareto']=True
            else:
                d['next_ranks'].append(int(r.batch_rank))
                if hasattr(r,'ehvi'):d['ehvi']=float(r.ehvi)
                if hasattr(r,'p_at_least_one_so_far'):d['p_cumulative']=float(r.p_at_least_one_so_far)
    out=list(designs.values())
    for i,d in enumerate(out):d['id']=f'd{i}'
    duplicates=len(nxt)-sum(1 for d in out if d['next_ranks'])
    return out,duplicates,len(pareto),len(nxt)

def structure_svg(smiles,folder,cache={}):
    """Write one SVG per SMILES, named by its hash; return the file name."""
    if smiles in cache:return cache[smiles]
    name=hashlib.sha1(smiles.encode()).hexdigest()[:16]+'.svg'
    path=folder/name
    if not path.exists():
        mol=Chem.MolFromSmiles(smiles)
        drawer=rdMolDraw2D.MolDraw2DSVG(320,200)
        opts=drawer.drawOptions();opts.clearBackground=False;opts.padding=0.06;opts.bondLineWidth=1.5
        if mol is not None:
            rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol)
        drawer.FinishDrawing();path.write_text(drawer.GetDrawingText())
    cache[smiles]=name
    return name

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir',type=Path,default=HERE/'output',help='Folder with pareto_predicted.csv and next_experiments.csv.')
    ap.add_argument('--out',type=Path,default=None,help='Output folder (default: RUN_DIR/compare).')
    ap.add_argument('--neighbors',type=int,default=5,help='Nearest training polymers listed per design.')
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    out=args.out or args.run_dir/'compare'
    (out/'structures').mkdir(parents=True,exist_ok=True)

    training=load_training()
    # Polymers measured in-house and passed with --observations (region_search.py writes them out).
    obs_file=args.run_dir/'observations_resolved.csv'
    if obs_file.exists():
        for i,r in enumerate(pd.read_csv(obs_file,encoding='utf-8-sig').to_dict('records')):
            training.append(dict(id=f'o{i}',units=json.loads(r['units_json']),fractions=json.loads(r['fractions_json']),kind='observation',
                                 refs=[],n={},label=f"{r['acids']} + {r['diamines']}",**{p:opt_float(r.get(p)) for p in PROPS}))
    # Measured polymers on the Pareto front that the optimizer had to beat (measured_front.csv).
    front_file=args.run_dir/'measured_front.csv'
    if front_file.exists():
        wanted={(tuple(json.loads(u)),tuple(round(f,3) for f in json.loads(f))) for u,f in pd.read_csv(front_file,encoding='utf-8-sig')[['units_json','fractions_json']].itertuples(index=False)}
        for r in training:
            total=sum(r['fractions']);r['front']=(tuple(r['units']),tuple(round(f/total,3) for f in r['fractions'])) in wanted
    designs,duplicates,n_pareto,n_next=load_designs(args.run_dir)
    names=monomer_names()
    for r in training:
        r['monomers']=monomers_from_units(r['units'],r['fractions'],names)
        if not r.get('label') and r['monomers'] and all(m['abbr'] for m in r['monomers']):r['label']=monomer_label(r['monomers'])
    Xt=np.array([composition_fingerprint(r['units'],r['fractions']) for r in training])
    Xd=np.array([composition_fingerprint(d['units'],d['fractions']) for d in designs])

    # Nearest training polymers, by the same similarity the GP kernel uses.
    S=tanimoto(Xd,Xt)
    for d,s in zip(designs,S):
        order=np.argsort(-s)[:args.neighbors]
        d['neighbors']=[[training[j]['id'],round(float(s[j]),3)] for j in order]
        d['nearest']=round(float(s[order[0]]),3)
    # How close each training polymer is to any design, for the map's hover.
    for r,s in zip(training,S.max(0)):r['nearest_design']=round(float(s),3)

    # Structure map.
    X=np.vstack([Xt,Xd])
    D=np.clip(1-tanimoto(X,X),0,None);np.fill_diagonal(D,0)
    xy=TSNE(n_components=2,metric='precomputed',init='random',perplexity=30,random_state=args.seed).fit_transform(D)
    for r,(x,y) in zip(training+designs,xy):r['x']=round(float(x),3);r['y']=round(float(y),3)

    for r in training+designs:r['svgs']=[structure_svg(s,out/'structures') for s in r['units']]
    for r in training+designs:
        for m in r.get('monomers') or []:
            if m['smiles']:m['svg']=structure_svg(m['smiles'],out/'structures')

    summary=json.loads((args.run_dir/'run_summary.json').read_text()) if (args.run_dir/'run_summary.json').exists() else {}
    meta=dict(n_observations=sum(r['kind']=='observation' for r in training),n_front=sum(bool(r.get('front')) for r in training),n_training=sum(r['kind']!='observation' for r in training),n_measured={p:sum(r[p] is not None for r in training if r['kind']!='observation') for p in PROPS},
              n_pareto=n_pareto,n_next=n_next,n_designs=len(designs),next_duplicates=duplicates,
              constraint=summary.get('arguments',{}).get('constraint',''),evaluated=summary.get('evaluated'),
              region=summary.get('arguments',{}).get('region',''),max_similarity=summary.get('arguments',{}).get('max_similarity'),
              p_at_least_one=summary.get('p_at_least_one'))
    (out/'data.js').write_text('window.DATA='+json.dumps(dict(meta=meta,training=training,designs=designs),ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
    (out/'plotly.min.js').unlink(missing_ok=True)  # left over from the earlier Plotly version of the page
    for name in ['index.html','app.js','style.css']:
        shutil.copyfile(HERE/'compare_web'/name,out/name)
    print(json.dumps(meta,ensure_ascii=False),'->',out/'index.html')

if __name__=='__main__':main()
