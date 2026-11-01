#!/usr/bin/env python3
"""Build polyimide repeat-unit p-SMILES from a dianhydride and a diamine.

The repeat unit is *-N(imide)...(imide)N-R-*, matching the '*' convention used in
polyimide_tg_modulus/repeat_units_used.tsv, and always has the mass of
dianhydride + diamine - 2 H2O (checked for every product).

Single pair:
 .venv/bin/python tci_polyimide/polyimide_psmiles.py --dianhydride 'SMILES' --diamine 'SMILES'
All combinations from CSV/TSV files holding a 'smiles' column:
 .venv/bin/python tci_polyimide/polyimide_psmiles.py --dianhydrides a.csv --diamines b.csv --output pi.csv

When both monomers are unsymmetric, two head-to-tail regioisomers exist: psmiles holds one
and alt_psmiles the other. Rows with polyimide_ok=False in an input file are skipped
unless --include-flagged is given.
"""
import argparse,csv,sys
from pathlib import Path
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

ANHYDRIDE=Chem.MolFromSmarts('[#6;R](=O)[#8;R][#6;R](=O)')
PRIMARY_AMINE=Chem.MolFromSmarts('[NX3;H2;+0;!$(N-[#6]=[O,S,N]);!$(N-[N,O,S,P]);$(N-[#6])]')
H2O_MW=Descriptors.MolWt(Chem.MolFromSmiles('O'))
ANHYDRIDE_RING_SIZES=(5,6)

class MonomerError(ValueError):pass

def parent_mol(smiles):
    """Return the largest fragment, without counterions, and the number of fragments."""
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise MonomerError(f'Invalid SMILES: {smiles}')
    frags=Chem.GetMolFrags(mol,asMols=True)
    return max(frags,key=lambda m:m.GetNumHeavyAtoms()),len(frags)

def find_anhydride_oxygens(mol):
    """Ring oxygen index of each cyclic anhydride that can close to a 5/6-membered imide."""
    oxygens=sorted({m[2] for m in mol.GetSubstructMatches(ANHYDRIDE,uniquify=False)})
    ring_info=mol.GetRingInfo()
    return [o for o in oxygens if ring_info.MinAtomRingSize(o) in ANHYDRIDE_RING_SIZES]

def count_all_anhydrides(mol):
    return len({m[2] for m in mol.GetSubstructMatches(ANHYDRIDE,uniquify=False)})

def find_primary_amines(mol):
    return sorted(m[0] for m in mol.GetSubstructMatches(PRIMARY_AMINE))

def _is_symmetric(mol,atoms):
    ranks=list(Chem.CanonicalRankAtoms(mol,breakTies=False))
    return ranks[atoms[0]]==ranks[atoms[1]]

def _build(dianhydride,diamine,oxygens,amines):
    """Bond imide N of oxygens[0] to the carbon of amines[0]; cap the other two ends with '*'."""
    offset=dianhydride.GetNumAtoms()
    rw=Chem.RWMol(Chem.CombineMols(dianhydride,diamine))
    Chem.Kekulize(rw,clearAromaticFlags=True)
    for oxygen in oxygens:
        atom=rw.GetAtomWithIdx(oxygen)
        atom.SetAtomicNum(7);atom.SetNoImplicit(False);atom.SetNumExplicitHs(0)
    linked=rw.GetAtomWithIdx(offset+amines[0])
    carbon=linked.GetNeighbors()[0].GetIdx()
    rw.AddBond(oxygens[0],carbon,Chem.BondType.SINGLE)
    star=rw.AddAtom(Chem.Atom(0))
    rw.AddBond(oxygens[1],star,Chem.BondType.SINGLE)
    capped=rw.GetAtomWithIdx(offset+amines[1])
    capped.SetAtomicNum(0);capped.SetNoImplicit(True);capped.SetNumExplicitHs(0)
    rw.RemoveAtom(linked.GetIdx())
    mol=rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

def make_polyimide(dianhydride_smiles,diamine_smiles):
    """Return psmiles, alt_psmiles ('' unless regioisomers exist) and the repeat-unit MW."""
    dianhydride,_=parent_mol(dianhydride_smiles)
    diamine,_=parent_mol(diamine_smiles)
    oxygens=find_anhydride_oxygens(dianhydride)
    amines=find_primary_amines(diamine)
    if count_all_anhydrides(dianhydride)!=2 or len(oxygens)!=2:
        raise MonomerError(f'Expected 2 cyclic 5/6-membered anhydrides: {dianhydride_smiles}')
    if len(amines)!=2:raise MonomerError(f'Expected 2 primary amines: {diamine_smiles}')
    a,b=oxygens;x,y=amines
    # (a,x)/(b,y) are the same chain with another break point, as are (a,y)/(b,x).
    first=min(_build(dianhydride,diamine,[a,b],[x,y]),_build(dianhydride,diamine,[b,a],[y,x]))
    second=min(_build(dianhydride,diamine,[a,b],[y,x]),_build(dianhydride,diamine,[b,a],[x,y]))
    psmiles,other=min(first,second),max(first,second)
    both_unsymmetric=not _is_symmetric(dianhydride,oxygens) and not _is_symmetric(diamine,amines)
    product=Chem.MolFromSmiles(psmiles)
    weight=Descriptors.MolWt(product)
    expected=Descriptors.MolWt(dianhydride)+Descriptors.MolWt(diamine)-2*H2O_MW
    stars=sum(atom.GetAtomicNum()==0 for atom in product.GetAtoms())
    if stars!=2 or abs(weight-expected)>0.01:
        raise MonomerError(f'Mass balance failed: {dianhydride_smiles} + {diamine_smiles}')
    return psmiles,other if both_unsymmetric else '',weight

def read_monomers(path,include_flagged):
    delimiter='\t' if path.suffix.lower() in('.tsv','.txt') else ','
    with path.open(encoding='utf-8-sig',newline='') as handle:
        rows=list(csv.DictReader(handle,delimiter=delimiter))
    if rows and 'smiles' not in rows[0]:raise ValueError(f"{path} needs a 'smiles' column.")
    return [r for r in rows if include_flagged or r.get('polyimide_ok','True')=='True']

def label(row):
    return row.get('tci_product_numbers') or row.get('name') or row.get('id') or row['smiles']

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dianhydride');ap.add_argument('--diamine')
    ap.add_argument('--dianhydrides',type=Path);ap.add_argument('--diamines',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--include-flagged',action='store_true')
    args=ap.parse_args()
    if args.dianhydride and args.diamine:
        psmiles,alt,_=make_polyimide(args.dianhydride,args.diamine)
        print(psmiles)
        if alt:print(alt)
        return
    if not(args.dianhydrides and args.diamines and args.output):
        ap.error('Give --dianhydride/--diamine, or --dianhydrides/--diamines/--output.')
    dianhydrides=read_monomers(args.dianhydrides,args.include_flagged)
    diamines=read_monomers(args.diamines,args.include_flagged)
    written=failed=0
    with args.output.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['dianhydride_id','dianhydride_name','dianhydride_smiles','diamine_id','diamine_name',
                         'diamine_smiles','psmiles','alt_psmiles','repeat_unit_mw'])
        for d in dianhydrides:
            for m in diamines:
                try:psmiles,alt,weight=make_polyimide(d['smiles'],m['smiles'])
                except(MonomerError,ValueError,RuntimeError) as error:
                    failed+=1;print(f'SKIP {label(d)} + {label(m)}: {error}',file=sys.stderr);continue
                writer.writerow([label(d),d.get('name',''),d['smiles'],label(m),m.get('name',''),m['smiles'],
                                 psmiles,alt,f'{weight:.3f}'])
                written+=1
    print(f'{len(dianhydrides)} dianhydrides x {len(diamines)} diamines: {written} written, {failed} failed -> {args.output}')

if __name__=='__main__':main()
