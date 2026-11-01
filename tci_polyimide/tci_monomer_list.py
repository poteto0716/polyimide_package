#!/usr/bin/env python3
"""List dianhydrides and diamines from the substances TCI has deposited in PubChem.

tcichemicals.com refuses automated access, so this uses the public PubChem PUG REST API
(source 'TCI (Tokyo Chemical Industry)'). A row means TCI registered the product in PubChem,
NOT that it is on sale today: check product_url before relying on availability.

 .venv/bin/python tci_polyimide/tci_monomer_list.py --outdir tci_polyimide/output

Products TCI did not deposit can be listed in supplemental_products.csv (product number and
CAS RN); their structures are looked up by CAS from any PubChem source and marked as supplemented.

Responses are cached in <outdir>/cache, so an interrupted run resumes; use a new --outdir
or delete the cache to refresh.
"""
import argparse,csv,json,sys,time,urllib.error,urllib.parse,urllib.request
from collections import Counter
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
from polyimide_psmiles import count_all_anhydrides,find_anhydride_oxygens,find_primary_amines,parent_mol

BASE='https://pubchem.ncbi.nlm.nih.gov/rest/pug'
SOURCE='TCI (Tokyo Chemical Industry)'
BATCH=100
PAUSE_S=0.25  # PubChem asks for at most 5 requests per second
PROPERTIES='Title,IUPACName,SMILES'
SUPPLEMENT_DATE='supplemented'
# Groups that compete with imide formation exclude a monomer from polyimide_ok.
BLOCKING_GROUPS={'carboxylic_acid':'[CX3](=O)[OX2H1]','sulfonic_acid':'[SX4](=O)(=O)[OX2H1]','thiol':'[SX2H]','aldehyde':'[CX3H1]=O',
    'acyl_halide':'[CX3](=O)[F,Cl,Br,I]','isocyanate':'N=C=O','epoxide':'[OX2;r3]'}
NOTED_GROUPS={'heteroaryl_amine':'[NX3;H2]-c:n','phenol':'[OX2H]c','alcohol':'[OX2H][CX4]','secondary_amine':'[NX3;H1;+0;!$(N-[#6]=[O,S,N])]([#6])[#6]',
    'tertiary_amine':'[NX3;H0;+0;!$(N-[#6]=[O,S,N])]([#6])([#6])[#6]','hydrazine':'[NX3]-[NX3]'}
BLOCKING={k:Chem.MolFromSmarts(v) for k,v in BLOCKING_GROUPS.items()}
NOTED={k:Chem.MolFromSmarts(v) for k,v in NOTED_GROUPS.items()}
ANHYDRIDE_IN_DIAMINE='anhydride'
COLUMNS=['tci_product_numbers','name','cas_rn','smiles','parent_smiles','form','polyimide_ok','flags','type_detail',
         'formula','mw','cid','sids','pubchem_deposit_date','product_url','iupac_name','tci_synonyms']

def request(url,data=None,attempts=6):
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,data=data),timeout=120) as response:
                body=response.read()
            time.sleep(PAUSE_S)
            return body
        except urllib.error.HTTPError as error:
            if error.code==404:return None
            if error.code not in(429,500,502,503,504) or attempt==attempts-1:raise
        except(urllib.error.URLError,TimeoutError):
            if attempt==attempts-1:raise
        time.sleep(2**attempt)

def cached_post(cache,name,url,field,ids):
    path=cache/f'{name}.json'
    if path.exists():return json.loads(path.read_text())
    body=request(url,urllib.parse.urlencode({field:','.join(map(str,ids))}).encode())
    data=json.loads(body) if body else {}
    path.write_text(json.dumps(data))
    return data

def chunks(items,size=BATCH):
    return [items[i:i+size] for i in range(0,len(items),size)]

def fetch_substances(cache):
    """Return {sid: product number, deposit date, synonyms, cid} for every TCI substance."""
    path=cache/'sids.txt'
    if not path.exists():
        path.write_text(request(f'{BASE}/substance/sourceall/{urllib.parse.quote(SOURCE)}/sids/TXT').decode())
    sids=[int(s) for s in path.read_text().split()]
    substances={}
    batches=chunks(sids)
    for i,batch in enumerate(batches):
        data=cached_post(cache,f'substance_{i:05d}',f'{BASE}/substance/sid/JSON','sid',batch)
        for record in data.get('PC_Substances',[]):
            db=record['source']['db']
            date=db.get('date',{}).get('std',{})
            substances[record['sid']['id']]={'product':db['source_id']['str'],
                'date':f"{date.get('year',0):04d}-{date.get('month',0):02d}-{date.get('day',0):02d}",
                'synonyms':record.get('synonyms',[]),'cid':None}
        cids=cached_post(cache,f'cids_{i:05d}',f'{BASE}/substance/sid/cids/JSON','sid',batch)
        for info in cids.get('InformationList',{}).get('Information',[]):
            found=info.get('CID',[])
            # More than one CID means the deposit does not resolve to a single structure.
            if len(found)==1 and info['SID'] in substances:substances[info['SID']]['cid']=found[0]
        if (i+1)%50==0 or i+1==len(batches):print(f'substances {i+1}/{len(batches)}',flush=True)
    return substances

def fetch_properties(cache,cids):
    properties={}
    batches=chunks(sorted(cids))
    for i,batch in enumerate(batches):
        data=cached_post(cache,f'property_{i:05d}',f'{BASE}/compound/cid/property/{PROPERTIES}/JSON','cid',batch)
        for row in data.get('PropertyTable',{}).get('Properties',[]):properties[row['CID']]=row
        if (i+1)%50==0 or i+1==len(batches):print(f'properties {i+1}/{len(batches)}',flush=True)
    return properties

def add_supplements(cache,path,substances,properties):
    """Add products from a user list. PubChem must resolve each CAS RN to exactly one compound."""
    if not path.exists():return 0
    with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    for row in rows:
        cas=row['cas_rn'].strip();product=row['tci_product_number'].strip()
        lookup=cache/f'supplement_cid_{cas}.json'
        if not lookup.exists():
            body=request(f'{BASE}/compound/name/{urllib.parse.quote(cas)}/cids/JSON')
            if body is None:raise ValueError(f'{product}: PubChem has no compound for CAS {cas}')
            lookup.write_bytes(body)
        cids=json.loads(lookup.read_text())['IdentifierList']['CID']
        if len(cids)!=1:raise ValueError(f'{product}: CAS {cas} resolves to {cids}')
        cid=cids[0]
        data=cached_post(cache,f'supplement_property_{cid}',f'{BASE}/compound/cid/property/{PROPERTIES}/JSON','cid',[cid])
        properties[cid]=data['PropertyTable']['Properties'][0]
        substances[f'supplement:{product}']={'product':product,'date':SUPPLEMENT_DATE,'synonyms':[cas],'cid':cid}
    return len(rows)

def present_groups(mol,groups):
    return [name for name,query in groups.items() if mol.HasSubstructMatch(query)]

def is_ortho_aromatic(mol,amines):
    """o-Diamines cyclize to benzimidazole-type rings instead of giving a linear polyimide."""
    a,b=[mol.GetAtomWithIdx(i).GetNeighbors()[0] for i in amines]
    return a.GetIsAromatic() and b.GetIsAromatic() and mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx()) is not None

def amine_type(mol,amines):
    aromatic=sum(mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIsAromatic() for i in amines)
    return {2:'aromatic',1:'aromatic-aliphatic',0:'aliphatic'}[aromatic]

def classify(smiles):
    """Return (dianhydride|diamine|None, parent mol, form, ok, flags, type detail)."""
    try:parent,fragments=parent_mol(smiles)
    except ValueError:return None,None,'',False,[],''
    form='free' if fragments==1 else 'salt_or_mixture'
    flags=[] if fragments==1 else ['salt_or_mixture']
    blocking=present_groups(parent,BLOCKING)
    noted=present_groups(parent,NOTED)
    all_anhydrides=count_all_anhydrides(parent)
    amines=find_primary_amines(parent)
    if all_anhydrides==2 and not amines:
        ring_ok=len(find_anhydride_oxygens(parent))==2
        if not ring_ok:flags.append('anhydride_ring_not_5_or_6')
        flags+=blocking+noted
        return 'dianhydride',parent,form,form=='free' and ring_ok and not blocking,flags,'cyclic'
    if len(amines)==2:
        if all_anhydrides:blocking=blocking+[ANHYDRIDE_IN_DIAMINE]
        ortho=is_ortho_aromatic(parent,amines)
        if ortho:flags.append('ortho_aromatic_diamine')
        flags+=blocking+noted
        return 'diamine',parent,form,form=='free' and not ortho and not blocking,flags,amine_type(parent,amines)
    return None,None,'',False,[],''

def is_cas(text):
    parts=text.split('-')
    return len(parts)==3 and all(p.isdigit() for p in parts) and len(parts[1])==2 and len(parts[2])==1

def build_rows(substances,properties):
    """One row per structure; products that share a structure are merged."""
    grouped={}
    for sid,substance in substances.items():
        if substance['cid'] in properties:grouped.setdefault(substance['cid'],[]).append((sid,substance))
    rows={'dianhydride':[],'diamine':[]}
    for cid,members in grouped.items():
        prop=properties[cid]
        smiles=prop.get('SMILES','')
        kind,parent,form,ok,flags,detail=classify(smiles)
        if kind is None:continue
        products=sorted({s['product'] for _,s in members})
        synonyms=sorted({n for _,s in members for n in s['synonyms'] if n not in products})
        rows[kind].append({'tci_product_numbers':';'.join(products),'name':prop.get('Title',''),
            'cas_rn':';'.join(n for n in synonyms if is_cas(n)),'smiles':smiles,'parent_smiles':Chem.MolToSmiles(parent),
            'form':form,'polyimide_ok':ok,'flags':';'.join(flags),'type_detail':detail,
            'formula':rdMolDescriptors.CalcMolFormula(parent),'mw':f'{Descriptors.MolWt(parent):.3f}','cid':cid,
            'sids':';'.join(str(sid) for sid,_ in members),'pubchem_deposit_date':max(s['date'] for _,s in members),
            'product_url':' '.join(f'https://www.tcichemicals.com/JP/ja/p/{p}' for p in products),
            'iupac_name':prop.get('IUPACName',''),'tci_synonyms':';'.join(n for n in synonyms if not is_cas(n))})
    for kind in rows:rows[kind].sort(key=lambda row:row['tci_product_numbers'])
    return rows

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS)
        writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir',type=Path,default=Path(__file__).resolve().parent/'output')
    ap.add_argument('--supplement',type=Path,default=Path(__file__).resolve().parent/'supplemental_products.csv')
    args=ap.parse_args()
    cache=args.outdir/'cache'
    cache.mkdir(parents=True,exist_ok=True)
    substances=fetch_substances(cache)
    cids={s['cid'] for s in substances.values() if s['cid']}
    properties=fetch_properties(cache,cids)
    supplemented=add_supplements(cache,args.supplement,substances,properties)
    rows=build_rows(substances,properties)
    write_csv(args.outdir/'tci_dianhydrides.csv',rows['dianhydride'])
    write_csv(args.outdir/'tci_diamines.csv',rows['diamine'])
    deposited=[s for s in substances.values() if s['date']!=SUPPLEMENT_DATE]
    summary={'source':SOURCE,'retrieved':time.strftime('%Y-%m-%d'),'substances':len(deposited),'supplemented_products':supplemented,
        'substances_without_single_structure':sum(not s['cid'] for s in deposited),
        'deposit_years':dict(sorted(Counter(s['date'][:4] for s in deposited).items())),
        **{f'{kind}s':len(items) for kind,items in rows.items()},
        **{f'{kind}s_polyimide_ok':sum(r['polyimide_ok'] for r in items) for kind,items in rows.items()}}
    (args.outdir/'summary.json').write_text(json.dumps(summary,indent=1,ensure_ascii=False))
    print(json.dumps(summary,indent=1,ensure_ascii=False))

if __name__=='__main__':sys.exit(main())
