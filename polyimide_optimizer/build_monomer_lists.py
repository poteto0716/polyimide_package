#!/usr/bin/env python3
"""Write the editable candidate lists, with common abbreviations, from the TCI monomer tables.

Abbreviations come from common polyimide usage, not from TCI. Monomers without an established
abbreviation keep their full name. Set use=False in the output to drop a monomer from the search.
 .venv/bin/python polyimide_optimizer/build_monomer_lists.py
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
TCI=ROOT.parent/'tci_polyimide/output'
# TCI name prefix -> abbreviation, other abbreviations separated by ';'
ACIDS={
 'Pyromellitic dianhydride':('PMDA',''),
 "4,4'-Carbonyldiphthalic anhydride":('BTDA',''),
 "5,5'-Bi-2-benzofuran-1,1',3,3'-tetrone":('s-BPDA','BPDA'),
 "[4,5'-Biisobenzofuran]-1,1',3,3'-tetrone":('a-BPDA',''),
 "4,4'-Oxydiphthalic anhydride":('ODPA','4,4\'-ODPA'),
 "3,4'-Oxydiphthalic Anhydride":("3,4'-ODPA",''),
 "4,4'-(Hexafluoroisopropylidene)diphthalic anhydride":('6FDA',''),
 "3,3',4,4'-Diphenyl sulfone tetracarboxylic dianhydride":('DSDA',''),
 '2,2-Bis(4-(3,4-dicarboxyphenoxy)phenyl)propane dianhydride':('BPADA',''),
 '9,9-Bis(3,4-dicarboxyphenyl)fluorene Dianhydride':('BPAF',''),
 'Naphthalenetetracarboxylic dianhydride':('NTDA','1,4,5,8-NTDA'),
 '2,3,6,7-Naphthalenetetracarboxylic 2,3:6,7-Dianhydride':('2,3,6,7-NTDA',''),
 "Naphtho(1,2-c:5,6-c')difuran-1,3,6,8-tetraone":('1,2,5,6-NTDA',''),
 '3,4,9,10-Perylenetetracarboxylic dianhydride':('PTCDA',''),
 "4,4'-(1,4-Phenylenedioxy)bisphthalic Anhydride":('HQDPA',''),
 '1,4-Phenylene bis(1,3-dioxo-1,3-dihydroisobenzofuran-5-carbo':('TAHQ',''),
 'Ethylene bis[1,3-dihydro-1,3-dioxoisobenzofuran-5-carboxylat':('TMEG',''),
 "4,4'-(Ethyne-1,2-diyl)diphthalic Anhydride":('EBPA',''),
 '1,2,3,4-Cyclobutanetetracarboxylic dianhydride':('CBDA',''),
 '1,2,3,4-Tetramethyl-1,2,3,4-cyclobutanetetracarboxylic Dianh':('TMCBDA',''),
 '1,2,4,5-Cyclohexanetetracarboxylic dianhydride':('HPMDA','CHDA-DA'),
 'Dicyclohexyl-3,4,3\',4\'-tetracarboxylic Dianhydride':('HBPDA','H-BPDA'),
 'CpODA':('CpODA',''),
 'Bicyclo[2.2.2]oct-7-ene-2,3:5,6-tetracarboxylic dianhydride':('BCDA','BTA'),
 'hexahydro-4,8-ethano-1H,3H-benzo(1,2-c:4,5-c\')difuran-1,3,5,':('BODA','HBTA'),
 "[3,3'-Bifuran]-2,2',5,5'-tetrone, tetrahydro-":('BDA',''),
 'meso-Butane-1,2,3,4-tetracarboxylic Dianhydride':('meso-BDA',''),
 "1H-Cyclopenta[1,2-c:3,4-c']difuran-1,3,4,6(3aH)-tetrone, tet":('CPDA',''),
 'Tetrahydro-1H-5,9-methanopyrano(3,4-d)oxepine-1,3,6,8(4H)-te':('TCA-AH',''),
 '5-(2,5-Dioxotetrahydrofuryl)-3-methyl-3-cyclohexene-1,2-dica':('DOCDA','MCTC'),
 'Naphtho[1,2-c]furan-1,3-dione, 3a,4,5,9b-tetrahydro-5-(tetra':('TDA',''),
 'Ethylenediaminetetraacetic dianhydride':('EDTAD',''),
}
DIAMINES={
 "4,4'-Diaminodiphenyl Ether":('ODA',"4,4'-ODA"),
 '3-(4-Aminophenoxy)aniline':("3,4'-ODA",''),
 "3,3'-Oxydianiline":("3,3'-ODA",''),
 'P-Phenylenediamine':('p-PDA','PDA;PPD'),
 'M-Phenylenediamine':('m-PDA','MPD'),
 '4,4-Diaminodiphenylmethane':('MDA',"4,4'-MDA"),
 "Benzenamine, 3,3'-methylenebis-":("3,3'-MDA",''),
 "(1,1'-Biphenyl)-4,4'-diamine, 2,2'-bis(trifluoromethyl)-":('TFMB','TFDB;PFMB'),
 "4,4'-Oxybis[3-(trifluoromethyl)aniline]":('6FODA',''),
 'Dapsone':("4,4'-DDS",'DDS'),
 "3,3'-Sulphonyldianiline":("3,3'-DDS",''),
 "4,4'-(Isopropylidenebis(4,1-phenyleneoxy))dianiline":('BAPP',''),
 "Benzenamine, 4,4'-[[2,2,2-trifluoro-1-(trifluoromethyl)ethyl":('HFBAPP','BDAF'),
 "4,4'-(Hexafluoroisopropylidene)dianiline":('6FDAm','BAFA;Bis-AF'),
 '2,2-Bis(3-amino-4-hydroxyphenyl)hexafluoropropane':('6FAP','BAHF;Bis-APAF'),
 '1,4-Bis(4-aminophenoxy)benzene':('TPE-Q','APB-144'),
 "Benzenamine, 4,4'-[1,3-phenylenebis(oxy)]bis-":('TPE-R','APB-134'),
 "Benzenamine, 3,3'-(1,3-phenylenebis(oxy))bis-":('APB-133','m-APB'),
 "4,4'-(1,1'-Biphenyl-4,4'-diyldioxy)dianiline":('BAPB',''),
 "4,4'-(Sulfonylbis(4,1-phenyleneoxy))bis(benzenamine)":('BAPS',''),
 "Benzenamine, 3,3'-[sulfonylbis(4,1-phenyleneoxy)]bis-":('m-BAPS',''),
 "4,4'-(9-Fluorenylidene)dianiline":('BAFL','FDA'),
 '9,9-Bis(4-amino-3-fluorophenyl)fluorene':('FFDA',''),
 '1,4-Bis(4-amino-2-trifluoromethylphenoxy)benzene':('p-6FAPB',''),
 "4,4'-[1,3-Phenylenebis(oxy)]bis[3-(trifluoromethyl)aniline]":('m-6FAPB',''),
 '2,2-Bis(3-amino-4-hydroxylphenyl)propane':('BAP',''),
 "4,4'-Sulphonylbis(2-aminophenol)":('BAS',''),
 "3,3'-Dihydroxybenzidine":('HAB',''),
 "4,4'-Diaminobenzanilide":('DABA',''),
 '4-Aminophenyl 4-Aminobenzoate':('APAB',''),
 '(4-(4-Aminobenzoyl)oxyphenyl) 4-aminobenzoate':('ABHQ',''),
 'Bis(4-aminophenyl) Terephthalate':('BPTP',''),
 '1H-Benzimidazol-6-amine, 2-(4-aminophenyl)-':('DAPBI',''),
 '2-(4-Aminophenyl)benzoxazol-5-amine':('DAPBO',''),
 'o-Tolidine':('o-TB',''),
 "2,2'-Dimethylbenzidine":('m-TB',''),
 "3,3',5,5'-Tetramethylbenzidine":('TMBZ',''),
 "4,4'-Diamino-3,3'-dimethyldiphenylmethane":('DMMDA',''),
 "4,4'-Methylenebis(2,6-xylidine)":('TMMDA',''),
 "4,4'-Methylenebis[2,6-diethylaniline]":('MDEA','M-DEA'),
 "4,4'-Methylenebis(2-ethyl-6-methylbenzenamine)":('MMEA','M-MEA'),
 "3,3'-Dichloro-4,4'-Diaminodiphenylmethane":('MOCA',''),
 "4,4'-Diaminobenzophenone":("4,4'-DABP",''),
 "3,3'-Diaminobenzophenone":("3,3'-DABP",''),
 "4,4'-Thiodianiline":('ASD',"4,4'-TDA"),
 '2,4-Diaminotoluene':('2,4-DAT',''),
 '2,6-Diaminotoluene':('2,6-DAT',''),
 '2,5-Diaminotoluene':('2,5-DAT',''),
 '2,4-Diaminomesitylene':('DAM',''),
 '2,3,5,6-Tetramethyl-1,4-phenylenediamine':('4MPD','DAD'),
 '1,4-Benzenediamine, 2,3,5,6-tetrafluoro-':('4FPPD',''),
 "Octafluoro-4,4'-biphenylenediamine":('OFB',''),
 '1,5-Diaminonaphthalene':('1,5-DAN',''),
 'Naphthalene-2,6-diamine':('2,6-DAN',''),
 '1,4-Diaminocyclohexane':('CHDA','DACH'),
 '1,3-Bis(aminomethyl)cyclohexane':('1,3-BAC',''),
 '1,4-Cyclohexanedimethanamine':('1,4-BAC',''),
 'Isophorone diamine':('IPDA',''),
 "4,4'-Diaminodicyclohexylmethane":('H12MDA','PACM'),
 '1,3-Benzenedimethanamine':('MXDA',''),
 '1,4-Benzenedimethanamine':('PXDA',''),
 'Hexamethylenediamine':('HMDA',''),
 'Ethylenediamine':('EDA',''),
 "3,3'-(1,1,3,3-Tetramethyldisiloxane-1,3-diyl)bispropylamine":('SiDA','APTMDS'),
 "4,4'-(1,4-Phenylenediisopropylidene)bisaniline":('Bisaniline-P',''),
 '1,3-Bis[2-(4-aminophenyl)-2-propyl]benzene':('Bisaniline-M',''),
 "4,4'-Isopropylidenedianiline":('Bisaniline-A',''),
 "4,4'-Cyclohexylidenebis[benzenamine]":('Bisaniline-Z',''),
}
def build(source,mapping,output):
    table=pd.read_csv(TCI/source,encoding='utf-8-sig',keep_default_na=False)
    table=table[table.polyimide_ok.astype(str)=='True']
    labels=[None]*len(table);used=set()
    for position,name in enumerate(table.name):
        hits=[p for p in mapping if name==p] or [p for p in mapping if len(p)>=60 and name.startswith(p)]
        if hits:labels[position]=mapping[hits[0]];used.add(hits[0])
    missing=set(mapping)-used
    if missing:raise SystemExit(f'Names not found in {source}: {sorted(missing)}')
    frame=pd.DataFrame({'use':True,'abbreviation':[l[0] if l else '' for l in labels],'other_abbreviations':[l[1] if l else '' for l in labels],
        'name':table.name.values,'tci_product_numbers':table.tci_product_numbers.values,'cas_rn':table.cas_rn.values,'smiles':table.smiles.values})
    named=frame.abbreviation.ne('')
    if frame.abbreviation[named].duplicated().any():raise SystemExit(f'Duplicate abbreviations in {output}')
    frame=pd.concat([frame[named].sort_values('abbreviation',key=lambda s:s.str.casefold()),frame[~named].sort_values('name',key=lambda s:s.str.casefold())])
    frame.to_csv(ROOT/output,index=False,encoding='utf-8-sig')
    print(output,len(frame),'monomers,',int(named.sum()),'with abbreviations')
if __name__=='__main__':
    build('tci_dianhydrides.csv',ACIDS,'acid_candidates.csv')
    build('tci_diamines.csv',DIAMINES,'diamine_candidates.csv')
