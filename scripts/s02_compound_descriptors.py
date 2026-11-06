"""Step 2 - structure-resolved compound identities and descriptors.

identity = first InChIKey block (connectivity; merges stereo-isomers) when PubChem resolved the name,
otherwise a normalised name string. Functional classes come from RDKit SMARTS (name regex fallback).
Outputs: tables/compound_identity.csv (name -> identity), tables/compound_descriptors.csv (identity -> features)
"""
import os, re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, AllChem

RDLogger.DisableLog('rdApp.*')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')

SMARTS = {
    'f_phenol': '[OX2H][c]',
    'f_methoxyphenol': '[OX2H]c:c[OX2][CH3]',
    'f_carboxylic_acid': '[CX3](=O)[OX2H1]',
    'f_ester': '[#6][CX3](=O)[OX2][#6]',
    'f_aldehyde': '[CX3H1](=O)[#6,#1]',
    'f_ketone': '[#6][CX3](=O)[#6]',
    'f_alcohol': '[OX2H][CX4]',
    'f_ether': '[OD2]([#6])[#6]',
    'f_furan': 'o1cccc1',
    'f_n_aromatic': '[n]',
    'f_amine': '[NX3;!$(N-C=O);!$(N-a)]',
    'f_amide': '[NX3][CX3](=O)',
    'f_nitrile': '[CX2]#[NX1]',
    'f_sulfur': '[#16]',
    'f_halogen': '[F,Cl,Br,I]',
    'f_alkene': '[CX3]=[CX3]',
    'f_anhydrosugar': '[OX2]1[CX4][CX4][CX4][CX4][CX4]1',
}
PATTS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}

CLASSES = ['N-heterocycles', 'Other N-compounds', 'S/halogen compounds', 'Anhydrosugars', 'Carboxylic acids',
           'Esters', 'Methoxyphenols', 'Phenols', 'Furans', 'Aldehydes', 'Ketones', 'Alcohols/ethers',
           'PAHs', 'Monoaromatics', 'Alkenes', 'Alkanes', 'Other oxygenates', 'Unresolved']


def fused_aromatic_rings(mol):
    ri = mol.GetRingInfo()
    arom = [set(r) for r in ri.AtomRings() if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    return len(arom), any(len(a & b) >= 2 for i, a in enumerate(arom) for b in arom[i + 1:])


def classify_mol(mol, f):
    elems = {a.GetSymbol() for a in mol.GetAtoms()}
    n_arom, fused = fused_aromatic_rings(mol)
    if 'N' in elems:
        ring_n = mol.HasSubstructMatch(Chem.MolFromSmarts('[#7;R]'))
        return 'N-heterocycles' if (f['f_n_aromatic'] or ring_n) else 'Other N-compounds'
    if elems & {'S', 'Cl', 'Br', 'F', 'I', 'P', 'Si'}:
        return 'S/halogen compounds'
    if 'O' not in elems:
        if n_arom >= 2 and fused:
            return 'PAHs'
        if n_arom >= 1:
            return 'Monoaromatics'
        if f['f_alkene']:
            return 'Alkenes'
        unsat = any(b.GetBondTypeAsDouble() > 1 for b in mol.GetBonds())
        return 'Alkenes' if unsat else 'Alkanes'
    nO = sum(a.GetSymbol() == 'O' for a in mol.GetAtoms())
    if f['f_anhydrosugar'] and nO >= 3:
        return 'Anhydrosugars'
    if f['f_carboxylic_acid']:
        return 'Carboxylic acids'
    if f['f_methoxyphenol']:
        return 'Methoxyphenols'
    if f['f_phenol']:
        return 'Phenols'
    if f['f_ester']:
        return 'Esters'
    if f['f_furan']:
        return 'Furans'
    if f['f_aldehyde']:
        return 'Aldehydes'
    if f['f_ketone']:
        return 'Ketones'
    if f['f_alcohol'] or f['f_ether']:
        return 'Alcohols/ethers'
    return 'Other oxygenates'


NAME_RULES = [
    ('N-heterocycles', r'pyridin|pyrrol|indol|quinolin|pyrazin|imidazol|pyrimidin|carbazol|pyrazol'),
    ('Other N-compounds', r'amin|amide|nitril|aniline|nitro|azo|urea|cyan'),
    ('S/halogen compounds', r'thio|sulf|chlor|brom|fluor|benzothiazol|mercapt|phosph|sil'),
    ('Anhydrosugars', r'levoglucosan|anhydro|glucopyran|sugar'),
    ('Carboxylic acids', r'acid'),
    ('Methoxyphenols', r'guaiacol|syring|methoxyphenol|methoxy-phenol|eugenol|vanill|creosol'),
    ('Phenols', r'phenol|cresol|catechol|hydroquinon|xylenol|resorcin'),
    ('Esters', r'ester|oate\b|acetate'),
    ('Furans', r'furan|furfural|furfuryl'),
    ('Aldehydes', r'aldehyde|\bal\b|anal\b'),
    ('Ketones', r'one\b|ketone'),
    ('Alcohols/ethers', r'ol\b|ether|alcohol'),
    ('PAHs', r'naphthal|anthrac|phenanthr|pyrene|fluoren|indene|biphenyl|azulene|acenaph|chrysene'),
    ('Monoaromatics', r'benzene|toluene|xylene|styrene|cumene|cymene|mesitylene'),
    ('Alkenes', r'ene\b|diene|limonene|ene,|yne\b'),
    ('Alkanes', r'ane\b|ane,|paraffin'),
]


def classify_name(n):
    s = str(n).lower()
    for cls, pat in NAME_RULES:
        if re.search(pat, s):
            return cls
    return 'Unresolved'


def norm_name(n):
    return re.sub(r'[^a-z0-9]', '', str(n).lower())


def parse_formula(fm):
    fm = str(fm).replace(' ', '')
    out = {}
    for el, cnt in re.findall(r'([A-Z][a-z]?)(\d*)', fm):
        out[el] = out.get(el, 0) + (int(cnt) if cnt else 1)
    return out


def elemental(fm_dict):
    C, H, O, N, S = (fm_dict.get(k, 0) for k in ['C', 'H', 'O', 'N', 'S'])
    X = sum(fm_dict.get(k, 0) for k in ['Cl', 'Br', 'F', 'I'])
    dbe = C - (H + X) / 2 + N / 2 + 1 if C else np.nan
    return {'nC': C, 'nH': H, 'nO': O, 'nN': N, 'nS': S, 'nX': X, 'DBE': dbe,
            'H_C': H / C if C else np.nan, 'O_C': O / C if C else np.nan, 'N_C': N / C if C else np.nan}


def main():
    rec = pd.read_csv(os.path.join(TAB, 'records_clean.csv'))
    pc = pd.read_csv(os.environ.get('PYROLYSIS_PUBCHEM_CACHE', os.path.join(ROOT, 'cache', 'pubchem_names.csv'))).drop_duplicates('name')
    pc['name'] = pc['name'].astype(str).str.strip()
    names = rec[['name_verbatim', 'chemical_formula']].copy()
    names['name_verbatim'] = names['name_verbatim'].astype(str).str.strip()
    form_map = names[names.chemical_formula.notna() & (names.chemical_formula.astype(str).str.match(r'^C\d'))] \
        .groupby('name_verbatim').chemical_formula.agg(lambda s: s.mode().iloc[0])
    ident = pd.DataFrame({'name_verbatim': sorted(names.name_verbatim.unique())})
    ident = ident.merge(pc[['name', 'smiles', 'iso_smiles', 'inchikey', 'formula', 'cid']], left_on='name_verbatim',
                        right_on='name', how='left').drop(columns='name')
    ident['data_formula'] = ident.name_verbatim.map(form_map)
    # guard: PubChem hit must agree with author formula when both exist (catches wrong name matches)
    def agree(r):
        if pd.isna(r.inchikey) or pd.isna(r.data_formula) or pd.isna(r.formula):
            return True
        a, b = parse_formula(r.formula), parse_formula(r.data_formula)
        return a.get('C', 0) == b.get('C', 0) and a.get('O', 0) == b.get('O', 0) and a.get('N', 0) == b.get('N', 0)
    ident['pubchem_formula_agree'] = ident.apply(agree, axis=1)
    resolved = ident.inchikey.notna() & ident.pubchem_formula_agree
    ident['identity'] = np.where(resolved, ident.inchikey.astype(str).str[:14], 'NAME_' + ident.name_verbatim.map(norm_name))
    ident['resolved'] = resolved.astype(int)
    ident.to_csv(os.path.join(TAB, 'compound_identity.csv'), index=False, encoding='utf-8-sig')

    rows = []
    for idt, grp in ident.groupby('identity'):
        r0 = grp.iloc[0]
        d = {'identity': idt, 'example_name': r0.name_verbatim, 'resolved': int(r0.resolved), 'smiles': ''}
        mol = None
        if r0.resolved:
            smi = r0.smiles if isinstance(r0.smiles, str) and r0.smiles else r0.iso_smiles
            mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
            d['smiles'] = smi
        if mol is not None:
            fm = rdMolDescriptors.CalcMolFormula(mol)
            d.update(elemental(parse_formula(fm)))
            flags = {k: int(mol.HasSubstructMatch(p)) for k, p in PATTS.items()}
            d.update(flags)
            d.update({'MW': Descriptors.MolWt(mol), 'logP': Crippen.MolLogP(mol),
                      'TPSA': rdMolDescriptors.CalcTPSA(mol), 'HBD': Lipinski.NumHDonors(mol),
                      'HBA': Lipinski.NumHAcceptors(mol), 'n_arom_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                      'n_rings': rdMolDescriptors.CalcNumRings(mol), 'frac_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
                      'n_rot': rdMolDescriptors.CalcNumRotatableBonds(mol)})
            d['chem_class'] = classify_mol(mol, flags)
            d['formula'] = fm
        else:
            fm = r0.data_formula if isinstance(r0.data_formula, str) else ''
            if fm:
                d.update(elemental(parse_formula(fm)))
                d['formula'] = fm
            d['chem_class'] = classify_name(r0.name_verbatim)
        rows.append(d)
    desc = pd.DataFrame(rows)
    # mixtures/fractions (e.g. 'Light naphtha', 'Aromatics') stay unresolved without formula
    desc['is_mixture_label'] = ((desc.resolved == 0) & desc.get('nC', pd.Series(np.nan)).isna()).astype(int)
    desc.to_csv(os.path.join(TAB, 'compound_descriptors.csv'), index=False, encoding='utf-8-sig')
    print('names', len(ident), 'resolved names', int(ident.resolved.sum()), 'formula conflicts',
          int((~ident.pubchem_formula_agree).sum()), 'identities', len(desc),
          'resolved identities', int(desc.resolved.sum()))
    print(desc.chem_class.value_counts().to_string())


if __name__ == '__main__':
    main()
