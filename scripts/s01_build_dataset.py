"""Step 1 - clean the literature-derived compound table and build experiment-level units.

Outputs (ml_study/tables):
  records_clean.csv      one row per reported compound record with harmonised descriptors
  experiments.csv        one row per experiment (study x feedstock x catalyst x technology x T x note)
Design decisions are recorded in ml_study/evidence/data_decisions.md.
"""
import os, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('PYROLYSIS_DATA', os.path.join(ROOT, 'data', 'compound_records.xlsx'))
TAB = os.path.join(ROOT, 'tables')

REVIEW_IDS = {9, 37, 45, 144, 162, 222}           # titles contain "review"/"recent advances"
DOWNSTREAM_IDS = {1, 2, 21, 30, 36, 116, 141, 177, 204, 241, 242, 243, 247, 249, 225}  # separation/upgrading of oils


def norm(s):
    if pd.isna(s):
        return ''
    return re.sub(r'\s+', ' ', str(s).replace('\n', ' ')).strip()


# ---------------------------------------------------------------- feedstock
def feedstock_family(f):
    s = f.lower()
    if s == '':
        return 'unknown'
    plastic_kw = ['hdpe', 'ldpe', 'pvc', 'pet', 'polystyrene', 'polyethylene', 'polypropylene', 'polycarbonate',
                  'plastic', 'mask', 'nitrile glove', 'ro membrane', 'mlp', 'nrpe', 'microplastic', 'pp-', 'ps+',
                  'pp', 'ps']
    biomass_kw = ['wood', 'sawdust', 'straw', 'husk', 'biomass', 'seed', 'pine', 'rsp', 'stover', 'corncob',
                  'shell', 'mahua', 'bark']
    is_plastic = any(re.search(r'(^|[^a-z])' + re.escape(k) + r'($|[^a-z])', s) for k in plastic_kw)
    is_biomass = any(k in s for k in biomass_kw)
    if 'tire oil' in s or 'tyre oil' in s or 'bio-oil' in s or 'coal tar' in s:
        return 'oil_upgrading'
    if ('tire' in s or 'tyre' in s or 'waste rubber' in s):
        return 'tire_rubber'
    if 'oil shale' in s and is_biomass:
        return 'biomass_fossil_blend'
    if 'oil shale' in s or 'coal' in s:
        return 'fossil'
    if is_plastic and is_biomass:
        return 'biomass_plastic_blend'
    if 'circuit board' in s or 'wind turbine' in s or 'foil-laminated' in s:
        return 'polymer_composite_ewaste'
    if is_plastic:
        return 'plastic'
    if 'sludge' in s or 'biosolid' in s:
        return 'sludge'
    if 'chitin' in s or 'chitosan' in s or 'feather' in s or 'tannery' in s:
        return 'n_rich_biopolymer'
    if 'chlorella' in s or 'laminaria' in s or 'algae' in s:
        return 'algae'
    if 'lignin' in s:
        return 'lignin'
    if 'cellulose' in s or 'cloth' in s or 'textile' in s:
        return 'cellulose_textile'
    if 'castor' in s or 'mahua' in s or 'rapeseed' in s:
        return 'oilseed'
    if any(k in s for k in ['coffee', 'grape', 'date palm', 'olive', 'areca', 'distillers', 'kitchen', 'food',
                            'grapefruit', 'cranberry', 'cocoa', 'açai', 'acai', 'cassava', 'citrus', 'dahlia',
                            'peanut', 'palm shell', 'compost']):
        return 'agro_food_residue'
    if any(k in s for k in ['straw', 'husk', 'stalk', 'corncob', 'grass', 'bagasse', 'stover', 'alternanthera',
                            'perilla', 'crop']):
        return 'herbaceous'
    if any(k in s for k in ['wood', 'sawdust', 'bark', 'pine', 'beech', 'fir', 'eucalypt', 'bamboo', 'mdf',
                            'particleboard', 'plywood', 'salix', 'cedar', 'oak', 'dalbergia', 'softwood',
                            'hardwood', 'bambusa', 'oil palm trunk', 'insulation']):
        return 'woody'
    if 'biomass' in s:
        return 'herbaceous'
    return 'other'


def plastic_fraction(f, fam):
    """Mass fraction of synthetic polymer in the feed (blends parsed when reported)."""
    s = f.lower()
    if fam in ('plastic', 'tire_rubber', 'polymer_composite_ewaste'):
        return 1.0
    if fam == 'biomass_plastic_blend':
        m = re.search(r'(\d+)%\s*mlp', s)
        if m:
            return float(m.group(1)) / 100
        if '50:50' in s:
            return 0.5
        return 0.5
    return 0.0


# ---------------------------------------------------------------- catalyst
def catalyst_family(c):
    s = c.lower()
    if s == '':
        return 'none'
    if 'zsm' in s and ('cao' in s or 'cement' in s):
        return 'zeolite_plus_basic'
    if any(k in s for k in ['zsm', 'hzsm', 'y-type', 'hy ', 'hy(', 'beta', 'mcm', 'sba', 'itq', 'cz-5', 'bh']):
        if any(k in s for k in ['ni', 'pd', 'pt', 'fe,', 'co ', 'zn', 'metal', 'ga']) and 'cz-5' not in s:
            return 'metal_zeolite'
        return 'zeolite'
    if s.startswith('hy'):
        return 'zeolite'
    if 'cao' in s or 'mgo' in s or 'cement' in s:
        return 'basic_oxide'
    if any(k in s for k in ['ni/', 'ni–', 'ni-', 'pd/', 'ru/', 'ni %', 'ni(no', 'pt']):
        return 'supported_metal'
    if 'char' in s:
        return 'carbon'
    if any(k in s for k in ['bentonite', 'attapulgite', 'al2o3.4sio2', 'kieselguhr']):
        return 'clay_silicate'
    if 'mlc' in s:
        return 'molten_salt'
    if 'tossa' in s:
        return 'solid_acid'
    if any(k in s for k in ['zno', 'zro2', 'wo3', 'mos2', 'mgzr', 'sncl2', 'fecl', 'copper', 'nio']):
        return 'metal_oxide_salt'
    return 'other_catalyst'


def catalyst_loading(c):
    m = re.search(r'(\d+(?:\.\d+)?)\s*\.?\s*wt', c.lower())
    if m:
        return float(m.group(1))
    m = re.search(r'\((\d+)%\)', c)
    if m:
        return float(m.group(1))
    return np.nan


# ---------------------------------------------------------------- technology
def tech_flags(t, catfam, fam):
    s = t.lower()
    mode = 'conventional'
    if 'fast' in s or 'pysolysis' in s:
        mode = 'fast'
    elif 'slow' in s:
        mode = 'slow'
    elif 'intermediate' in s:
        mode = 'intermediate'
    elif 'steam' in s:
        mode = 'steam'
    return {
        'tech_mode': mode,
        'is_microwave': int('microwave' in s),
        'is_copyrolysis': int('co-pyrolysis' in s or 'copyrol' in s or fam.endswith('blend')),
        'is_catalytic': int(catfam != 'none' or 'catalytic' in s),
        'is_two_step': int('two-step' in s or 'staged' in s),
        'is_molten_salt': int('molten' in s),
    }


# ---------------------------------------------------------------- temperature
def parse_temperature(t):
    s = norm(t).replace('–', '-').replace('—', '-')
    if s == '' or s.lower() == 'nan':
        return np.nan, np.nan, 0
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
    if not nums:
        return np.nan, np.nan, 0
    upper_only = int(s.startswith('<'))
    if len(nums) == 1:
        return nums[0], 0.0, upper_only
    lo, hi = min(nums), max(nums)
    return (lo + hi) / 2, hi - lo, upper_only


def main():
    df = pd.read_excel(DATA, 'Extracted data (8738)')
    refs = pd.read_excel(DATA, 'References')
    for c in ['feedstock_class', 'catalyst_type', 'technology_type', 'temperature_c', 'separation_technique',
              'quant_unit_note', 'name_verbatim', 'chemical_formula']:
        df[c] = df[c].map(norm)
    df['row_id'] = np.arange(len(df))
    df['ra'] = pd.to_numeric(df['relative_area_percentage'].astype(str).str.replace(' ', ''), errors='coerce')
    df['is_review'] = df.record_id.isin(REVIEW_IDS).astype(int)
    df['is_downstream'] = (df.record_id.isin(DOWNSTREAM_IDS) | (df.separation_technique != '')).astype(int)

    # temperature column of record 187 increments row-by-row (extraction artefact) -> experiment median
    temp_raw = df['temperature_c'].copy()
    m187 = df.record_id == 187
    df.loc[m187, 'temperature_c'] = ''
    key = ['record_id', 'feedstock_class', 'catalyst_type', 'technology_type', 'temperature_c',
           'separation_technique', 'quant_unit_note']
    df['exp_key'] = df[key].astype(str).agg(' | '.join, axis=1)
    codes = {k: i for i, k in enumerate(pd.unique(df['exp_key']))}
    df['exp_id'] = df['exp_key'].map(codes)

    T = df['temperature_c'].map(parse_temperature)
    df['T_mid'] = [x[0] for x in T]
    df['T_span'] = [x[1] for x in T]
    df['T_upper_only'] = [x[2] for x in T]
    t187 = temp_raw[m187].map(lambda x: parse_temperature(x)[0])
    df.loc[m187, 'T_mid'] = t187.groupby(df.loc[m187, 'exp_id']).transform('median')
    df.loc[m187, 'T_span'] = 0.0

    df['feed_family'] = df['feedstock_class'].map(feedstock_family)
    df['plastic_frac'] = [plastic_fraction(f, fam) for f, fam in zip(df.feedstock_class, df.feed_family)]
    df['cat_family'] = df['catalyst_type'].map(catalyst_family)
    df['cat_loading'] = df['catalyst_type'].map(catalyst_loading)
    tf = pd.DataFrame([tech_flags(t, c, f) for t, c, f in zip(df.technology_type, df.cat_family, df.feed_family)])
    df = pd.concat([df.reset_index(drop=True), tf], axis=1)
    df['unit_wt'] = df['quant_unit_note'].str.contains('wt', case=False).astype(int)
    df['is_pygcms'] = df['quant_unit_note'].str.contains('py-gc|py–gc|analytical', case=False).astype(int)

    # experiment table
    g = df.groupby('exp_id')
    exp = g.agg(record_id=('record_id', 'first'), feedstock_class=('feedstock_class', 'first'),
                catalyst_type=('catalyst_type', 'first'), technology_type=('technology_type', 'first'),
                temperature_raw=('temperature_c', 'first'), quant_unit_note=('quant_unit_note', 'first'),
                feed_family=('feed_family', 'first'), plastic_frac=('plastic_frac', 'first'),
                cat_family=('cat_family', 'first'), cat_loading=('cat_loading', 'first'),
                tech_mode=('tech_mode', 'first'), is_microwave=('is_microwave', 'first'),
                is_copyrolysis=('is_copyrolysis', 'first'), is_catalytic=('is_catalytic', 'first'),
                is_two_step=('is_two_step', 'first'), is_molten_salt=('is_molten_salt', 'first'),
                T_mid=('T_mid', 'median'), T_span=('T_span', 'first'), T_upper_only=('T_upper_only', 'first'),
                unit_wt=('unit_wt', 'first'), is_review=('is_review', 'first'),
                is_downstream=('is_downstream', 'first'),
                n_records=('row_id', 'size'), n_numeric=('ra', 'count'), sum_ra=('ra', 'sum'),
                max_ra=('ra', 'max')).reset_index()
    pos = df[df.ra > 0].groupby('exp_id')['ra']
    exp['L_report'] = exp.exp_id.map(pos.min())          # study-specific reporting threshold (min reported value)
    exp['L_q05'] = exp.exp_id.map(pos.quantile(0.05))
    exp = exp.merge(refs.rename(columns={'title': 'study_title'}), on='record_id', how='left')

    # eligibility for the censored product-slate analysis
    exp['eligible'] = ((exp.is_review == 0) & (exp.is_downstream == 0) & (exp.n_numeric >= 8) &
                       (exp.sum_ra >= 20) & (exp.sum_ra <= 105) & (exp.unit_wt == 0) &
                       (exp.feed_family != 'oil_upgrading') & (exp.L_report > 0)).astype(int)

    df.to_csv(os.path.join(TAB, 'records_clean.csv'), index=False, encoding='utf-8-sig')
    exp.to_csv(os.path.join(TAB, 'experiments.csv'), index=False, encoding='utf-8-sig')
    e = exp[exp.eligible == 1]
    print('records', len(df), 'experiments', len(exp), 'eligible experiments', len(e),
          'eligible studies', e.record_id.nunique(), 'eligible records', int(e.n_numeric.sum()))
    print(e.feed_family.value_counts().to_string())
    print(e.cat_family.value_counts().to_string())
    print(e.tech_mode.value_counts().to_string())
    print(e[['T_mid', 'n_numeric', 'sum_ra', 'L_report']].describe().to_string())
    print('missing T in eligible', e.T_mid.isna().sum())


if __name__ == '__main__':
    main()
