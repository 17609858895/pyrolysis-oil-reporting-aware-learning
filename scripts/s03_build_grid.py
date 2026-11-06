"""Step 3 - experiment x compound grid with censoring bounds (the learning table).

Each (experiment e, candidate compound c) pair carries an interval [lo, hi] on its GC-MS area%:
  quantified tier (Q), reported value y      -> exact  lo = hi = y
  Q, not listed                              -> left-censored   (0, L_e),  L_e = min reported value in e
  Q, listed without value                    -> interval [L_e, 100]
  presence-only tier (P), listed             -> interval [L_lo, 100]    (L_lo = 10th pct of Q thresholds)
  P, not listed                              -> left-censored   (0, L_hi)  (L_hi = 90th pct of Q thresholds)
Outputs: tables/grid.parquet, tables/grid_meta.json
"""
import os, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
import sys
MIN_STUDIES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = sys.argv[2] if len(sys.argv) > 2 else 'grid'
Q_LO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10     # P-tier listed:     y >= Q_LO quantile of L_e
Q_HI = float(sys.argv[4]) if len(sys.argv) > 4 else 0.90     # P-tier not listed: y <  Q_HI quantile of L_e
BROAD = len(sys.argv) > 5 and sys.argv[5] == 'broad'          # keep only broad quantified profiles

FEED_ORDER = ['woody', 'herbaceous', 'agro_food_residue', 'oilseed', 'lignin', 'cellulose_textile', 'algae',
              'n_rich_biopolymer', 'sludge', 'plastic', 'tire_rubber', 'biomass_plastic_blend',
              'polymer_composite_ewaste', 'fossil', 'biomass_fossil_blend', 'other', 'unknown']
CAT_MAP = {'none': 'none', 'zeolite': 'zeolite', 'metal_zeolite': 'zeolite', 'zeolite_plus_basic': 'zeolite',
           'basic_oxide': 'basic_oxide', 'supported_metal': 'metal', 'metal_oxide_salt': 'metal',
           'molten_salt': 'other_cat', 'solid_acid': 'other_cat', 'carbon': 'other_cat',
           'clay_silicate': 'other_cat', 'other_catalyst': 'other_cat'}
CAT_ORDER = ['none', 'zeolite', 'basic_oxide', 'metal', 'other_cat']
TECH_ORDER = ['conventional', 'fast', 'slow', 'intermediate', 'steam']
COMP_FEATS = ['nC', 'nH', 'nO', 'nN', 'nS', 'nX', 'DBE', 'H_C', 'O_C', 'N_C', 'MW', 'logP', 'TPSA', 'HBD', 'HBA',
              'n_arom_rings', 'n_rings', 'frac_csp3', 'n_rot', 'f_phenol', 'f_methoxyphenol', 'f_carboxylic_acid',
              'f_ester', 'f_aldehyde', 'f_ketone', 'f_alcohol', 'f_ether', 'f_furan', 'f_n_aromatic', 'f_amine',
              'f_amide', 'f_nitrile', 'f_sulfur', 'f_halogen', 'f_alkene', 'f_anhydrosugar']


def experiment_features(exp):
    X = pd.DataFrame(index=exp.index)
    for f in FEED_ORDER[:-2]:
        X['feed_' + f] = (exp.feed_family == f).astype(int)
    X['plastic_frac'] = exp.plastic_frac
    cat = exp.cat_family.map(CAT_MAP).fillna('other_cat')
    for c in CAT_ORDER[1:]:
        X['cat_' + c] = (cat == c).astype(int)
    for t in TECH_ORDER[1:]:
        X['tech_' + t] = (exp.tech_mode == t).astype(int)
    X['microwave'] = exp.is_microwave
    X['copyrolysis'] = exp.is_copyrolysis
    X['T_mid'] = exp.T_mid
    X['T_span'] = exp.T_span
    return X


def main():
    rec = pd.read_csv(os.path.join(TAB, 'records_clean.csv'))
    exp = pd.read_csv(os.path.join(TAB, 'experiments.csv'))
    ident = pd.read_csv(os.path.join(TAB, 'compound_identity.csv'))
    desc = pd.read_csv(os.path.join(TAB, 'compound_descriptors.csv'))

    primary = (exp.is_review == 0) & (exp.is_downstream == 0) & (exp.feed_family != 'oil_upgrading')
    exp['tier'] = np.where(exp.eligible == 1, 'Q', np.where(primary & (exp.n_records >= 8), 'P', 'X'))
    if BROAD:          # informative-censoring sensitivity: drop narrow / selective quantified lists
        exp.loc[(exp.tier == 'Q') & ~((exp.n_numeric >= 20) & (exp.sum_ra >= 60)), 'tier'] = 'X'
    exp = exp[exp.tier != 'X'].copy()

    rec['name_verbatim'] = rec['name_verbatim'].astype(str).str.strip()
    rec = rec.merge(ident[['name_verbatim', 'identity']], on='name_verbatim', how='left')
    rec = rec[rec.exp_id.isin(exp.exp_id)]
    good_ids = set(desc.loc[(desc.resolved == 1), 'identity'])
    rec = rec[rec.identity.isin(good_ids)]

    # candidate vocabulary: structure-resolved compounds listed in >= MIN_STUDIES distinct primary studies
    n_st = rec.groupby('identity').record_id.nunique()
    vocab = sorted(n_st[n_st >= MIN_STUDIES].index)
    print('candidate compounds', len(vocab))

    Lq = exp.loc[exp.tier == 'Q', 'L_report']
    L_lo, L_hi = float(Lq.quantile(Q_LO)), float(Lq.quantile(Q_HI))

    agg = rec[rec.identity.isin(vocab)].groupby(['exp_id', 'identity']).agg(
        y=('ra', lambda s: s[s > 0].sum() if (s > 0).any() else np.nan),
        has_zero=('ra', lambda s: (s == 0).any())).reset_index()
    grid = pd.MultiIndex.from_product([exp.exp_id.values, vocab], names=['exp_id', 'identity']).to_frame(index=False)
    grid = grid.merge(agg, on=['exp_id', 'identity'], how='left')
    grid['listed'] = grid.set_index(['exp_id', 'identity']).index.isin(agg.set_index(['exp_id', 'identity']).index)
    grid = grid.merge(exp[['exp_id', 'record_id', 'tier', 'L_report']], on='exp_id', how='left')
    q = grid.tier == 'Q'
    lo = np.zeros(len(grid)); hi = np.zeros(len(grid)); ctype = np.empty(len(grid), dtype=object)
    exact = q & grid.listed & grid.y.notna()
    lo[exact], hi[exact], ctype[exact] = grid.y[exact], grid.y[exact], 'exact'
    m = q & grid.listed & grid.y.isna() & ~grid.has_zero.fillna(False).astype(bool)
    lo[m], hi[m], ctype[m] = grid.L_report[m], 100.0, 'interval'
    m = q & (~grid.listed | (grid.listed & grid.y.isna() & grid.has_zero.fillna(False).astype(bool)))
    lo[m], hi[m], ctype[m] = 0.0, grid.L_report[m], 'left'
    m = (~q) & grid.listed
    lo[m], hi[m], ctype[m] = L_lo, 100.0, 'interval'
    m = (~q) & (~grid.listed)
    lo[m], hi[m], ctype[m] = 0.0, L_hi, 'left'
    grid['lo'], grid['hi'], grid['ctype'] = lo, hi, ctype
    grid['detected'] = grid.listed.astype(int)
    grid['L_eval'] = np.where(q, grid.L_report, np.nan)

    XE = experiment_features(exp.set_index('exp_id'))
    XC = desc.set_index('identity').loc[vocab, COMP_FEATS].fillna(0)
    grid = grid.join(XE, on='exp_id').join(XC, on='identity')
    grid = grid.join(desc.set_index('identity')[['chem_class', 'example_name']], on='identity')
    grid.to_parquet(os.path.join(TAB, f'{OUT}.parquet'), index=False)
    if OUT == 'grid':
        exp.to_csv(os.path.join(TAB, 'experiments_tiered.csv'), index=False, encoding='utf-8-sig')
    meta = {'n_candidates': len(vocab), 'L_lo': L_lo, 'L_hi': L_hi, 'min_studies': MIN_STUDIES,
            'exp_features': list(XE.columns), 'comp_features': COMP_FEATS,
            'n_exp_Q': int((exp.tier == 'Q').sum()), 'n_exp_P': int((exp.tier == 'P').sum()),
            'n_study_Q': int(exp.loc[exp.tier == 'Q', 'record_id'].nunique()),
            'n_study_P': int(exp.loc[exp.tier == 'P', 'record_id'].nunique()),
            'n_rows': len(grid), 'ctype_counts': grid.ctype.value_counts().to_dict()}
    json.dump(meta, open(os.path.join(TAB, f'{OUT}_meta.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))
    print(grid.groupby('tier').detected.mean())


if __name__ == '__main__':
    main()
