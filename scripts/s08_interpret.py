"""Step 8 - two-channel interpretation of the cure-AFT model and counterfactual product-slate curves.

Channel 1 (formation): classifier pi(x) = P(compound formed in the oil)
Channel 2 (abundance): AFT location mu(x) = E[ln area% | formed]
TreeSHAP contributions of each channel are aggregated into feature groups.
Counterfactual curves: class composition E[Y 1(Y>=0.1%)] summed by class for typical experiments of each
feedstock family while sweeping temperature with/without zeolite (associational, model-based; compared with
within-study contrasts in s05/s07).
Outputs: tables/shap_groups.csv, tables/shap_top_features.csv, tables/counterfactual_curves.csv
"""
import os, sys, json, warnings
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s04_benchmark import fit_cure_aft, build_X, aft_outputs   # noqa
from s05_within_study_causal import CLASS_GROUPS               # noqa

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
DIST = sys.argv[1] if len(sys.argv) > 1 else 'normal'
SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def feature_group(f):
    if f.startswith('feed_') or f == 'plastic_frac':
        return 'Feedstock'
    if f.startswith('cat_'):
        return 'Catalyst'
    if f in ('T_mid', 'T_span') or f.startswith('tech_') or f in ('microwave', 'copyrolysis'):
        return 'Temperature & reactor'
    if f.startswith('prior_'):
        return 'Compound prior'
    return 'Molecular structure'


def main():
    grid = pd.read_parquet(os.path.join(TAB, 'grid.parquet'))
    meta = json.load(open(os.path.join(TAB, 'grid_meta.json')))
    feats = meta['exp_features'] + meta['comp_features']
    names = feats + ['prior_det', 'prior_logab']
    Xtr, _ = build_X(grid, grid.iloc[:5], feats, 0, True)
    aft, clf = fit_cure_aft(Xtr, grid.lo.values, grid.hi.values, grid.detected.values == 1, DIST, SCALE, 0)
    q = grid.tier.values == 'Q'
    rng = np.random.default_rng(0)
    idx = rng.choice(np.where(q)[0], size=min(12000, q.sum()), replace=False)
    D = xgb.DMatrix(Xtr[idx], feature_names=names)
    rows = []
    for ch, bst in [('formation (pi)', clf), ('abundance (mu)', aft)]:
        bst.feature_names = names
        sh = bst.predict(D, pred_contribs=True)[:, :-1]
        imp = np.abs(sh).mean(0)
        for f, v in zip(names, imp):
            rows.append({'channel': ch, 'feature': f, 'group': feature_group(f), 'mean_abs_shap': v})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(TAB, 'shap_top_features.csv'), index=False)
    G = S.groupby(['channel', 'group']).mean_abs_shap.sum().reset_index()
    G['share'] = G.mean_abs_shap / G.groupby('channel').mean_abs_shap.transform('sum')
    G.to_csv(os.path.join(TAB, 'shap_groups.csv'), index=False)
    print(G.round(3).to_string())
    for ch in S.channel.unique():
        print(ch); print(S[S.channel == ch].nlargest(12, 'mean_abs_shap')[['feature', 'mean_abs_shap']].round(3).to_string())

    # counterfactual composition curves
    inv = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
    exp_cols = meta['exp_features']
    cand = grid.drop_duplicates('identity').set_index('identity')
    comp = cand[meta['comp_features']]
    prior = pd.DataFrame(Xtr[:, -2:], columns=['prior_det', 'prior_logab'], index=grid.identity.values)
    prior = prior[~prior.index.duplicated()].loc[comp.index]
    out = []
    E = grid[grid.tier == 'Q'].drop_duplicates('exp_id')
    for fam in ['woody', 'herbaceous', 'agro_food_residue', 'sludge', 'plastic', 'n_rich_biopolymer', 'lignin']:
        base = E[E['feed_' + fam] == 1][exp_cols].median()
        if base.isna().all():
            continue
        for cat in ['none', 'zeolite']:
            for T in np.arange(400, 751, 50):
                x = base.copy(); x['T_mid'] = T; x['T_span'] = 0
                for c in [k for k in exp_cols if k.startswith('cat_')]:
                    x[c] = 0
                if cat == 'zeolite':
                    x['cat_zeolite'] = 1
                Xe = np.c_[np.tile(x.values, (len(comp), 1)), comp.values, prior.values]
                Dm = xgb.DMatrix(Xe, feature_names=names)
                pi = clf.predict(Dm); mu = aft.predict(Dm, output_margin=True)
                _, _, ey = aft_outputs(mu, np.full(len(mu), 0.1), DIST, SCALE)
                contrib = pd.Series(pi * ey, index=comp.index)
                cls = cand.loc[comp.index, 'chem_class'].map(inv).fillna('Other')
                agg = contrib.groupby(cls.values).sum()
                for k, v in agg.items():
                    out.append({'feed_family': fam, 'catalyst': cat, 'T': T, 'class': k, 'pred_area': v,
                                'share': v / agg.sum()})
    pd.DataFrame(out).to_csv(os.path.join(TAB, 'counterfactual_curves.csv'), index=False)
    print('curves written')


if __name__ == '__main__':
    main()
