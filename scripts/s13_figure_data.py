"""Step 13 - data products for the formal Figures 4-6.

1. exceedance map: final model (modal configuration logistic sigma 1.5, all studies) -> P(area >= 1 %) for ten
   value-added compounds across feedstock families, with and without zeolite, at 500 C;
2. formation-probability heatmap: mean out-of-fold pi (final benchmark) by product class x feedstock family;
3. per-family detection AUC of the final comparison (out-of-fold);
4. applicability domain: per held-out oil, standardized distance to the nearest training oil vs detection AUC.
Outputs in tables/fig_*.csv
"""
import os, sys, json, warnings
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s04_benchmark import fit_cure_aft, build_X, compound_prior, aft_outputs, make_folds  # noqa
from s05_within_study_causal import CLASS_GROUPS                           # noqa

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
DIST, SCALE = 'logistic', 1.5
TARGETS = {'Phenol': 'Phenol', '2-Methoxy-phenol': 'Guaiacol', '2,6-Dimethoxy phenol': 'Syringol',
           '1,2-Benzenediol (Catechol)': 'Catechol', '1,4:3,6-Dianhydro-α-D-glucopyranose (Levoglucosan)': 'Levoglucosan',
           'Acetic Acid': 'Acetic acid', 'Benzene, methyl-': 'Toluene', 'Styrene': 'Styrene', 'Naphthalene': 'Naphthalene',
           'Pyridine': 'Pyridine'}
FAMILIES = ['woody', 'herbaceous', 'agro_food_residue', 'lignin', 'sludge', 'n_rich_biopolymer', 'plastic']
INV = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}


def exceedance_map(grid, meta):
    feats = meta['exp_features'] + meta['comp_features']
    names = feats + ['prior_det', 'prior_logab']
    Xall, _ = build_X(grid, grid.iloc[:5], feats, 0, True)
    cache = os.path.join(ROOT, 'cache')
    fa, fc = os.path.join(cache, 'final_model_aft.json'), os.path.join(cache, 'final_model_pi.json')
    if os.path.exists(fa) and os.path.exists(fc):
        aft, clf = xgb.Booster(model_file=fa), xgb.Booster(model_file=fc)
    else:
        aft, clf = fit_cure_aft(Xall, grid.lo.values, grid.hi.values, grid.detected.values == 1, DIST, SCALE, 0)
        os.makedirs(cache, exist_ok=True)
        aft.save_model(fa); clf.save_model(fc)
    cand = grid.drop_duplicates('identity').set_index('identity')
    tid = cand[cand.example_name.isin(TARGETS)].copy()
    # Training uses cross-fitted priors; inference uses all available training
    # studies, exactly as in the outer-fold benchmark prediction path.
    prior = np.column_stack(compound_prior(grid, tid.reset_index()))
    E = grid[grid.tier == 'Q'].drop_duplicates('exp_id')
    rows = []
    for fam in FAMILIES:
        base = E[E['feed_' + fam] == 1][meta['exp_features']].median()
        for cat in ['none', 'zeolite']:
            for T in [450, 500, 550, 600, 650]:
                x = base.copy(); x['T_mid'] = T; x['T_span'] = 0
                for c in [k for k in meta['exp_features'] if k.startswith('cat_')]:
                    x[c] = 0
                if cat == 'zeolite':
                    x['cat_zeolite'] = 1
                X = np.c_[np.tile(x.values, (len(tid), 1)), tid[meta['comp_features']].values, prior]
                D = xgb.DMatrix(X)
                pi = clf.predict(D); mu = aft.predict(D, output_margin=True)
                for thr in [1.0, 5.0]:
                    p = pi * aft_outputs(mu, np.full(len(mu), thr), DIST, SCALE)[0]
                    for nm, v in zip(tid.example_name, p):
                        rows.append({'compound': TARGETS[nm], 'feed_family': fam, 'catalyst': cat, 'T': T,
                                     'threshold': thr, 'p_exceed': v})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, 'fig_exceedance_map.csv'), index=False)
    # where was each target actually observed >= 1 % (reality check for the map)
    q = grid[(grid.tier == 'Q') & grid.example_name.isin(TARGETS)].copy()
    fam_of = q[[f'feed_{f}' for f in FAMILIES]].idxmax(axis=1).str.replace('feed_', '')
    fam_of[q[[f'feed_{f}' for f in FAMILIES]].sum(1) == 0] = 'other'
    q['feed_family'] = fam_of.values; q['cat'] = np.where(q.cat_zeolite == 1, 'zeolite', np.where(
        q[['cat_basic_oxide', 'cat_metal', 'cat_other_cat']].sum(1) > 0, 'other', 'none'))
    q['obs_ge1'] = ((q.ctype == 'exact') & (q.lo >= 1.0)).astype(int)
    q['compound'] = q.example_name.map(TARGETS)
    obs = q.groupby(['compound', 'feed_family', 'cat']).agg(n_oils=('obs_ge1', 'size'), frac_ge1=('obs_ge1', 'mean'))
    obs.reset_index().to_csv(os.path.join(TAB, 'fig_exceedance_observed.csv'), index=False)
    return aft, clf, Xall


def pd_curves(grid, meta, aft, clf, Xall):
    """Family-level partial dependence of the product slate on temperature with/without zeolite:
    every quantified oil of the family is used as a base condition, then predictions are averaged."""
    ef, cf = meta['exp_features'], meta['comp_features']
    cand = grid.drop_duplicates('identity').set_index('identity')
    prior = np.column_stack(compound_prior(grid, cand.reset_index()))
    cls = cand.chem_class.map(INV).fillna('Other').values
    E = grid[grid.tier == 'Q'].drop_duplicates('exp_id')
    rows = []
    for fam in FAMILIES:
        B = E[E['feed_' + fam] == 1][ef]
        for cat in ['none', 'zeolite']:
            for T in range(400, 701, 50):
                shares = []
                for _, b in B.iterrows():
                    x = b.copy(); x['T_mid'] = T; x['T_span'] = 0
                    for c in [k for k in ef if k.startswith('cat_')]:
                        x[c] = 0
                    if cat == 'zeolite':
                        x['cat_zeolite'] = 1
                    X = np.c_[np.tile(x.values.astype(float), (len(cand), 1)), cand[cf].values, prior]
                    D = xgb.DMatrix(X); pi = clf.predict(D); mu = aft.predict(D, output_margin=True)
                    ey = aft_outputs(mu, np.full(len(mu), 0.1), DIST, SCALE)[2]
                    a = pd.Series(pi * ey).groupby(cls).sum(); shares.append(a / a.sum())
                S = pd.concat(shares, axis=1)
                for k in S.index:
                    rows.append({'feed_family': fam, 'catalyst': cat, 'T': T, 'class': k,
                                 'share': S.loc[k].mean(), 'sd': S.loc[k].std(), 'n_base': S.shape[1]})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'fig_counterfactual_curves_pd.csv'), index=False)


def formation_heatmap(grid, exp):
    P = pd.read_parquet(os.path.join(TAB, 'final_predictions.parquet'))
    P = P[P.method == 'RLCure']
    cls = grid.drop_duplicates('identity').set_index('identity').chem_class.map(INV).fillna('Other')
    P['group'] = P.identity.map(cls)
    P = P.merge(exp[['exp_id', 'feed_family']], on='exp_id')
    H = P[P.feed_family.isin(FAMILIES)].groupby(['group', 'feed_family']).agg(
        pi=('pi', 'mean'), p_listed=('p_det', 'mean'), obs=('detected', 'mean'), n=('detected', 'size')).reset_index()
    H.to_csv(os.path.join(TAB, 'fig_formation_heatmap.csv'), index=False)


def per_family_auc(exp):
    R = pd.read_csv(os.path.join(TAB, 'final_results.csv')).drop_duplicates(['method', 'exp_id', 'seed', 'fold'])
    R = R.merge(exp[['exp_id', 'feed_family']], on='exp_id')
    g = R.groupby(['method', 'feed_family']).agg(det_auc=('det_auc', 'mean'), sd=('det_auc', 'std'),
                                                 n=('exp_id', 'nunique')).reset_index()
    g.to_csv(os.path.join(TAB, 'fig_family_auc.csv'), index=False)


def applicability(grid, meta, exp):
    R = pd.read_csv(os.path.join(TAB, 'final_results.csv')).drop_duplicates(['method', 'exp_id', 'seed', 'fold'])
    R = R[R.method.isin(['RLCure', 'Hurdle+P-cal', 'LightGBM-tuned'])]
    ef = meta['exp_features']
    E = grid.drop_duplicates('exp_id').set_index('exp_id')[ef + ['record_id', 'tier']]
    studies = grid.loc[grid.tier == 'Q', 'record_id'].unique()
    rows = []
    for r in range(3):
        folds = make_folds(studies, 5, r)
        for f, ts in enumerate(folds):
            test = E[(E.tier == 'Q') & E.record_id.isin(ts)].index
            train = E[~E.record_id.isin(ts)].index
            train_x = E.loc[train, ef].astype(float)
            test_x = E.loc[test, ef].astype(float)
            med = train_x.median()
            train_x, test_x = train_x.fillna(med), test_x.fillna(med)
            center = train_x.mean()
            scale = train_x.std().replace(0, 1).fillna(1)
            ztr, zte = (train_x - center) / scale, (test_x - center) / scale
            D = np.sqrt(((zte.values[:, None, :] - ztr.values[None, :, :]) ** 2).sum(-1))
            for e, d in zip(test, D.min(1)):
                rows.append({'exp_id': e, 'seed': r, 'fold': f, 'nn_dist': d})
    A = pd.DataFrame(rows)
    # Assign one distance group to each oil/repeat before copying rows by model.
    A['domain'] = pd.qcut(A.nn_dist.rank(method='first'), 3, labels=['near', 'intermediate', 'far'])
    A = A.merge(R[['method', 'exp_id', 'seed', 'fold', 'det_auc', 'top10']], on=['exp_id', 'seed', 'fold'])
    A.to_csv(os.path.join(TAB, 'fig_applicability.csv'), index=False)
    print(A.groupby(['domain', 'method']).det_auc.mean().unstack().round(3))


def main():
    grid = pd.read_parquet(os.path.join(TAB, 'grid.parquet'))
    meta = json.load(open(os.path.join(TAB, 'grid_meta.json')))
    exp = pd.read_csv(os.path.join(TAB, 'experiments_tiered.csv'))
    per_family_auc(exp)
    formation_heatmap(grid, exp)
    applicability(grid, meta, exp)
    aft, clf, Xall = exceedance_map(grid, meta)
    pd_curves(grid, meta, aft, clf, Xall)
    print('figure data written')


if __name__ == '__main__':
    main()
