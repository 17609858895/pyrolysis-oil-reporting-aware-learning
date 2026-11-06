"""Step 15 - supplementary analyses reported in the Supporting Information.

Modes (python scripts/s15_si_analyses.py <mode> [jobs]):
  stack   NNLS weights of the 22-learner stack in each of the 15 outer folds, using the same outer splits, seeds,
          inner study folds and out-of-fold prior as s04 (only the weight-fitting step is repeated);
          also the inner out-of-fold RMSE of each base learner.          -> tables/si_stack_weights.csv
  causal  sensitivity of the within-study random-effects estimates (zeolite and any catalyst) to the sampling
          variance imputed for single-contrast studies, to the pseudo-count and to single studies; sensitivity
          of pooled and within-study DML to the pseudo-count and to the number and assignment of grouped folds.
                                                                         -> tables/si_causal_*.csv
  ale     accumulated local effects (ALE) and partial dependence (PD) of temperature on the predicted class
          shares of the product slate, pooled over all quantified oils (final model, logistic, sigma = 1.5).
                                                                         -> tables/si_temperature_ale_pd.csv
"""
import os, sys, json, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')


# =================================================================== stacking weights
def stack_fold(grid, feats, test_studies, seed, fold):
    from s04_benchmark import build_X, make_folds, regressors, CONVENTIONAL
    tr = grid[~grid.record_id.isin(test_studies)].copy()
    te = grid[(grid.tier == 'Q') & grid.record_id.isin(test_studies)].copy()
    Xtr, _ = build_X(tr, te, feats, seed, True)
    posm = tr.ctype.values == 'exact'
    Xpos, ypos = Xtr[posm], np.log10(tr.lo.values[posm])
    st = tr.record_id.values[posm]; oof = np.zeros((posm.sum(), len(CONVENTIONAL)))
    for f in make_folds(np.unique(st), 3, seed + 7):
        mm = np.isin(st, list(f))
        for j, b in enumerate(CONVENTIONAL):
            oof[mm, j] = np.ravel(regressors(seed)[b].fit(Xpos[~mm], ypos[~mm]).predict(Xpos[mm]))
    A = np.c_[np.ones(len(oof)), oof]
    coef, _ = nnls(np.c_[A, -np.ones(len(oof))], ypos)
    icpt = coef[0] - coef[-1]; wts = coef[1:-1]
    rows = [{'seed': seed, 'fold': fold, 'learner': b, 'weight': w,
             'oof_rmse': float(np.sqrt(np.mean((oof[:, j] - ypos) ** 2)))}
            for j, (b, w) in enumerate(zip(CONVENTIONAL, wts))]
    stack_pred = icpt + oof @ wts
    rows.append({'seed': seed, 'fold': fold, 'learner': 'Stack (in-sample on inner OOF)', 'weight': np.nan,
                 'oof_rmse': float(np.sqrt(np.mean((stack_pred - ypos) ** 2)))})
    rows.append({'seed': seed, 'fold': fold, 'learner': 'Intercept', 'weight': icpt, 'oof_rmse': np.nan})
    return pd.DataFrame(rows)


def run_stack(jobs):
    from s04_benchmark import make_folds
    grid = pd.read_parquet(os.path.join(TAB, 'grid.parquet'))
    meta = json.load(open(os.path.join(TAB, 'grid_meta.json')))
    feats = meta['exp_features'] + meta['comp_features']
    studies = grid.loc[grid.tier == 'Q', 'record_id'].unique()
    todo = [(make_folds(studies, 5, r)[f], r, f) for r in range(3) for f in range(5)]
    out = Parallel(n_jobs=jobs, verbose=5)(delayed(stack_fold)(grid, feats, ts, r, f) for ts, r, f in todo)
    W = pd.concat(out)
    W.to_csv(os.path.join(TAB, 'si_stack_weights.csv'), index=False)
    s = W[W.learner != 'Intercept'].groupby('learner').agg(weight_mean=('weight', 'mean'), weight_sd=('weight', 'std'),
                                                          n_nonzero=('weight', lambda v: int((v > 1e-6).sum())),
                                                          oof_rmse=('oof_rmse', 'mean'))
    print(s.sort_values('weight_mean', ascending=False).round(3).to_string())


# =================================================================== within-study sensitivity
def pool(per, v):
    from s05_within_study_causal import dersimonian_laird
    rows = []
    for k, g in per.assign(vv=v).groupby('class'):
        mu, se, tau2, I2 = dersimonian_laird(g.lr, g.vv)
        rows.append({'class': k, 'k_studies': len(g), 'effect_lr': mu, 'se': se, 'lo95': mu - 1.96 * se,
                     'hi95': mu + 1.96 * se, 'tau2': tau2, 'I2': I2})
    return pd.DataFrame(rows)


def dml_custom(Y, D, X, groups, k=5, seed=None, demean=False):
    import lightgbm as lgb
    Y, D, X = Y.copy(), D.copy(), X.copy()
    if demean:
        Y = Y - Y.groupby(groups).transform('mean'); D = D - D.groupby(groups).transform('mean')
        X = X - X.groupby(groups).transform('mean')
    ug = np.array(sorted(groups.unique()))
    if seed is None:                                              # GroupKFold as in the main analysis
        from sklearn.model_selection import GroupKFold
        splits = list(GroupKFold(n_splits=k).split(X, Y, groups))
    else:                                                         # random assignment of studies to folds
        rng = np.random.default_rng(seed); rng.shuffle(ug)
        fid = {g: i % k for i, g in enumerate(ug)}
        f = groups.map(fid).values
        splits = [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]
    ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in splits:
        kw = dict(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5, verbose=-1, n_jobs=1,
                  random_state=0)
        my = lgb.LGBMRegressor(**kw).fit(X.iloc[tr], Y.iloc[tr]); md = lgb.LGBMRegressor(**kw).fit(X.iloc[tr], D.iloc[tr])
        ry[te] = Y.iloc[te] - my.predict(X.iloc[te]); rd[te] = D.iloc[te] - md.predict(X.iloc[te])
    theta = (rd * ry).sum() / (rd * rd).sum()
    psi = (ry - theta * rd) * rd; J = (rd * rd).mean()
    se = np.sqrt((pd.Series(psi).groupby(groups.values).sum() ** 2).sum()) / (len(Y) * J)
    return theta, se


def run_causal():
    import s05_within_study_causal as s05
    E, _ = s05.class_table()
    groups = list(s05.CLASS_GROUPS)
    treats = [('zeolite', ['zeolite', 'metal_zeolite', 'zeolite_plus_basic']),
              ('any catalyst', [c for c in E.cat_family.unique() if c != 'none'])]
    # ---- (1) variance assumption for single-contrast studies, (2) pseudo-count, (3) leave-one-study-out
    var_rows, loo_rows = [], []
    for pseudo in [0.1, 0.5, 1.0, 2.0]:
        s05.PSEUDO = pseudo
        for label, treat in treats:
            per, _ = s05.within_study_contrasts(E, groups, treat)
            v_obs = (per.sd ** 2 / per.n)
            has = v_obs > 0
            vmed = v_obs[has].median(); vmax = v_obs[has].max()
            settings = [('median (main)', np.maximum(vmed, 0.05)), ('0.25 x median', np.maximum(0.25 * vmed, 0.05)),
                        ('0.5 x median', np.maximum(0.5 * vmed, 0.05)), ('2 x median', 2 * max(vmed, 0.05)),
                        ('4 x median', 4 * max(vmed, 0.05)), ('maximum replicate variance', vmax)]
            if pseudo != 0.5:
                settings = settings[:1]
            for name, imp in settings:
                v = v_obs.where(has, imp)
                P = pool(per, v).assign(treatment=label, pseudo_count=pseudo, variance=name, v_imputed=imp,
                                        n_imputed=int((~has).sum()), n_study_class=len(per))
                var_rows.append(P)
            if pseudo == 0.5:
                v = v_obs.where(has, np.maximum(vmed, 0.05))
                for sid in sorted(per.record_id.unique()):
                    keep = per.record_id != sid
                    P = pool(per[keep], v[keep]).assign(treatment=label, left_out=sid)
                    loo_rows.append(P)
    s05.PSEUDO = 0.5
    V = pd.concat(var_rows); V.to_csv(os.path.join(TAB, 'si_causal_variance_pseudocount.csv'), index=False)
    L = pd.concat(loo_rows); L.to_csv(os.path.join(TAB, 'si_causal_leave_one_study_out.csv'), index=False)
    # ---- (4) DML: pseudo-count and grouped-fold assignment
    dml_rows = []
    for label, treat in treats:
        sub = E[(E.cat_family == 'none') | E.cat_family.isin(treat)].copy()
        D = sub.cat_family.isin(treat).astype(float)
        Xc = pd.get_dummies(sub[['feed_family', 'tech_mode']]).astype(float)
        Xc['T_mid'] = sub.T_mid.fillna(sub.T_mid.median()); Xc['plastic_frac'] = sub.plastic_frac
        Xc['microwave'] = sub.is_microwave; Xc['copyrolysis'] = sub.is_copyrolysis
        for k in groups:
            for pseudo in [0.1, 0.5, 1.0, 2.0]:
                Y = np.log(sub[k] + pseudo)
                configs = [('GroupKFold k=5 (main)', 5, None)]
                if pseudo == 0.5:
                    configs += [('GroupKFold k=3', 3, None), ('GroupKFold k=10', 10, None)] + \
                               [(f'random study folds k=5, seed {s}', 5, s) for s in range(5)]
                for name, kk, sd in configs:
                    for dm in [False, True]:
                        th, se = dml_custom(Y, D, Xc, sub.record_id, kk, sd, dm)
                        dml_rows.append({'treatment': label, 'class': k, 'pseudo_count': pseudo, 'folds': name,
                                         'estimator': 'within-study DML' if dm else 'pooled DML', 'theta': th,
                                         'se': se, 'lo95': th - 1.96 * se, 'hi95': th + 1.96 * se})
    M = pd.DataFrame(dml_rows); M.to_csv(os.path.join(TAB, 'si_causal_dml_sensitivity.csv'), index=False)
    pd.set_option('display.width', 220)
    z = V[(V.treatment == 'zeolite') & V['class'].isin(['Phenolics', 'N-compounds', 'Aromatic hydrocarbons'])]
    print(z[['class', 'pseudo_count', 'variance', 'effect_lr', 'lo95', 'hi95', 'I2']].round(3).to_string())
    print(L[(L.treatment == 'zeolite') & L['class'].isin(['Phenolics', 'N-compounds'])].groupby('class')
          .effect_lr.agg(['min', 'max']).round(3))
    m = M[(M.treatment == 'zeolite') & M['class'].isin(['Phenolics', 'N-compounds', 'Aromatic hydrocarbons'])]
    print(m.groupby(['class', 'estimator']).theta.agg(['min', 'max']).round(3))


# =================================================================== temperature ALE vs PD
def run_ale():
    import xgboost as xgb
    from s04_benchmark import fit_cure_aft, build_X, compound_prior, aft_outputs
    from s05_within_study_causal import CLASS_GROUPS
    DIST, SCALE = 'logistic', 1.5
    inv = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
    grid = pd.read_parquet(os.path.join(TAB, 'grid.parquet'))
    meta = json.load(open(os.path.join(TAB, 'grid_meta.json')))
    ef, cf = meta['exp_features'], meta['comp_features']
    Xall, _ = build_X(grid, grid.iloc[:5], ef + cf, 0, True)
    cache = os.path.join(ROOT, 'cache'); os.makedirs(cache, exist_ok=True)
    fa, fc = os.path.join(cache, 'final_model_aft.json'), os.path.join(cache, 'final_model_pi.json')
    if os.path.exists(fa) and os.path.exists(fc):
        aft, clf = xgb.Booster(model_file=fa), xgb.Booster(model_file=fc)
    else:
        aft, clf = fit_cure_aft(Xall, grid.lo.values, grid.hi.values, grid.detected.values == 1, DIST, SCALE, 0)
        aft.save_model(fa); clf.save_model(fc)
    cand = grid.drop_duplicates('identity').set_index('identity')
    # Match held-out inference: prior summaries use every training study.
    prior = np.column_stack(compound_prior(grid, cand.reset_index()))
    cls = cand.chem_class.map(inv).fillna('Other').values
    E = grid[(grid.tier == 'Q')].drop_duplicates('exp_id')
    E = E[E.T_mid.notna()]

    def shares(Xe):
        """class shares of the predicted slate for each oil row of Xe (n_oils x n_exp_features)."""
        n, m = len(Xe), len(cand)
        X = np.c_[np.repeat(Xe, m, axis=0), np.tile(cand[cf].values, (n, 1)), np.tile(prior, (n, 1))]
        Dm = xgb.DMatrix(X); pi = clf.predict(Dm); mu = aft.predict(Dm, output_margin=True)
        ey = aft_outputs(mu, np.full(len(mu), 0.1), DIST, SCALE)[2]
        c = pd.DataFrame({'oil': np.repeat(np.arange(n), m), 'cls': np.tile(cls, n), 'v': pi * ey})
        S = c.pivot_table(index='oil', columns='cls', values='v', aggfunc='sum')
        return S.div(S.sum(1), axis=0)

    tcol = ef.index('T_mid')
    rng = np.random.default_rng(0)

    def center(acc, w):
        mid = (acc[1:] + acc[:-1]) / 2
        return acc - (mid * w[:, None]).sum(0) / w.sum()

    def ale_pd(B, label, nq=6):
        z = np.unique(np.quantile(B.T_mid, np.linspace(0, 1, nq + 1)))
        # Interpolated quantiles can form an interval with no observed oil.
        # Merge that interval so every local difference has data support.
        while len(z) > 2:
            ix = np.clip(np.searchsorted(z, B.T_mid.values, side='left'), 1, len(z) - 1)
            counts0 = np.bincount(ix, minlength=len(z))[1:]
            empty = np.flatnonzero(counts0 == 0)
            if not len(empty):
                break
            edge = int(empty[0] + 1)
            z = np.delete(z, edge if edge < len(z) - 1 else edge - 1)
        bins = np.clip(np.searchsorted(z, B.T_mid.values, side='left'), 1, len(z) - 1)
        Dk, counts = [], []
        for k in range(1, len(z)):
            idx = np.where(bins == k)[0]
            lo, hi = B.values[idx].copy(), B.values[idx].copy()
            lo[:, tcol] = z[k - 1]; hi[:, tcol] = z[k]
            Dk.append((shares(hi) - shares(lo)).fillna(0)); counts.append(len(idx))
        cols = sorted(set().union(*[d.columns for d in Dk]))
        Dk = [d.reindex(columns=cols, fill_value=0).values for d in Dk]
        w = np.array(counts, float)
        acc = np.vstack([np.zeros(len(cols)), np.cumsum([d.mean(0) if len(d) else np.zeros(len(cols)) for d in Dk], 0)])
        ale = center(acc, w)
        boots = []
        for _ in range(1000):
            mb = [d[rng.integers(0, len(d), len(d))].mean(0) if len(d) else np.zeros(len(cols)) for d in Dk]
            boots.append(center(np.vstack([np.zeros(len(cols)), np.cumsum(mb, 0)]), w))
        lo95, hi95 = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        pdv = []
        for t in z:
            Xe = B.values.copy(); Xe[:, tcol] = t
            pdv.append(shares(Xe).reindex(columns=cols, fill_value=0).mean().values)
        PD = np.array(pdv)
        rows = []
        for i, t in enumerate(z):
            for j, c in enumerate(cols):
                rows.append({'group': label, 'n_oils': len(B), 'T': t, 'class': c, 'ale': ale[i, j],
                             'ale_lo95': lo95[i, j], 'ale_hi95': hi95[i, j], 'pd': PD[i, j],
                             'pd_centered': PD[i, j] - PD[:, j].mean(),
                             'n_bin_left_of_edge': counts[i - 1] if i else np.nan})
        print(label, 'edges', np.round(z, 1), 'counts', counts, flush=True)
        return rows

    groups = [('All quantified oils', E)]
    for fam in ['woody', 'herbaceous', 'agro_food_residue', 'lignin', 'sludge', 'n_rich_biopolymer', 'plastic']:
        g = E[E['feed_' + fam] == 1]
        if len(g) >= 10 and g.T_mid.nunique() >= 3:
            groups.append((fam, g))
    rows = []
    for label, g in groups:
        rows += ale_pd(g[ef].astype(float).reset_index(drop=True), label, nq=6 if label.startswith('All') else 4)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TAB, 'si_temperature_ale_pd.csv'), index=False)
    pd.set_option('display.width', 250)
    for label in out.group.unique():
        o = out[out.group == label]
        print(label); print(o.pivot_table(index='T', columns='class', values='ale').round(3).to_string())
        print(o.pivot_table(index='T', columns='class', values='pd').round(3).to_string())


if __name__ == '__main__':
    mode = sys.argv[1]
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    {'stack': lambda: run_stack(jobs), 'causal': run_causal, 'ale': run_ale}[mode]()
