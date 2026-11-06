"""Step 4 - leave-study-out benchmark: conventional (positive-only) learners vs censoring-aware survival learners.

All methods are scored on the SAME held-out quantified-tier (Q) studies; presence-only (P) experiments of
held-out studies are also removed from training. Every method receives the same features:
experiment descriptors + compound structure descriptors + out-of-fold compound prior (detection frequency and
mean log abundance computed only from training studies).

Metrics per held-out experiment (averaged):
  det_auc / det_ap  listed vs non-listed candidate compounds (detection above the study's reporting limit L_e)
  top10             recall of the 10 most abundant observed candidates among the predicted top-10 contributors
  rho_abund         Spearman rho between predicted (conditional) and observed abundance of listed compounds
  mae_log           |log10 predicted - log10 observed| for listed compounds
Pooled probabilistic metrics: Brier score of P(listed), exceedance AUC/Brier at t = 1 % and 5 % area.
"""
import os, json, argparse, warnings, time
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")          # one thread per worker (avoids OpenMP oversubscription on Windows)
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, nnls
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model as lm
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
LN10 = np.log(10.0)
THRESHOLDS = [1.0, 5.0]
FIXED_L = 0.35            # pooled median reporting limit (ablation "fixedL")


# =================================================================== conventional regressors
from sklearn.base import BaseEstimator, RegressorMixin


class SubsampledGPR(BaseEstimator, RegressorMixin):
    """Exact GP on a random subset (O(n^3) cost); keeps the 22-learner benchmark tractable."""
    def __init__(self, n_max=1500, seed=0):
        self.n_max, self.seed = n_max, seed

    def fit(self, X, y):
        idx = np.random.default_rng(self.seed).permutation(len(X))[:self.n_max]
        self.gp_ = GaussianProcessRegressor(kernel=ConstantKernel() * RBF(5.0) + WhiteKernel(), normalize_y=True,
                                            random_state=self.seed).fit(X[idx], y[idx])
        return self

    def predict(self, X):
        return self.gp_.predict(X)


def regressors(seed=0):
    sc = lambda m: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), m)
    im = lambda m: make_pipeline(SimpleImputer(strategy='median'), m)
    return {
        'OLS': sc(lm.LinearRegression()),
        'Ridge': sc(lm.Ridge(alpha=10.0)),
        'Lasso': sc(lm.Lasso(alpha=0.01, max_iter=5000)),
        'ElasticNet': sc(lm.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
        'BayesRidge': sc(lm.BayesianRidge()),
        'Huber': sc(lm.HuberRegressor(max_iter=500)),
        'PLS': sc(PLSRegression(n_components=10)),
        'KNN': sc(KNeighborsRegressor(n_neighbors=15, weights='distance')),
        'SVR': sc(SVR(C=3.0, epsilon=0.1)),
        'KernelRidge': sc(KernelRidge(alpha=1.0, kernel='rbf', gamma=0.02)),
        'GPR': sc(SubsampledGPR(n_max=1000, seed=seed)),
        'DecisionTree': im(DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=seed)),
        'RandomForest': im(RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                                                 n_jobs=1, random_state=seed)),
        'ExtraTrees': im(ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, n_jobs=1,
                                             random_state=seed)),
        'Bagging': im(BaggingRegressor(n_estimators=100, random_state=seed)),
        'AdaBoost': im(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.05,
                                         random_state=seed)),
        'GBDT': im(GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             random_state=seed)),
        'HistGB': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=20,
                                                early_stopping=False, random_state=seed),
        'XGBoost': xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, min_child_weight=5, n_jobs=1, random_state=seed),
        'LightGBM': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, n_jobs=1,
                                      random_state=seed, verbose=-1),
        'CatBoost': CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05, verbose=0, thread_count=1,
                                      random_seed=seed),
        'MLP': sc(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600, early_stopping=True,
                               random_state=seed)),
    }


CONVENTIONAL = ['OLS', 'Ridge', 'Lasso', 'ElasticNet', 'BayesRidge', 'Huber', 'PLS', 'KNN', 'SVR', 'KernelRidge',
                'GPR', 'DecisionTree', 'RandomForest', 'ExtraTrees', 'Bagging', 'AdaBoost', 'GBDT', 'HistGB',
                'XGBoost', 'LightGBM', 'CatBoost', 'MLP']


# =================================================================== censored building blocks
def aft_cdf(z, dist):
    if dist == 'normal':
        return stats.norm.cdf(z)
    if dist == 'logistic':
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return 1 - np.exp(-np.exp(np.clip(z, -50, 50)))     # 'extreme' (Gumbel) as in XGBoost


def _pdf_grid(dist):
    zg = np.linspace(-6, 6, 241)
    if dist == 'normal':
        pdf = stats.norm.pdf(zg)
    elif dist == 'logistic':
        pdf = np.exp(-zg) / (1 + np.exp(-zg)) ** 2
    else:
        pdf = np.exp(zg - np.exp(zg))
    return zg, pdf / np.trapz(pdf, zg)


def aft_outputs(mu_ln, L, dist, scale):
    """Exceedance probability and finite-grid log-abundance/contribution scores.
    Integration uses a normalized error density on [-6, 6], not an unrestricted mean.
    """
    zL = (np.log(L) - mu_ln) / scale
    p = 1 - aft_cdf(zL, dist)
    zg, pdf = _pdf_grid(dist)
    W = pdf[None, :] * (zg[None, :] >= zL[:, None])
    den = np.trapz(W, zg, axis=1) + 1e-12
    cond = (mu_ln + scale * np.trapz(W * zg[None, :], zg, axis=1) / den) / LN10
    cond = np.where(den < 1e-9, np.log10(L), cond)
    ey = np.trapz(W * np.exp(mu_ln[:, None] + scale * zg[None, :]), zg, axis=1)
    return p, cond, ey


def fit_xgb_aft(X, lo, hi, dist='normal', scale=1.0, seed=0, rounds=600, w=None):
    d = xgb.DMatrix(X, weight=w)
    d.set_float_info('label_lower_bound', np.asarray(lo, float))
    d.set_float_info('label_upper_bound', np.asarray(hi, float))
    p = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik', 'aft_loss_distribution': dist,
         'aft_loss_distribution_scale': scale, 'tree_method': 'hist', 'learning_rate': 0.04, 'max_depth': 6,
         'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'lambda': 1.0, 'seed': seed,
         'nthread': 1}
    return xgb.train(p, d, num_boost_round=rounds)


def fit_xgb_clf(X, y, seed=0, rounds=400, w=None):
    d = xgb.DMatrix(X, label=y, weight=w)
    p = {'objective': 'binary:logistic', 'tree_method': 'hist', 'learning_rate': 0.04, 'max_depth': 5,
         'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.8, 'lambda': 2.0, 'seed': seed,
         'nthread': 1}
    return xgb.train(p, d, num_boost_round=rounds)


def fit_cure_aft(X, lo, hi, listed, dist='normal', scale=1.0, seed=0, n_iter=6, fast=False):
    """Boosted mixture-cure AFT fitted by EM.
    Latent Z indexes the positive-abundance component (Z=1) versus a point mass at zero (Z=0).
    Listed records have Z=1. The latent component is not an observed chemical formation label:
        P(not listed) = (1 - pi) + pi * F(L | present)
    E-step weight w = pi F / (1 - pi + pi F); M-step: AFT on present-weighted rows, classifier on soft labels."""
    listed = np.asarray(listed, bool)
    w = np.where(listed, 1.0, 0.5)
    for it in range(n_iter):
        last = it == n_iter - 1
        f = 0.5 if fast else 1.0                                   # fast = inner-CV model selection
        aft = fit_xgb_aft(X, lo, hi, dist, scale, seed, rounds=int((600 if last else 300) * f), w=w)
        mu = aft.predict(xgb.DMatrix(X), output_margin=True)
        F = aft_cdf((np.log(np.maximum(hi, 1e-9)) - mu) / scale, dist)
        clf = fit_xgb_clf(X, np.where(listed, 1.0, w), seed, rounds=int((400 if last else 250) * f))
        pi = np.clip(clf.predict(xgb.DMatrix(X)), 1e-4, 1 - 1e-4)
        w = np.where(listed, 1.0, pi * F / (1 - pi + pi * F))
    return aft, clf


def fit_tobit(X, lo, hi):
    imp = SimpleImputer(strategy='median').fit(X); sc = StandardScaler().fit(imp.transform(X))
    Z = np.c_[np.ones(len(X)), sc.transform(imp.transform(X))]
    llo = np.where(lo > 0, np.log10(np.maximum(lo, 1e-9)), -np.inf); lhi = np.log10(hi)
    ex = np.isclose(lo, hi)

    def nll(th):
        b, s = th[:-1], np.exp(th[-1]); mu = Z @ b
        out = np.empty(len(mu))
        out[ex] = -stats.norm.logpdf(lhi[ex], mu[ex], s)
        c = ~ex
        out[c] = -np.log(np.maximum(stats.norm.cdf((lhi[c] - mu[c]) / s) - stats.norm.cdf((llo[c] - mu[c]) / s), 1e-12))
        return out.mean() + 1e-3 * (b[1:] ** 2).sum()
    th0 = np.r_[-0.5, np.zeros(Z.shape[1] - 1), 0.0]
    res = minimize(nll, th0, method='L-BFGS-B', options={'maxiter': 400})
    return imp, sc, res.x


def predict_tobit(m, X):
    imp, sc, th = m
    Z = np.c_[np.ones(len(X)), sc.transform(imp.transform(X))]
    return Z @ th[:-1] * LN10, np.exp(th[-1]) * LN10


# =================================================================== features
def make_folds(studies, k, seed):
    rng = np.random.default_rng(seed)
    s = np.array(sorted(studies)); rng.shuffle(s)
    return [set(s[i::k]) for i in range(k)]


def compound_prior(src, target):
    det = src.groupby('identity').detected.mean()
    ex = src[src.ctype == 'exact']
    ab = ex.groupby('identity').lo.apply(lambda s: np.log10(s).mean())
    g = np.log10(ex.lo).mean() if len(ex) else -0.3
    return target.identity.map(det).fillna(0).values, target.identity.map(ab).fillna(g).values


def build_X(tr, te, feats, seed, use_prior=True):
    Xtr, Xte = tr[feats].values.astype(float), te[feats].values.astype(float)
    if not use_prior:
        return Xtr, Xte
    pd_tr, pa_tr = np.zeros(len(tr)), np.zeros(len(tr))
    for f in make_folds(tr.record_id.unique(), 5, seed + 101):           # out-of-fold within training studies
        m = tr.record_id.isin(f).values
        a, b = compound_prior(tr[~m], tr[m]); pd_tr[m], pa_tr[m] = a, b
    a, b = compound_prior(tr, te)
    return np.c_[Xtr, pd_tr, pa_tr], np.c_[Xte, a, b]


# =================================================================== evaluation
def evaluate(te, out):
    t = te.assign(sd=out['score'], ab=out['abund'], ct=out['contrib'])
    rows = []
    for eid, g in t.groupby('exp_id'):
        r = {'exp_id': eid, 'record_id': g.record_id.iloc[0]}
        y = g.detected.values
        if 0 < y.sum() < len(y):
            r['det_auc'] = roc_auc_score(y, g.sd); r['det_ap'] = average_precision_score(y, g.sd)
        ex = g[g.ctype == 'exact']
        if len(ex) >= 5:
            r['rho_abund'] = stats.spearmanr(ex.ab, np.log10(ex.lo)).correlation
            r['mae_log'] = np.mean(np.abs(ex.ab - np.log10(ex.lo)))
            k = min(10, len(ex))
            r['top10'] = len(set(ex.nlargest(k, 'lo').identity) & set(g.nlargest(k, 'ct').identity)) / k
        rows.append(r)
    ev = pd.DataFrame(rows)
    pooled = {}
    if out.get('p_det') is not None:
        pooled['brier_det'] = np.mean((out['p_det'] - te.detected.values) ** 2)
    for tval in THRESHOLDS:
        valid = (te.L_eval.values <= tval) & (te.ctype.values != 'interval')
        lab = ((te.ctype.values == 'exact') & (te.lo.values >= tval))[valid]
        sc = out['exceed'][tval][valid] if out.get('exceed') is not None else out['abund'][valid]
        if 0 < lab.sum() < len(lab):
            pooled[f'exc_auc_{tval:g}'] = roc_auc_score(lab, sc)
            if out.get('exceed') is not None:
                pooled[f'exc_brier_{tval:g}'] = np.mean((sc - lab) ** 2)
    for k, v in pooled.items():
        ev[k] = v
    return ev


# =================================================================== methods
def run_method(m, tr, te, Xtr, Xte, seed, Lte):
    q = tr.tier.values == 'Q'
    posm = tr.ctype.values == 'exact'
    Xpos, ypos = Xtr[posm], np.log10(tr.lo.values[posm])
    usesP = '+P' in m
    sel = np.ones(len(tr), bool) if usesP else q
    if m in CONVENTIONAL:
        p = np.ravel(regressors(seed)[m].fit(Xpos, ypos).predict(Xte))
        return {'score': p, 'abund': p, 'contrib': 10 ** p}
    if m == 'Stack-Conv':                                                  # stacking of the 22 learners
        st = tr.record_id.values[posm]; oof = np.zeros((posm.sum(), len(CONVENTIONAL)))
        for f in make_folds(np.unique(st), 3, seed + 7):
            mm = np.isin(st, list(f))
            for j, b in enumerate(CONVENTIONAL):
                oof[mm, j] = np.ravel(regressors(seed)[b].fit(Xpos[~mm], ypos[~mm]).predict(Xpos[mm]))
        A = np.c_[np.ones(len(oof)), oof]
        coef, _ = nnls(np.c_[A, -np.ones(len(oof))], ypos)                # intercept may be negative
        icpt = coef[0] - coef[-1]; wts = coef[1:-1]
        P = np.column_stack([np.ravel(regressors(seed)[b].fit(Xpos, ypos).predict(Xte)) for b in CONVENTIONAL])
        p = icpt + P @ wts
        return {'score': p, 'abund': p, 'contrib': 10 ** p, 'weights': dict(zip(CONVENTIONAL, wts))}
    if m == 'Substitute-L/2':                                              # LOD/2 substitution
        src = tr[q]; yy = np.where(src.ctype == 'exact', src.lo, np.where(src.ctype == 'left', src.hi / 2, src.lo))
        mdl = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, n_jobs=1, random_state=seed, verbose=-1)
        p = mdl.fit(Xtr[q], np.log10(yy)).predict(Xte)
        return {'score': p, 'abund': p, 'contrib': 10 ** p}
    if m.startswith('Hurdle'):                                             # two-part model
        Lfeat = np.log10(np.where(tr.tier.values == 'Q', tr.L_report.values, FIXED_L))
        clf = fit_xgb_clf(np.c_[Xtr, Lfeat][sel], tr.detected.values[sel], seed)
        pdet = clf.predict(xgb.DMatrix(np.c_[Xte, np.log10(Lte)]))
        exc = {t: clf.predict(xgb.DMatrix(np.c_[Xte, np.full(len(Xte), np.log10(t))])) for t in THRESHOLDS}
        mu = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=1, random_state=seed, verbose=-1).fit(Xpos, ypos).predict(Xte)
        return {'score': pdet, 'abund': mu, 'contrib': pdet * 10 ** mu, 'p_det': pdet, 'exceed': exc}
    if m.startswith('Tobit'):
        mdl = fit_tobit(Xtr[sel], tr.lo.values[sel], tr.hi.values[sel])
        mu, s = predict_tobit(mdl, Xte)
        p, cond, ey = aft_outputs(mu, Lte, 'normal', s)
        exc = {t: aft_outputs(mu, np.full(len(mu), t), 'normal', s)[0] for t in THRESHOLDS}
        return {'score': p, 'abund': cond, 'contrib': ey, 'p_det': p, 'exceed': exc}
    if m.startswith('AFT') or m.startswith('CureAFT'):                     # <AFT|CureAFT>-<dist>-<scale>[+P][-fixedL]
        parts = m.replace('+P', '').split('-')
        dist, scale = parts[1], float(parts[2])
        lo, hi = tr.lo.values.copy(), tr.hi.values.copy()
        if m.endswith('fixedL'):
            hi[tr.ctype.values == 'left'] = FIXED_L
            lo[(tr.ctype.values == 'interval')] = FIXED_L
        listed = tr.detected.values == 1
        if m.startswith('CureAFT'):
            aft, clf = fit_cure_aft(Xtr[sel], lo[sel], hi[sel], listed[sel], dist, scale, seed)
            pi = clf.predict(xgb.DMatrix(Xte))
        else:
            aft = fit_xgb_aft(Xtr[sel], lo[sel], hi[sel], dist, scale, seed)
            pi = np.ones(len(Xte))
        mu = aft.predict(xgb.DMatrix(Xte), output_margin=True)
        p, cond, ey = aft_outputs(mu, Lte, dist, scale)
        exc = {t: pi * aft_outputs(mu, np.full(len(mu), t), dist, scale)[0] for t in THRESHOLDS}
        return {'score': pi * p, 'abund': cond, 'contrib': pi * ey, 'p_det': pi * p, 'exceed': exc, 'pi': pi,
                'mu': mu}
    if m == 'Prior_frequency':
        p, ab = Xte[:, -2], Xte[:, -1]
        return {'score': p, 'abund': ab, 'contrib': p * 10 ** ab, 'p_det': p}
    raise ValueError(m)


def run_fold(grid, feats, test_studies, methods, seed, fold, use_prior=True, save_pred=False, family=None):
    if family is None:                                   # leave-study-out
        te = grid[(grid.tier == 'Q') & grid.record_id.isin(test_studies)].copy()
        tr = grid[~grid.record_id.isin(test_studies)].copy()
    else:                                                # leave-feedstock-family-out (studies of that family removed)
        fam = grid['feed_' + family] == 1
        te = grid[(grid.tier == 'Q') & fam].copy()
        tr = grid[~grid.record_id.isin(te.record_id.unique()) & ~fam].copy()
    Xtr, Xte = build_X(tr, te, feats, seed, use_prior)
    Lte = te.L_eval.values
    res, preds = [], []
    for m in methods:
        t0 = time.time()
        try:
            out = run_method(m, tr, te, Xtr, Xte, seed, Lte)
            ev = evaluate(te, out)
            ev['method'] = m; ev['seed'] = seed; ev['fold'] = fold; ev['fit_s'] = time.time() - t0
            res.append(ev)
            if save_pred:
                pr = te[['exp_id', 'record_id', 'identity', 'ctype', 'lo', 'hi', 'detected', 'L_eval']].copy()
                for k in ['score', 'abund', 'contrib', 'p_det', 'pi', 'mu']:
                    if out.get(k) is not None:
                        pr[k] = out[k]
                for tval in THRESHOLDS:
                    if out.get('exceed') is not None:
                        pr[f'exc_{tval:g}'] = out['exceed'][tval]
                pr['method'] = m; pr['seed'] = seed; preds.append(pr)
        except Exception as e:
            print('FAIL', m, repr(e)[:300], flush=True)
    return (pd.concat(res) if res else pd.DataFrame()), (pd.concat(preds) if preds else pd.DataFrame())


METHOD_SETS = {
    'main': ['Prior_frequency'] + CONVENTIONAL + ['Stack-Conv', 'Substitute-L/2', 'Hurdle', 'Hurdle+P', 'Tobit+P',
             'AFT-normal-1.0', 'AFT-normal-1.0+P', 'AFT-logistic-1.0+P', 'AFT-extreme-1.0+P',
             'CureAFT-normal-1.0', 'CureAFT-normal-1.0+P', 'CureAFT-normal-1.0+P-fixedL', 'AFT-normal-1.0+P-fixedL'],
    'core': ['Prior_frequency'] + CONVENTIONAL + ['Substitute-L/2', 'Hurdle', 'Hurdle+P', 'Tobit+P',
             'AFT-normal-1.0', 'AFT-normal-1.0+P', 'AFT-logistic-1.0+P', 'AFT-extreme-1.0+P', 'AFT-normal-1.0+P-fixedL',
             'CureAFT-normal-1.0', 'CureAFT-normal-1.0+P', 'CureAFT-normal-1.0+P-fixedL'],
    'stack': ['Stack-Conv'],
    'censored': ['Prior_frequency', 'LightGBM', 'Hurdle+P', 'AFT-normal-1.0+P', 'CureAFT-normal-1.0+P'],
    'ablation': ['Prior_frequency', 'SVR', 'LightGBM', 'Hurdle+P', 'AFT-normal-1.0+P', 'CureAFT-normal-1.0+P'],
    'scale': ['AFT-normal-0.6+P', 'AFT-normal-1.5+P', 'CureAFT-normal-0.6+P', 'CureAFT-normal-1.5+P',
              'CureAFT-logistic-1.0+P', 'CureAFT-extreme-1.0+P'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--methods', default='main')
    ap.add_argument('--out', default='benchmark')
    ap.add_argument('--jobs', type=int, default=15)
    ap.add_argument('--no-prior', action='store_true')
    ap.add_argument('--save-pred', action='store_true')
    ap.add_argument('--drop-feats', default='')
    ap.add_argument('--family-out', action='store_true')
    ap.add_argument('--grid', default='grid')
    a = ap.parse_args()
    grid = pd.read_parquet(os.path.join(TAB, f'{a.grid}.parquet'))
    meta = json.load(open(os.path.join(TAB, f'{a.grid}_meta.json')))
    feats = meta['exp_features'] + meta['comp_features']
    if a.drop_feats:
        pref = a.drop_feats.split(',')
        feats = [f for f in feats if not any(f.startswith(p) for p in pref)]
    methods = METHOD_SETS.get(a.methods, a.methods.split(','))
    studies = grid.loc[grid.tier == 'Q', 'record_id'].unique()
    if a.family_out:
        fams = ['woody', 'herbaceous', 'agro_food_residue', 'sludge', 'n_rich_biopolymer', 'plastic', 'oilseed',
                'polymer_composite_ewaste', 'cellulose_textile', 'lignin']
        jobs = [(None, 0, fm) for fm in fams]
    else:
        jobs = [(make_folds(studies, a.k, r)[f], r, f) for r in range(a.repeats) for f in range(a.k)]
    print('methods', len(methods), 'folds', len(jobs), 'features', len(feats) + (0 if a.no_prior else 2), flush=True)
    out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_fold)(grid, feats, ts, methods, r, f, not a.no_prior,
                                                               a.save_pred, f if a.family_out else None)
                                             for ts, r, f in jobs)
    res = pd.concat([o[0] for o in out])
    res.to_csv(os.path.join(TAB, f'{a.out}_results.csv'), index=False)
    if a.save_pred:
        pd.concat([o[1] for o in out]).to_parquet(os.path.join(TAB, f'{a.out}_predictions.parquet'), index=False)
    cols = ['det_auc', 'det_ap', 'top10', 'rho_abund', 'mae_log', 'brier_det', 'exc_auc_1', 'exc_brier_1',
            'exc_auc_5', 'exc_brier_5', 'fit_s']
    cols = [c for c in cols if c in res.columns]
    summ = res.groupby('method')[cols].mean().sort_values('det_auc', ascending=False)
    pd.set_option('display.width', 250)
    print(summ.round(3).to_string())
    summ.to_csv(os.path.join(TAB, f'{a.out}_summary.csv'))


if __name__ == '__main__':
    main()
