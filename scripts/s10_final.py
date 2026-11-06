"""Step 10 - final pipeline (RL-CureBoost) with nested model selection and cross-fitted calibration.

Inside every OUTER training set (leave-study-out, identical folds to s04):
  1. inner 3-fold leave-study-out split of the training studies (P-tier oils of inner-validation studies removed);
  2. fit the cure-AFT for each (distribution, sigma) in CURE_GRID on inner-training studies; score the
     prespecified primary endpoint (mean within-oil detection AUC) on inner-validation quantified oils;
  3. select the best configuration; fit an isotonic map P(listed) -> observed listed rate on the inner
     out-of-fold predictions of that configuration (cross-fitted calibration);
  4. refit the selected cure-AFT on all training studies; predict the outer test studies; ranking uses the raw
     probability, calibration metrics use the isotonic-mapped probability.
Fair comparators get the same treatment: Hurdle+P-cal (inner cross-fitted isotonic) and positive-only
LightGBM / ExtraTrees tuned on the same inner splits for the same primary endpoint.

Usage: python s10_final.py --mode benchmark   (15 outer folds)
       python s10_final.py --mode stability   (EM seed stability on repeat-0 folds)
"""
import os, sys, json, argparse, time, warnings
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s04_benchmark import (fit_cure_aft, fit_xgb_clf, aft_cdf, aft_outputs, build_X, evaluate, make_folds,  # noqa
                           THRESHOLDS, FIXED_L, TAB)

warnings.filterwarnings('ignore')


def bx(tr, te, feats, seed, up=True):
    return build_X(tr, te, feats, seed, up)


CURE_GRID = [('normal', 1.0), ('normal', 1.5), ('normal', 2.0), ('logistic', 1.0), ('logistic', 1.5)]
LGB_GRID = [(15, 10), (15, 40), (63, 10), (63, 40)]            # num_leaves, min_child_samples
ET_GRID = [(1, 0.3), (1, 0.8), (5, 0.3), (5, 0.8)]             # min_samples_leaf, max_features


def within_oil_auc(df, s):
    t = df.assign(s=s); out = []
    for _, g in t.groupby('exp_id'):
        y = g.detected.values
        if 0 < y.sum() < len(y):
            out.append(roc_auc_score(y, g.s))
    return float(np.mean(out)) if out else np.nan


def inner_splits(tr, seed, k=3):
    st = tr.loc[tr.tier == 'Q', 'record_id'].unique()
    for f in make_folds(st, k, seed + 202):
        v = tr.record_id.isin(f).values
        yield tr[~v], tr[v & (tr.tier.values == 'Q')], np.where(v & (tr.tier.values == 'Q'))[0]


def cure_predict(aft, clf, X, L, dist, scale):
    D = xgb.DMatrix(X); mu = aft.predict(D, output_margin=True)
    pi = clf.predict(D) if clf is not None else np.ones(len(X))
    return pi, mu, pi * (1 - aft_cdf((np.log(L) - mu) / scale, dist))


def fit_model(trn, Xa, cfg, seed, cure=True, fast=False):
    if cure:
        return fit_cure_aft(Xa, trn.lo.values, trn.hi.values, trn.detected.values == 1, cfg[0], cfg[1], seed,
                            n_iter=4 if fast else 6, fast=fast)
    from s04_benchmark import fit_xgb_aft
    return fit_xgb_aft(Xa, trn.lo.values, trn.hi.values, cfg[0], cfg[1], seed, rounds=300 if fast else 600), None


def variant_data(tr, variant):
    """Ablation variants of the training data: noP = drop presence-only oils; fixedL = common limit."""
    t = tr.copy()
    if 'noP' in variant:
        t = t[t.tier == 'Q'].copy()
    if 'fixedL' in variant:
        t.loc[t.ctype == 'left', 'hi'] = FIXED_L
        t.loc[t.ctype == 'interval', 'lo'] = FIXED_L
    return t


def select_cure(tr, feats, seed, up=True, cure=True):
    q = tr.tier.values == 'Q'
    oof = {c: np.full(len(tr), np.nan) for c in CURE_GRID}
    for trn, val, idx in inner_splits(tr, seed):
        Xa, Xb = bx(trn, val, feats, seed, up)
        for c in CURE_GRID:
            aft, clf = fit_model(trn, Xa, c, seed, cure, fast=True)
            oof[c][idx] = cure_predict(aft, clf, Xb, val.L_eval.values, c[0], c[1])[2]
    scores = {c: within_oil_auc(tr[q], oof[c][q]) for c in CURE_GRID}
    best = max(scores, key=scores.get)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof[best][q], tr.detected.values[q])
    return best, iso, scores


def hurdle_fit_predict(trn, Xa, Xb, Lb, seed, thresholds=()):
    Lf = np.log10(np.where(trn.tier.values == 'Q', trn.L_report.values, FIXED_L))
    clf = fit_xgb_clf(np.c_[Xa, Lf], trn.detected.values, seed)
    p = clf.predict(xgb.DMatrix(np.c_[Xb, np.log10(Lb)]))
    exc = {t: clf.predict(xgb.DMatrix(np.c_[Xb, np.full(len(Xb), np.log10(t))])) for t in thresholds}
    return p, exc


def positive_model(kind, cfg, seed):
    if kind == 'lgb':
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=cfg[0], min_child_samples=cfg[1],
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                                 verbose=-1)
    return make_pipeline(SimpleImputer(strategy='median'),
                         ExtraTreesRegressor(n_estimators=300, min_samples_leaf=cfg[0], max_features=cfg[1], n_jobs=1,
                                             random_state=seed))


def tuned_positive(kind, tr, te, Xtr, Xte, feats, seed, up=True):
    grid = LGB_GRID if kind == 'lgb' else ET_GRID
    q = tr.tier.values == 'Q'
    oof = {c: np.full(len(tr), np.nan) for c in grid}
    for trn, val, idx in inner_splits(tr, seed):
        Xa, Xb = bx(trn, val, feats, seed, up)
        pm = trn.ctype.values == 'exact'
        for c in grid:
            oof[c][idx] = positive_model(kind, c, seed).fit(Xa[pm], np.log10(trn.lo.values[pm])).predict(Xb)
    scores = {c: within_oil_auc(tr[q], oof[c][q]) for c in grid}
    best = max(scores, key=scores.get)
    pm = tr.ctype.values == 'exact'
    p = positive_model(kind, best, seed).fit(Xtr[pm], np.log10(tr.lo.values[pm])).predict(Xte)
    return {'score': p, 'abund': p, 'contrib': 10 ** p}, best


def run_fold(grid, feats, test_studies, seed, fold, methods, family=None, use_prior=True):
    up = use_prior
    if family is None:
        te = grid[(grid.tier == 'Q') & grid.record_id.isin(test_studies)].copy()
        tr = grid[~grid.record_id.isin(test_studies)].copy()
    else:                                        # leave-feedstock-family-out (studies of that family removed)
        fam = grid['feed_' + family] == 1
        te = grid[(grid.tier == 'Q') & fam].copy()
        tr = grid[~grid.record_id.isin(te.record_id.unique()) & ~fam].copy()
    Xtr, Xte = bx(tr, te, feats, seed, up)
    Lte = te.L_eval.values
    res, preds = [], []
    for m in methods:
        t0 = time.time(); extra = {}
        if m in ('RLCure', 'RLCure-noP', 'RLCure-fixedL', 'RLAFT'):
            trm = variant_data(tr, m)
            Xtr_m, Xte_m = (Xtr, Xte) if m in ('RLCure', 'RLAFT') else bx(trm, te, feats, seed, up)
            cure = m != 'RLAFT'
            best, iso, scores = select_cure(trm, feats, seed, up, cure)
            dist, sc = best
            aft, clf = fit_model(trm, Xtr_m, best, seed, cure)
            pi, mu, p = cure_predict(aft, clf, Xte_m, Lte, dist, sc)
            _, cond, ey = aft_outputs(mu, Lte, dist, sc)
            exc = {t: pi * aft_outputs(mu, np.full(len(mu), t), dist, sc)[0] for t in THRESHOLDS}
            out = {'score': p, 'abund': cond, 'contrib': pi * ey, 'p_det': iso.predict(p), 'exceed': exc,
                   'p_raw': p, 'pi': pi, 'mu': mu}
            extra = {'cfg': f'{dist}-{sc}', 'inner_auc': scores[best]}
        elif m == 'Hurdle+P-cal':
            oof = np.full(len(tr), np.nan); q = tr.tier.values == 'Q'
            for trn, val, idx in inner_splits(tr, seed):
                Xa, Xb = bx(trn, val, feats, seed, up)
                oof[idx] = hurdle_fit_predict(trn, Xa, Xb, val.L_eval.values, seed)[0]
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof[q], tr.detected.values[q])
            p, exc = hurdle_fit_predict(tr, Xtr, Xte, Lte, seed, THRESHOLDS)
            pm = tr.ctype.values == 'exact'
            mu = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, n_jobs=1, random_state=seed,
                                   verbose=-1).fit(Xtr[pm], np.log10(tr.lo.values[pm])).predict(Xte)
            out = {'score': p, 'abund': mu, 'contrib': p * 10 ** mu, 'p_det': iso.predict(p), 'exceed': exc,
                   'p_raw': p}
        elif m in ('LightGBM-tuned', 'ExtraTrees-tuned'):
            out, best = tuned_positive('lgb' if m.startswith('Light') else 'et', tr, te, Xtr, Xte, feats, seed, up)
            extra = {'cfg': str(best)}
        else:
            raise ValueError(m)
        ev = evaluate(te, out)
        if out.get('p_raw') is not None:
            ev['brier_raw'] = np.mean((out['p_raw'] - te.detected.values) ** 2)
        ev['method'] = m; ev['seed'] = seed; ev['fold'] = fold; ev['fit_s'] = time.time() - t0
        for k, v in extra.items():
            ev[k] = v
        res.append(ev)
        pr = te[['exp_id', 'record_id', 'identity', 'ctype', 'lo', 'hi', 'detected', 'L_eval']].copy()
        for k in ['score', 'abund', 'contrib', 'p_det', 'p_raw', 'pi', 'mu']:
            if out.get(k) is not None:
                pr[k] = out[k]
        if out.get('exceed') is not None:
            for t in THRESHOLDS:
                pr[f'exc_{t:g}'] = out['exceed'][t]
        pr['method'] = m; pr['seed'] = seed; preds.append(pr)
    return pd.concat(res), pd.concat(preds)


def stability_fold(grid, feats, test_studies, fold, cfg, seeds=(0, 1, 2, 3, 4)):
    te = grid[(grid.tier == 'Q') & grid.record_id.isin(test_studies)].copy()
    tr = grid[~grid.record_id.isin(test_studies)].copy()
    Xtr, Xte = build_X(tr, te, feats, 0)
    P, rows = {}, []
    for s in seeds:
        aft, clf = fit_cure_aft(Xtr, tr.lo.values, tr.hi.values, tr.detected.values == 1, cfg[0], cfg[1], s)
        P[s] = cure_predict(aft, clf, Xte, te.L_eval.values, cfg[0], cfg[1])[2]
        rows.append({'fold': fold, 'seed': s, 'det_auc': within_oil_auc(te, P[s])})
    cors = [stats.spearmanr(P[a], P[b]).correlation for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    df = pd.DataFrame(rows); df['pairwise_spearman_mean'] = np.mean(cors); df['pairwise_spearman_min'] = np.min(cors)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='benchmark')
    ap.add_argument('--jobs', type=int, default=15)
    ap.add_argument('--methods', default='RLCure,Hurdle+P-cal,LightGBM-tuned,ExtraTrees-tuned')
    ap.add_argument('--cfg', default='')
    ap.add_argument('--grid', default='grid')
    ap.add_argument('--out', default='final')
    ap.add_argument('--no-prior', action='store_true')
    ap.add_argument('--drop-feats', default='')
    ap.add_argument('--family-out', action='store_true')
    a = ap.parse_args()
    grid = pd.read_parquet(os.path.join(TAB, f'{a.grid}.parquet'))
    meta = json.load(open(os.path.join(TAB, f'{a.grid}_meta.json')))
    feats = meta['exp_features'] + meta['comp_features']
    if a.drop_feats == 'structure':
        feats = meta['exp_features']
    elif a.drop_feats:
        feats = [f for f in feats if not any(f.startswith(p) for p in a.drop_feats.split(','))]
    studies = grid.loc[grid.tier == 'Q', 'record_id'].unique()
    if a.mode == 'benchmark':
        if a.family_out:
            fams = ['woody', 'herbaceous', 'agro_food_residue', 'sludge', 'n_rich_biopolymer', 'plastic', 'oilseed',
                    'polymer_composite_ewaste', 'cellulose_textile', 'lignin']
            jobs = [(None, 0, fm) for fm in fams]
        else:
            jobs = [(make_folds(studies, 5, r)[f], r, f) for r in range(3) for f in range(5)]
        print('features', len(feats), 'prior', not a.no_prior, 'folds', len(jobs), flush=True)
        out = Parallel(n_jobs=a.jobs, verbose=5)(delayed(run_fold)(grid, feats, ts, r, f, a.methods.split(','),
                                                                   f if a.family_out else None, not a.no_prior)
                                                 for ts, r, f in jobs)
        res = pd.concat([o[0] for o in out]); res.to_csv(os.path.join(TAB, f'{a.out}_results.csv'), index=False)
        pd.concat([o[1] for o in out]).to_parquet(os.path.join(TAB, f'{a.out}_predictions.parquet'), index=False)
        cols = [c for c in ['det_auc', 'det_ap', 'top10', 'rho_abund', 'mae_log', 'brier_det', 'brier_raw',
                            'exc_auc_1', 'exc_brier_1', 'exc_auc_5', 'exc_brier_5', 'fit_s'] if c in res]
        summ = res.groupby('method')[cols].mean().sort_values('det_auc', ascending=False)
        summ.to_csv(os.path.join(TAB, f'{a.out}_summary.csv'))
        pd.set_option('display.width', 250); print(summ.round(3).to_string())
        if 'cfg' in res:
            print(res.drop_duplicates(['method', 'seed', 'fold']).groupby('method').cfg.value_counts().to_string())
    else:
        dist, sc = a.cfg.split('-'); cfg = (dist, float(sc))
        jobs = [(make_folds(studies, 5, 0)[f], f) for f in range(5)]
        out = Parallel(n_jobs=a.jobs)(delayed(stability_fold)(grid, feats, ts, f, cfg) for ts, f in jobs)
        res = pd.concat(out); res.to_csv(os.path.join(TAB, 'final_em_stability.csv'), index=False)
        print(res.groupby('fold').agg(auc_mean=('det_auc', 'mean'), auc_sd=('det_auc', 'std'),
                                      rho_mean=('pairwise_spearman_mean', 'first'),
                                      rho_min=('pairwise_spearman_min', 'first')).round(4).to_string())


if __name__ == '__main__':
    main()
