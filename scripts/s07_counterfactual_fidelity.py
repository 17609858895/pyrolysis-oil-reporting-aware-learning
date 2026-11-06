"""Step 7 - do model counterfactuals reproduce held-out within-study catalyst contrasts?

For every quantified study s that contains a non-catalytic reference and a catalytic experiment on the same
feedstock family, train each learner on all OTHER studies, then predict the class-level product slate of the
catalytic experiment and of its matched reference (same model, only catalyst descriptors differ).
Predicted effect  = log((sum_c in class k E[Y_c 1(Y_c >= L)] + 0.5) / (... reference ... + 0.5))
Observed effect   = same quantity from the reported tables (candidate compounds only).
Fidelity = Pearson/Spearman correlation and sign agreement over (study, class) pairs.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s04_benchmark import run_method, build_X            # noqa: E402
from s05_within_study_causal import CLASS_GROUPS          # noqa: E402

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
PSEUDO = 0.5
METHODS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['LightGBM', 'Stack-Conv', 'Hurdle+P', 'AFT-normal-1.0+P',
                                                             'CureAFT-normal-1.0+P']
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ''


def contrasts(exp):
    rows = []
    q = exp[exp.tier == 'Q']
    for sid, g in q.groupby('record_id'):
        for fam, gf in g.groupby('feed_family'):
            ref = gf[gf.cat_family == 'none']; trt = gf[gf.cat_family != 'none']
            for _, t in trt.iterrows():
                if len(ref) == 0:
                    continue
                r = ref.iloc[(ref.T_mid - t.T_mid).abs().argsort().values[:1]].iloc[0]
                rows.append({'record_id': sid, 'exp_cat': t.exp_id, 'exp_ref': r.exp_id, 'cat_family': t.cat_family,
                             'feed_family': fam})
    return pd.DataFrame(rows)


def run_study(grid, feats, sid, pairs, methods, inv):
    te = grid[grid.record_id == sid].copy()
    tr = grid[grid.record_id != sid].copy()
    Xtr, Xte = build_X(tr, te, feats, 0, True)
    L = np.full(len(te), 0.01)          # evaluate contributions above a common low limit
    out = []
    te['group'] = te.chem_class.map(inv).fillna('Other')
    obs = te.assign(v=np.where(te.ctype == 'exact', te.lo, 0)).groupby(['exp_id', 'group']).v.sum()
    for m in methods:
        if m == 'RLCure':                               # final pipeline: nested selection inside training studies
            import s10_final as F
            best, _, _ = F.select_cure(tr, feats, 0)
            aft, clf = F.fit_model(tr, Xtr, best, 0, True)
            pi, mu, _ = F.cure_predict(aft, clf, Xte, L, best[0], best[1])
            from s04_benchmark import aft_outputs
            o = {'contrib': pi * aft_outputs(mu, L, best[0], best[1])[2]}
        else:
            o = run_method(m, tr, te, Xtr, Xte, 0, L)
        pred = te.assign(v=o['contrib']).groupby(['exp_id', 'group']).v.sum()
        for _, p in pairs.iterrows():
            for k in CLASS_GROUPS:
                pc, pr = pred.get((p.exp_cat, k), 0), pred.get((p.exp_ref, k), 0)
                oc, orf = obs.get((p.exp_cat, k), 0), obs.get((p.exp_ref, k), 0)
                out.append({'method': m, 'record_id': sid, 'exp_cat': p.exp_cat, 'cat_family': p.cat_family,
                            'class': k, 'pred_lr': np.log((pc + PSEUDO) / (pr + PSEUDO)),
                            'obs_lr': np.log((oc + PSEUDO) / (orf + PSEUDO))})
    return pd.DataFrame(out)


def main():
    grid = pd.read_parquet(os.path.join(TAB, 'grid.parquet'))
    meta = json.load(open(os.path.join(TAB, 'grid_meta.json')))
    feats = meta['exp_features'] + meta['comp_features']
    exp = pd.read_csv(os.path.join(TAB, 'experiments_tiered.csv'))
    pairs = contrasts(exp)
    inv = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
    res = Parallel(n_jobs=len(pairs.record_id.unique()))(
        delayed(run_study)(grid, feats, sid, pairs[pairs.record_id == sid], METHODS, inv)
        for sid in pairs.record_id.unique())
    res = pd.concat(res)
    res.to_csv(os.path.join(TAB, f'counterfactual_fidelity_pairs{SUFFIX}.csv'), index=False)
    summ = []
    for m, g in res.groupby('method'):
        nz = g[g.obs_lr.abs() > 0.2]                    # observed contrast magnitude exceeds the prespecified 0.2 cutoff
        summ.append({'method': m, 'n_pairs_all': len(g), 'pearson_all': stats.pearsonr(g.pred_lr, g.obs_lr)[0],
                     'n_pairs_nonzero': len(nz), 'pearson_nonzero': stats.pearsonr(nz.pred_lr, nz.obs_lr)[0],
                     'sign_agree_nonzero': np.mean(np.sign(nz.pred_lr) == np.sign(nz.obs_lr)),
                     'mae_all': np.mean(np.abs(g.pred_lr - g.obs_lr))})
    summ = pd.DataFrame(summ).sort_values('pearson_nonzero', ascending=False)
    summ.to_csv(os.path.join(TAB, f'counterfactual_fidelity_summary{SUFFIX}.csv'), index=False)
    print(summ.round(3).to_string())


if __name__ == '__main__':
    main()
