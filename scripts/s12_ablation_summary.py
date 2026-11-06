"""Step 12 - summarise ablations of the final pipeline.

Paired ablations (same 44 held-out studies as the main benchmark):
  component: -presence-only tier (RLCure-noP), -study-specific limits (RLCure-fixedL), -cure fraction (RLAFT)
  features : -compound prior (abl_noprior), -structure descriptors (abl_nostruct)
  P-tier interval bounds: Q25/Q75 (abl_pq25), Q50/Q50 (abl_pq50) vs Q10/Q90 (main)
Study-level Wilcoxon on (full model - ablation); fold-level Wilcoxon for exceedance/calibration metrics.
Unpaired sensitivities (different candidate set or oils): >=3-study candidates (abl_min3), broad profiles
(abl_broad), leave-feedstock-family-out (final_familyout): methods compared within each setting.
Outputs: tables/ablation_summary.csv and tables/ablation_settings.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')


def load(name):
    f = os.path.join(TAB, f'{name}_results.csv')
    return pd.read_csv(f).drop_duplicates(['method', 'exp_id', 'seed', 'fold']) if os.path.exists(f) else None


def study_diff(full, abl, metric):
    a = full.groupby('record_id')[metric].mean(); b = abl.groupby('record_id')[metric].mean()
    d = (a - b).dropna()
    return d


def fold_diff(full, abl, metric):
    a = full.groupby(['seed', 'fold'])[metric].first(); b = abl.groupby(['seed', 'fold'])[metric].first()
    return (a - b).dropna()


def boot_ci(d, n=4000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = rng.choice(d, (n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    main_r = load('final'); comp = load('abl_components')
    full = main_r[main_r.method == 'RLCure']
    paired = []
    specs = []
    if comp is not None:
        specs += [('− presence-only studies', comp[comp.method == 'RLCure-noP']),
                  ('− study-specific limits', comp[comp.method == 'RLCure-fixedL']),
                  ('− cure fraction (AFT)', comp[comp.method == 'RLAFT'])]
    for nm, lab in [('abl_noprior', '− compound prior'), ('abl_nostruct', '− structure descriptors'),
                    ('abl_pq25', 'P bounds Q25/Q75'), ('abl_pq50', 'P bounds Q50/Q50')]:
        r = load(nm)
        if r is not None:
            specs.append((lab, r[r.method == 'RLCure']))
    for lab, abl in specs:
        row = {'ablation': lab}
        for m in ['det_auc', 'det_ap', 'top10', 'mae_log']:
            d = study_diff(full, abl, m) * (-1 if m == 'mae_log' else 1)
            lo, hi = boot_ci(d)
            row.update({f'{m}_full': full[m].mean(), f'{m}_abl': abl[m].mean(), f'{m}_gain': d.mean(),
                        f'{m}_ci_lo': lo, f'{m}_ci_hi': hi, f'{m}_p': stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0})
        for m in ['exc_auc_5', 'exc_brier_5', 'brier_det']:
            d = fold_diff(full, abl, m) * (-1 if 'brier' in m else 1)
            lo, hi = boot_ci(d)
            row.update({f'{m}_ci_lo': lo, f'{m}_ci_hi': hi})
            row.update({f'{m}_full': full.groupby(['seed', 'fold'])[m].first().mean(),
                        f'{m}_abl': abl.groupby(['seed', 'fold'])[m].first().mean(), f'{m}_gain': d.mean(),
                        f'{m}_p': stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0})
        paired.append(row)
    P = pd.DataFrame(paired)
    P.to_csv(os.path.join(TAB, 'ablation_summary.csv'), index=False)
    pd.set_option('display.width', 250)
    if len(P):
        print(P[['ablation', 'det_auc_abl', 'det_auc_gain', 'det_auc_p', 'top10_gain', 'top10_p', 'exc_auc_5_gain',
                 'exc_auc_5_p', 'brier_det_gain', 'brier_det_p']].round(4).to_string())

    sett = []
    for nm, lab in [('final', 'Main (171 candidates, 44 studies)'), ('abl_min3', '≥3-study candidates (261)'),
                    ('abl_broad', 'Broad profiles only'), ('final_familyout', 'Leave-feedstock-family-out'),
                    ('abl_noprior', '− compound prior'), ('abl_nostruct', '− structure descriptors')]:
        r = load(nm)
        if r is None:
            continue
        for m, g in r.groupby('method'):
            if m not in ('RLCure', 'Hurdle+P-cal', 'LightGBM-tuned'):
                continue
            sett.append({'setting': lab, 'method': m, 'det_auc': g.det_auc.mean(), 'det_ap': g.det_ap.mean(),
                         'top10': g.top10.mean(), 'exc_auc_5': g.groupby(['seed', 'fold']).exc_auc_5.first().mean()})
    S = pd.DataFrame(sett)
    S.to_csv(os.path.join(TAB, 'ablation_settings.csv'), index=False)
    if len(S):
        print(S.pivot_table(index='setting', columns='method', values=['det_auc', 'top10']).round(3).to_string())



if __name__ == '__main__':
    main()
