"""Prespecified primary comparisons for the final model (RLCure).
Study level (n = 44 held-out studies; Wilcoxon signed-rank, Holm within each metric over 5 comparators):
detection AUC, detection AP, top-10 recall, abundance rho, log10 MAE.
Fold level (n = 15 outer folds) vs the calibrated hurdle: Brier, exceedance AUC/Brier at 1 % and 5 %."""
import os
import numpy as np, pandas as pd
from scipy import stats
TAB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tables')
r = pd.concat([pd.read_csv(os.path.join(TAB, f)) for f in ['benchmark_results.csv', 'stack_results.csv', 'final_results.csv']])
REF = 'RLCure'
COMPS = ['LightGBM-tuned', 'ExtraTrees-tuned', 'Stack-Conv', 'Hurdle+P-cal', 'Prior_frequency']
MET = {'det_auc': 1, 'det_ap': 1, 'top10': 1, 'rho_abund': 1, 'mae_log': -1}
st = r.groupby(['method', 'record_id'])[list(MET)].mean()
rows = []
for m, sgn in MET.items():
    sub = []
    for c in COMPS:
        d = ((st.loc[REF][m] - st.loc[c][m]) * sgn).dropna()
        sub.append({'level': 'study', 'metric': m, 'comparator': c, 'n': len(d), 'ref_mean': st.loc[REF][m].mean(),
                    'comp_mean': st.loc[c][m].mean(), 'improvement': d.mean(), 'win_rate': (d > 0).mean(),
                    'p': stats.wilcoxon(d).pvalue})
    sub = pd.DataFrame(sub).sort_values('p').reset_index(drop=True)
    sub['p_holm'] = np.maximum.accumulate([min(1, p * (len(sub) - i)) for i, p in enumerate(sub.p)])
    rows.append(sub)
fl = r.groupby(['method', 'seed', 'fold']).first()
for m, sgn in {'brier_det': -1, 'exc_auc_1': 1, 'exc_auc_5': 1, 'exc_brier_1': -1, 'exc_brier_5': -1}.items():
    d = ((fl.loc[REF][m] - fl.loc['Hurdle+P-cal'][m]) * sgn).dropna()
    rows.append(pd.DataFrame([{'level': 'fold', 'metric': m, 'comparator': 'Hurdle+P-cal', 'n': len(d),
                               'ref_mean': fl.loc[REF][m].mean(), 'comp_mean': fl.loc['Hurdle+P-cal'][m].mean(),
                               'improvement': d.mean(), 'win_rate': (d > 0).mean(),
                               'p': stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0}]))
T = pd.concat(rows, ignore_index=True)
T.to_csv(os.path.join(TAB, 'final_primary_tests.csv'), index=False)
pd.set_option('display.width', 220)
print(T.round(4).to_string())
