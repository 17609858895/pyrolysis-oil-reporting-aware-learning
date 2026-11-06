"""Paired study-level comparisons (Wilcoxon signed-rank, Holm-adjusted) of the proposed cure model vs competitors.
Unit = held-out study (metrics averaged over its oils and over the 3 repeats), so tests respect study clustering."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TAB = os.path.join(ROOT, 'tables')
files = [f for f in ['benchmark_results.csv', 'stack_results.csv', 'final_results.csv', 'scale_results.csv'] if os.path.exists(os.path.join(TAB, f))]
r = pd.concat([pd.read_csv(os.path.join(TAB, f)) for f in files])
REF = sys.argv[1] if len(sys.argv) > 1 else 'CureAFT-normal-1.0+P'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'paired_tests.csv'
metrics = ['det_auc', 'det_ap', 'top10', 'rho_abund']
st = r.groupby(['method', 'record_id'])[metrics].mean()
comps = ['LightGBM-tuned', 'ExtraTrees-tuned', 'Hurdle+P-cal', 'CureAFT-normal-1.0+P', 'Prior_frequency', 'SVR', 'ExtraTrees', 'AdaBoost', 'LightGBM', 'XGBoost', 'CatBoost', 'Stack-Conv', 'Substitute-L/2',
         'Hurdle+P', 'Tobit+P', 'AFT-normal-1.0+P', 'CureAFT-normal-1.0', 'CureAFT-normal-1.0+P-fixedL', 'AFT-normal-1.0+P-fixedL']
rows = []
for c in comps:
    if c == REF or c not in st.index.get_level_values(0):
        continue
    for m in metrics:
        a = st.loc[REF][m]; b = st.loc[c][m]; d = (a - b).dropna()
        if len(d) < 5:
            continue
        p = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        rows.append({'comparator': c, 'metric': m, 'n_studies': len(d), 'mean_diff': d.mean(), 'median_diff': d.median(),
                     'win_rate': (d > 0).mean(), 'p': p})
T = pd.DataFrame(rows)
T['p_holm'] = np.nan
for m, g in T.groupby('metric'):
    order = g.p.sort_values().index; k = len(order)
    adj = np.maximum.accumulate([min(1, g.p[i] * (k - j)) for j, i in enumerate(order)])
    T.loc[order, 'p_holm'] = adj
T.to_csv(os.path.join(TAB, OUT), index=False)
pd.set_option('display.width', 200)
print(T[T.metric.isin(['det_auc', 'top10'])].round(4).to_string())
