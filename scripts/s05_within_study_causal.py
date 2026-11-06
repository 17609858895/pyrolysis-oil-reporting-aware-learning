"""Step 5 - catalyst effects on product-class composition: within-study contrasts vs pooled estimators.

Design-based estimand: for study s with a non-catalytic reference experiment(s) and catalytic experiment(s)
at the same feedstock family, effect_s,k = mean_cat log((A_k + c)/(A_ref + c)) where A_k is the summed area% of
class k (all structure-resolved records of the experiment) and c = 0.5 area% (pseudo-count for non-listed classes).
Studies are pooled by a DerSimonian-Laird random-effects model (between-study heterogeneity tau^2, I^2).
Compared against: (i) naive pooled regression across studies, (ii) DML partialling-out with study-grouped
cross-fitting (no study fixed effects), (iii) DML on study-demeaned data (within-study DML).
Outputs: tables/causal_within_study.csv, tables/causal_estimator_comparison.csv
"""
import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, 'tables')
PSEUDO = 0.5
CLASS_GROUPS = {
    'Aromatic hydrocarbons': ['Monoaromatics', 'PAHs'],
    'Phenolics': ['Phenols', 'Methoxyphenols'],
    'Acids & esters': ['Carboxylic acids', 'Esters'],
    'Carbonyls & furans': ['Ketones', 'Aldehydes', 'Furans'],
    'Aliphatic hydrocarbons': ['Alkanes', 'Alkenes'],
    'N-compounds': ['N-heterocycles', 'Other N-compounds'],
    'Anhydrosugars': ['Anhydrosugars'],
}


def class_table():
    rec = pd.read_csv(os.path.join(TAB, 'records_clean.csv'))
    ident = pd.read_csv(os.path.join(TAB, 'compound_identity.csv'))
    desc = pd.read_csv(os.path.join(TAB, 'compound_descriptors.csv'))
    exp = pd.read_csv(os.path.join(TAB, 'experiments_tiered.csv'))
    exp = exp[exp.tier == 'Q']
    rec['name_verbatim'] = rec.name_verbatim.astype(str).str.strip()
    rec = rec.merge(ident[['name_verbatim', 'identity']], on='name_verbatim').merge(
        desc[['identity', 'chem_class']], on='identity')
    rec = rec[rec.exp_id.isin(exp.exp_id) & (rec.ra > 0)]
    inv = {c: g for g, cs in CLASS_GROUPS.items() for c in cs}
    rec['group'] = rec.chem_class.map(inv).fillna('Other')
    A = rec.pivot_table(index='exp_id', columns='group', values='ra', aggfunc='sum').fillna(0)
    A = A.reindex(columns=list(CLASS_GROUPS) + ['Other'], fill_value=0)
    return exp.set_index('exp_id').join(A, how='inner'), A.columns.tolist()


def dersimonian_laird(y, v):
    y, v = np.asarray(y, float), np.asarray(v, float)
    w = 1 / v; ybar = (w * y).sum() / w.sum(); Q = (w * (y - ybar) ** 2).sum(); k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    ws = 1 / (v + tau2); mu = (ws * y).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return mu, se, tau2, I2


def within_study_contrasts(E, groups, treat):
    rows = []
    for sid, g in E.groupby('record_id'):
        for fam, gf in g.groupby('feed_family'):
            ref = gf[gf.cat_family == 'none']
            trt = gf[gf.cat_family.isin(treat)]
            if len(ref) == 0 or len(trt) == 0:
                continue
            # match on temperature when several references exist
            for _, t in trt.iterrows():
                r = ref.iloc[(ref.T_mid - t.T_mid).abs().argsort().values[:1]]
                for k in groups:
                    rows.append({'record_id': sid, 'feed_family': fam, 'class': k, 'cat_type': t.catalyst_type,
                                 'T': t.T_mid,
                                 'lr': np.log((t[k] + PSEUDO) / (r[k].values[0] + PSEUDO))})
    d = pd.DataFrame(rows)
    if d.empty:
        return d, pd.DataFrame()
    per = d.groupby(['record_id', 'class']).agg(lr=('lr', 'mean'), n=('lr', 'size'), sd=('lr', 'std')).reset_index()
    # within-study sampling variance: from replicate catalytic conditions when available, else pooled median
    per['v'] = (per.sd.fillna(np.nan) ** 2 / per.n)
    vmed = per.loc[per.v > 0, 'v'].median() if (per.v > 0).any() else 0.1
    per['v'] = per.v.where(per.v > 0, np.maximum(vmed, 0.05))
    pooled = []
    for k, g in per.groupby('class'):
        mu, se, tau2, I2 = dersimonian_laird(g.lr, g.v)
        pooled.append({'class': k, 'k_studies': len(g), 'effect_lr': mu, 'se': se, 'lo95': mu - 1.96 * se,
                       'hi95': mu + 1.96 * se, 'tau2': tau2, 'I2': I2, 'fold_change': np.exp(mu)})
    return per, pd.DataFrame(pooled)


def dml_plr(Y, D, X, groups, demean=False, seed=0):
    """Partialling-out DML with study-grouped cross-fitting. demean=True subtracts study means first."""
    Y, D = Y.copy(), D.copy()
    X = X.copy()
    if demean:
        Y = Y - Y.groupby(groups).transform('mean'); D = D - D.groupby(groups).transform('mean')
        X = X - X.groupby(groups).transform('mean')
    ry, rd = np.zeros(len(Y)), np.zeros(len(Y))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, Y, groups):
        my = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5,
                               verbose=-1, n_jobs=1, random_state=seed).fit(X.iloc[tr], Y.iloc[tr])
        md = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=8, min_child_samples=5,
                               verbose=-1, n_jobs=1, random_state=seed).fit(X.iloc[tr], D.iloc[tr])
        ry[te] = Y.iloc[te] - my.predict(X.iloc[te]); rd[te] = D.iloc[te] - md.predict(X.iloc[te])
    theta = (rd * ry).sum() / (rd * rd).sum()
    psi = (ry - theta * rd) * rd
    # cluster-robust SE by study
    J = (rd * rd).mean()
    cl = pd.Series(psi).groupby(groups.values).sum()
    se = np.sqrt((cl ** 2).sum()) / (len(Y) * J)
    return theta, se


def main():
    E, cols = class_table()
    groups = list(CLASS_GROUPS)
    out_pooled, out_cmp = [], []
    for label, treat in [('zeolite', ['zeolite', 'metal_zeolite', 'zeolite_plus_basic']),
                         ('any catalyst', [c for c in E.cat_family.unique() if c != 'none'])]:
        per, pooled = within_study_contrasts(E, groups, treat)
        if pooled.empty:
            continue
        per['treatment'] = label; pooled['treatment'] = label
        out_pooled.append(pooled)
        per.to_csv(os.path.join(TAB, f'causal_per_study_{label.replace(" ", "_")}.csv'), index=False)
        # estimator comparison on the same experiments
        sub = E[(E.cat_family == 'none') | E.cat_family.isin(treat)].copy()
        D = sub.cat_family.isin(treat).astype(float)
        Xc = pd.get_dummies(sub[['feed_family', 'tech_mode']]).astype(float)
        Xc['T_mid'] = sub.T_mid.fillna(sub.T_mid.median()); Xc['plastic_frac'] = sub.plastic_frac
        Xc['microwave'] = sub.is_microwave; Xc['copyrolysis'] = sub.is_copyrolysis
        for k in groups:
            Y = np.log(sub[k] + PSEUDO)
            # naive pooled OLS-type contrast (difference in means across all studies)
            naive = Y[D == 1].mean() - Y[D == 0].mean()
            th, se = dml_plr(Y, D, Xc, sub.record_id)
            thw, sew = dml_plr(Y, D, Xc, sub.record_id, demean=True)
            pr = pooled[pooled['class'] == k]
            out_cmp.append({'treatment': label, 'class': k, 'naive_diff': naive, 'dml_pooled': th,
                            'dml_pooled_se': se, 'dml_within': thw, 'dml_within_se': sew,
                            'meta_within': pr.effect_lr.values[0] if len(pr) else np.nan,
                            'meta_se': pr.se.values[0] if len(pr) else np.nan,
                            'k_studies': pr.k_studies.values[0] if len(pr) else 0,
                            'n_exp_treated': int(D.sum()), 'n_exp_control': int((1 - D).sum()),
                            'n_studies': sub.record_id.nunique()})
    pd.concat(out_pooled).to_csv(os.path.join(TAB, 'causal_within_study.csv'), index=False)
    cmp = pd.DataFrame(out_cmp)
    cmp.to_csv(os.path.join(TAB, 'causal_estimator_comparison.csv'), index=False)
    pd.set_option('display.width', 200)
    print(pd.concat(out_pooled).round(3).to_string())
    print(cmp.round(3).to_string())


if __name__ == '__main__':
    main()
