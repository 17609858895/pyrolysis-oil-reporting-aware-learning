"""Check model fitting, bounded scores and study grouping on synthetic data."""
from pathlib import Path
import sys
import numpy as np
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from s04_benchmark import aft_cdf, aft_outputs, fit_cure_aft, make_folds

def main():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 3))
    y = np.exp(.3 + .5 * X[:, 0] + rng.normal(scale=.4, size=len(X)))
    listed = y >= 1.0
    lo = np.where(listed, y, 0.)
    hi = np.where(listed, y, 1.)
    aft, clf = fit_cure_aft(X, lo, hi, listed, dist='logistic', scale=1.5, n_iter=1, fast=True)
    matrix = xgb.DMatrix(X)
    mu = aft.predict(matrix, output_margin=True)
    pi = clf.predict(matrix)
    p, conditional, contribution = aft_outputs(mu, np.ones(len(X)), 'logistic', 1.5)
    assert np.isfinite(np.c_[p, conditional, contribution, pi]).all()
    assert ((0 <= p * pi) & (p * pi <= 1)).all()
    assert (contribution >= 0).all()
    p1 = pi * (1 - aft_cdf((np.log(1.) - mu) / 1.5, 'logistic'))
    p5 = pi * (1 - aft_cdf((np.log(5.) - mu) / 1.5, 'logistic'))
    assert (p1 >= p5).all()
    studies = set(range(20))
    folds = make_folds(list(studies), 5, 0)
    assert set.union(*folds) == studies
    assert all(not a & b for i, a in enumerate(folds) for b in folds[i + 1:])
    print('PASS: synthetic EM fit, finite scores, threshold ordering and disjoint study folds')

if __name__ == '__main__':
    main()
