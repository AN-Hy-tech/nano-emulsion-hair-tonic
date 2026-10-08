"""The one place cross-validation folds are made.

Grouping is the loader's (`composition_groups`); this module only decides how those
groups are cut into folds. 109 rows hold 101 distinct compositions, so splitting
rows lets a model score by recognising a replicate it trained on -- every fold here
is cut on the group, never on the row.

Three fold makers, for three jobs:

- `grouped_folds` -- the protocol. Repeated GroupKFold, every reported score.
- `random_folds`  -- the comparator for test 7, and **nothing else**. It leaks on
  purpose. No number from it is ever reported as a model score.
- `interval_folds` -- leave-one-group-out, which PR3's jackknife+ takes its
  residuals from. Leave-one-*row*-out would leak through the replicates.
"""
from typing import NamedTuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


class Fold(NamedTuple):
    """Row positions, not labels: index into the analysis set as it was loaded."""
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def _as_array(groups) -> np.ndarray:
    return np.asarray(groups)


def grouped_folds(groups, n_splits=None, n_repeats=None, seed=None) -> list[Fold]:
    """Repeated GroupKFold on composition. The protocol; every reported score.

    Each repeat reshuffles the group-to-fold assignment under a seed derived from
    `config.SEED`, so the whole set of folds is a deterministic function of that one
    seed (test 14) while the repeats still differ from each other.
    """
    g = _as_array(groups)
    n_splits = config.N_SPLITS if n_splits is None else n_splits
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    seed = config.SEED if seed is None else seed

    folds = []
    for r in range(n_repeats):
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(g)), groups=g)):
            folds.append(Fold(r, k, tr, te))
    return folds


def random_folds(groups, n_splits=None, n_repeats=None, seed=None) -> list[Fold]:
    """Row-wise KFold, ignoring composition. **Leaks on purpose.**

    Test 7's comparator only. Same fold count, same repeats and the same seeds as
    `grouped_folds`, so the difference between them is the leak and not the setup.
    Never use this for a score that reaches the report.
    """
    g = _as_array(groups)
    n_splits = config.N_SPLITS if n_splits is None else n_splits
    n_repeats = config.N_REPEATS if n_repeats is None else n_repeats
    seed = config.SEED if seed is None else seed

    folds = []
    for r in range(n_repeats):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(g)))):
            folds.append(Fold(r, k, tr, te))
    return folds


def interval_folds(groups) -> list[Fold]:
    """Leave-one-group-out: one fold per composition, holding out all of its rows.

    PR3's jackknife+ residuals come from here. Leave-one-row-out would train on a
    held-out row's own replicate and shrink the intervals by exactly the amount the
    replicate spread says we cannot claim.
    """
    g = _as_array(groups)
    cv = LeaveOneGroupOut()
    return [
        Fold(0, k, tr, te)
        for k, (tr, te) in enumerate(cv.split(np.zeros(len(g)), groups=g))
    ]


def probe_estimator():
    """Standardise, then ridge at a fixed alpha. The test-7 probe, not the model.

    The ladder and its nested tuning are PR4's. This exists so the leakage delta is
    measured with one estimator held constant on both sides of the comparison.
    """
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=config.LEAKAGE_PROBE_ALPHA, random_state=None),
    )


def _fit_predict(X, y, fold: Fold, estimator=None):
    Xa, ya = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    est = probe_estimator() if estimator is None else estimator
    est.fit(Xa[fold.train], ya[fold.train])
    return est.predict(Xa[fold.test])


def cv_r2_by_repeat(X, y, folds, estimator=None) -> np.ndarray:
    """Pooled out-of-fold R-squared, one value per repeat.

    **Pooled over a repeat's folds, never a mean of per-fold R-squared.** Measured
    2026-10-08: with ~22 rows per test fold and a target spanning 0-495 d, one fold
    holding the 495 d row scores about -20 and swamps the mean, which made the
    statistic read as noise. Pooling is also the arithmetic the report quotes.

    Scored against the pooled mean of `y`. Pass `y` on the scale being fitted --
    the architecture fits `log1p`, and PR3 owns the labelled transform.
    """
    ya = np.asarray(y, dtype=float)
    out = []
    for r in sorted({f.repeat for f in folds}):
        pred = np.full(len(ya), np.nan)
        for f in (x for x in folds if x.repeat == r):
            pred[f.test] = _fit_predict(X, ya, f, estimator)
        if np.isnan(pred).any():
            raise ValueError(f"repeat {r} left rows untested")
        ss_res = float(((ya - pred) ** 2).sum())
        ss_tot = float(((ya - ya.mean()) ** 2).sum())
        out.append(1.0 - ss_res / ss_tot)
    return np.asarray(out, dtype=float)


def cv_r2(X, y, folds, estimator=None) -> float:
    """Mean pooled out-of-fold R-squared across repeats.

    Negative values are kept, not clipped: under grouped CV on 101 compositions a
    negative R-squared is a real possible outcome, and it is pre-registered as one.
    """
    return float(cv_r2_by_repeat(X, y, folds, estimator).mean())


def oof_residuals(X, y, folds, estimator=None) -> np.ndarray:
    """Out-of-fold residuals, one per row, in row order.

    Requires folds that test every row exactly once -- `interval_folds` does, and
    `grouped_folds` does only within a single repeat. Raises otherwise rather than
    silently averaging a row twice, because PR3's interval width depends on each
    residual being one independent draw.
    """
    ya = np.asarray(y, dtype=float)
    res = np.full(len(ya), np.nan)
    seen = np.zeros(len(ya), dtype=int)
    for f in folds:
        res[f.test] = ya[f.test] - _fit_predict(X, y, f, estimator)
        seen[f.test] += 1
    if not (seen == 1).all():
        raise ValueError(
            "oof_residuals needs folds that test each row exactly once; "
            f"got rows tested {sorted(set(seen.tolist()))} times"
        )
    return res
