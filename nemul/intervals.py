"""Jackknife+ prediction intervals, built on leave-one-group-out folds.

The jackknife+ of Barber, Candes, Ramdas and Tibshirani: for a new point, pair each
leave-one-out model's prediction with that held-out point's residual, then take
quantiles over those pairs. It assumes no error distribution and its guarantee
survives a misspecified model, which matters here because ours is a penalised linear
fit on 11 composition features while the residual carries unobserved process
variance.

**The folds are leave-one-*group*-out, not leave-one-row-out**
(`splits.interval_folds`). With 15 replicate rows, a row-wise jackknife trains on a
point's own twin and reports intervals narrower than the replicate spread can
support. This module refuses folds that hold out a row more than once rather than
averaging the duplicate away.
"""
from typing import NamedTuple

import numpy as np

from . import splits


class Band(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    n_folds: int
    method: str = "jackknife+ on leave-one-group-out folds"


def jackknife_plus(X, y, X_new, folds, estimator=None, alpha: float = 0.1) -> Band:
    """Jackknife+ interval per row of `X_new`, at level `1 - alpha`.

    Each fold contributes its model's prediction at every new point, paired with the
    absolute residual of each row it held out. The lower bound is the `alpha`
    quantile of `prediction - residual` and the upper the `1 - alpha` quantile of
    `prediction + residual`. That is the + form: the quantile runs over leave-one-out
    *models*, not over one model's residuals.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    Xn = np.asarray(X_new, dtype=float)
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")

    seen = np.zeros(len(ya), dtype=int)
    preds, resid = [], []
    for f in folds:
        est = splits.probe_estimator() if estimator is None else estimator
        est.fit(Xa[f.train], ya[f.train])
        held = np.abs(ya[f.test] - est.predict(Xa[f.test]))
        p_new = est.predict(Xn)
        for r in held:            # one (model, residual) pair per held-out row
            preds.append(p_new)
            resid.append(float(r))
        seen[f.test] += 1

    if not (seen == 1).all():
        raise ValueError(
            "jackknife+ needs folds that hold out each row exactly once -- pass "
            "splits.interval_folds(groups). Folds that revisit a row narrow the "
            f"interval silently; rows here were held out {sorted(set(seen.tolist()))} times"
        )

    P = np.asarray(preds)                  # (n_pairs, n_new)
    R = np.asarray(resid)[:, None]
    return Band(
        lower=np.quantile(P - R, alpha, axis=0),
        upper=np.quantile(P + R, 1.0 - alpha, axis=0),
        alpha=alpha,
        n_folds=len(folds),
    )
