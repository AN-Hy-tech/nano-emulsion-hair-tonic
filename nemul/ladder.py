"""The model ladder, and the one validation function every rung climbs through.

Five rungs: mean baseline, ridge, elastic net, Gaussian process, small GBM. The point
of a ladder is comparability, so no rung chooses its own folds, its own scorer or its
own target scale -- `validate` owns all three and each rung is only an estimator
handed to it. A number that did not come out of `validate` does not go in the table.

Three things worth knowing before reading the scores:

- **Everything is fitted on `log1p(stability_days)`** (`target.FIT_SCALE`), and R2 is
  reported on that scale. The fit-scale ceiling is the one to read it against, and
  that ceiling is not above zero -- see `planning/project.md`. Both ceilings get
  reported, never the flattering one alone.
- **R2 is pooled within a repeat**, not averaged over folds. PR2 found the per-fold
  mean is swamped by whichever fold holds the 495 d row; `splits.cv_r2_by_repeat`
  carries that reasoning and this module uses the same pooling.
- **Spearman is a secondary metric, not a target.** It is computed on the same pooled
  out-of-fold predictions. It is here because the ordering is where the only
  non-negative signal has shown up, and because "which formulation next" is an
  ordering question. Nothing is tuned on it.

Tuning is nested: `_Tuned` searches its grid in an inner `GroupKFold` over the
training rows only, so the outer fold never scores a hyperparameter chosen with its
own data. The inner folds are grouped too -- an ungrouped inner loop would tune on a
replicate's twin.
"""
import hashlib
from typing import NamedTuple

import numpy as np
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, splits, target


class RungResult(NamedTuple):
    """One rung's score. Everything needed to read it, including its spread."""
    name: str
    r2: float                  # mean over repeats of the pooled within-repeat R2
    r2_by_repeat: np.ndarray
    spread: float              # std across repeats; the ruler for any margin
    margin_over_mean: float    # R2 minus the mean baseline's, on the same folds
    spearman: float            # mean over repeats of pooled out-of-fold Spearman
    scale: str
    n_folds: int
    fold_signature: str
    detail: str = ""


class _Tuned:
    """Nested tuning: an inner grouped grid search on the training rows only.

    Not a sklearn meta-estimator on purpose -- it needs `groups` at fit time and the
    plain estimator protocol has nowhere to put them. `validate` passes them in.
    """

    def __init__(self, base, grid: dict, label: str, scoring="r2", stratified=False):
        self.base, self.grid, self.label = base, grid, label
        self.scoring, self.stratified = scoring, stratified

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError(
                "nested tuning needs the training rows' groups: an ungrouped inner "
                "loop tunes on a replicate's twin"
            )
        n_inner = min(config.N_INNER_SPLITS, len(np.unique(groups)))
        # Stratified for the binarised rung: 19 positives in 109 rows, so an
        # unstratified inner fold can hold none and roc_auc is then undefined.
        maker = StratifiedGroupKFold if self.stratified else GroupKFold
        search = GridSearchCV(
            self.base, self.grid, cv=maker(n_splits=n_inner), scoring=self.scoring
        )
        search.fit(X, y, groups=groups)
        self.best_ = search.best_estimator_
        self.best_params_ = search.best_params_
        return self

    def predict(self, X):
        return self.best_.predict(X)

    def predict_proba(self, X):
        return self.best_.predict_proba(X)


def mean_baseline():
    """Predict the training mean. The rung every other rung has to clear."""
    return DummyRegressor(strategy="mean")


def ridge_tuned():
    return _Tuned(
        make_pipeline(StandardScaler(), Ridge()),
        {"ridge__alpha": list(config.RIDGE_ALPHA_GRID)},
        "ridge",
    )


def elastic_net_tuned():
    return _Tuned(
        make_pipeline(StandardScaler(), ElasticNet(max_iter=50000, tol=1e-5)),
        {
            "elasticnet__alpha": list(config.RIDGE_ALPHA_GRID),
            "elasticnet__l1_ratio": list(config.L1_RATIO_GRID),
        },
        "elastic_net",
    )


def gaussian_process():
    """Matern 3/2 plus a WhiteKernel, so the learned noise is a checkable number.

    The GP earns its rung because `WhiteKernel.noise_level` is an estimate of the
    same quantity `ceiling.pure_error_variance` measures from the replicates. Two
    independent routes to one number is a test of both.
    """
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(0.5)
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=config.SEED),
    )


def gradient_boosting():
    """Deliberately small: depth 2 on 109 rows is a non-linearity check, not a model.

    If this does not beat ridge by more than the across-repeat spread, that is the
    result and it gets reported (workcycle.md, Decisions).
    """
    return make_pipeline(
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=config.SEED
        ),
    )


RUNGS = {
    "mean": mean_baseline,
    "ridge": ridge_tuned,
    "elastic_net": elastic_net_tuned,
    "gp": gaussian_process,
    "gbm": gradient_boosting,
}

LADDER_ORDER = ("mean", "ridge", "elastic_net", "gp", "gbm")


def fold_signature(folds) -> str:
    """A hash of the held-out indices, in order. Two rungs on the same folds match."""
    h = hashlib.sha256()
    for f in folds:
        h.update(np.asarray(f.test, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def _oof_by_repeat(X, y, groups, estimator_factory, folds):
    """Pooled out-of-fold predictions, one full-length vector per repeat.

    Yields `(repeat, y_true, y_pred)`. Raises if a repeat leaves a row untested, for
    the same reason `splits.oof_residuals` does: a partially covered repeat gives a
    score on a silently different row set.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    ga = np.asarray(groups)
    for r in sorted({f.repeat for f in folds}):
        pred = np.full(len(ya), np.nan)
        for f in (f for f in folds if f.repeat == r):
            est = estimator_factory()
            if isinstance(est, _Tuned):
                est.fit(Xa[f.train], ya[f.train], groups=ga[f.train])
            else:
                est.fit(Xa[f.train], ya[f.train])
            pred[f.test] = est.predict(Xa[f.test])
        if np.isnan(pred).any():
            raise ValueError(
                f"repeat {r} left {int(np.isnan(pred).sum())} rows untested; the "
                "score would be computed on a different row set than the others"
            )
        yield r, ya, pred


def validate(X, y, groups, estimator, name: str, folds=None, baseline_r2=None) -> RungResult:
    """The one validation function. Fits on the log1p scale, scores pooled per repeat.

    `estimator` may be an instance or a zero-argument factory; it is rebuilt for each
    fold either way, so no state crosses a fold boundary.
    """
    ga = np.asarray(groups)
    folds = splits.grouped_folds(ga, seed=config.SEED) if folds is None else folds
    z = target.to_fit_scale(np.asarray(y, dtype=float))

    factory = estimator if callable(estimator) else (lambda e=estimator: _clone(e))

    r2s, rhos, detail = [], [], ""
    for _, z_true, z_pred in _oof_by_repeat(X, z, ga, factory, folds):
        r2s.append(float(r2_score(z_true, z_pred)))
        rhos.append(float(stats.spearmanr(z_true, z_pred).statistic))
    r2a = np.asarray(r2s, dtype=float)

    detail = _describe(factory, X, z, ga)
    r2 = float(r2a.mean())
    # Undefined on one repeat, and said so rather than letting numpy warn: the block
    # folds in `confound.block_folds` are a single pass, so there is no across-repeat
    # spread to quote. A margin without its spread is read with that in mind.
    spread = float(r2a.std(ddof=1)) if len(r2a) > 1 else float("nan")
    return RungResult(
        name=name,
        r2=r2,
        r2_by_repeat=r2a,
        spread=spread,
        margin_over_mean=float("nan") if baseline_r2 is None else r2 - baseline_r2,
        spearman=float(np.mean(rhos)),
        scale=target.FIT_SCALE,
        n_folds=len(folds),
        fold_signature=fold_signature(folds),
        detail=detail,
    )


def _clone(estimator):
    from sklearn.base import clone

    if isinstance(estimator, _Tuned):
        return _Tuned(clone(estimator.base), estimator.grid, estimator.label,
                      estimator.scoring, estimator.stratified)
    return clone(estimator)


def _describe(factory, X, z, groups) -> str:
    """One line about the rung, fitted on everything. Not a score: a sanity note.

    For the GP this is the learned noise level, which is the number the replicate
    pure-error variance is the independent check on. For a tuned rung it is the
    chosen hyperparameter, so a grid pinned at an endpoint is visible.
    """
    est = factory()
    try:
        if isinstance(est, _Tuned):
            est.fit(X, z, groups=groups)
            return ", ".join(f"{k.split('__')[-1]}={v:g}" for k, v in est.best_params_.items())
        est.fit(X, z)
        gp = getattr(est, "named_steps", {}).get("gaussianprocessregressor")
        if gp is not None:
            noise = gp.kernel_.k2.noise_level
            return f"learned noise var={noise:.4g} on the {target.FIT_SCALE} scale"
    except Exception as exc:  # a note is never allowed to fail a run
        return f"(no detail: {type(exc).__name__})"
    return ""


def run_ladder(X, y, groups, rungs=None, folds=None) -> list[RungResult]:
    """Every rung, in order, on one set of folds, with margins against the baseline.

    The mean baseline is always scored first and always scored, even if it is not in
    `rungs`, because every margin in the table is measured against it.
    """
    names = list(LADDER_ORDER if rungs is None else rungs)
    ga = np.asarray(groups)
    folds = splits.grouped_folds(ga, seed=config.SEED) if folds is None else folds

    base = validate(X, y, ga, RUNGS["mean"], "mean", folds=folds, baseline_r2=None)
    base = base._replace(margin_over_mean=0.0)

    out = []
    for n in names:
        if n == "mean":
            out.append(base)
            continue
        if n not in RUNGS:
            raise ValueError(f"{n!r} is not a rung; known rungs are {list(RUNGS)}")
        out.append(validate(X, y, ga, RUNGS[n], n, folds=folds, baseline_r2=base.r2))
    return out
