"""The replicate noise ceiling: how much of the variance composition could explain.

15 of our 109 rows share a composition with another row while reporting a different
stability. That spread cannot be explained by any model of composition, so it bounds
every score we report. The bound is `1 - sigma2_pure / sigma2_total`.

Two things this module is careful about, both from `workcycle.md`:

- **The variance ratio is the reported figure.** The sum-of-squares variant is the
  lack-of-fit denominator; it sits near 1 here only because almost nothing is
  replicated, so it reads as good news when it is really an absence of data. It
  lives in its own function and the report path never calls it.
- **The point estimate is never quoted alone, and never as a plus-minus.** It rests
  on few replicate df, so its chi-square interval is wide and left-skewed. The
  returned object offers an interval and deliberately offers no symmetric summary.
"""
from typing import NamedTuple

import numpy as np
from scipy import stats


class PureError(NamedTuple):
    variance: float
    df: int
    n_groups: int


class Ceiling(NamedTuple):
    """A ceiling with its interval. No `se`, no `plus_minus`: see the module docstring."""
    point: float
    lo: float
    hi: float
    df: int
    scale: str
    interval_method: str


def _as_arrays(y, groups):
    ya = np.asarray(y, dtype=float)
    ga = np.asarray(groups)
    if len(ya) != len(ga):
        raise ValueError("y and groups must be the same length")
    return ya, ga


def pure_error_variance(y, groups) -> PureError:
    """Pooled within-group variance and its df: the direct noise estimate.

    Pooled as the summed squared deviations from each group mean over `n - n_groups`,
    so a group of three contributes 2 df. Groups of one contribute nothing and are
    not an error -- most of our compositions are singletons.
    """
    ya, ga = _as_arrays(y, groups)
    ss, df, replicated = 0.0, 0, 0
    for g in np.unique(ga):
        vals = ya[ga == g]
        if len(vals) < 2:
            continue
        ss += float(((vals - vals.mean()) ** 2).sum())
        df += len(vals) - 1
        replicated += 1
    if df == 0:
        raise ValueError(
            "no replicate group has two or more rows, so there is no pure-error "
            "estimate; the ceiling is not computable on this set"
        )
    return PureError(ss / df, df, replicated)


def noise_ceiling(y, groups, scale: str, conf: float = 0.95) -> Ceiling:
    """`1 - sigma2_pure / sigma2_total`, with a chi-square interval on the pure df.

    The interval comes from the chi-square interval on `sigma2_pure` mapped through
    the formula: more noise means a lower ceiling, so the bounds swap. It is
    asymmetric about the point, which is the honest shape on few df.

    **`scale` is required and is not inferred.** The ceiling is scale-dependent, and
    quoting a days-scale ceiling against a log1p-scale score is the mistake test 13
    exists to prevent -- a default of `days` would have labelled a log1p ceiling
    wrongly and silently, which is how it failed when first written. Use
    `recompute_on_fit_scale` for the fit scale and the label comes with it.
    """
    if not scale:
        raise ValueError(
            "pass scale explicitly: 'days' for untransformed stability_days, or call "
            "recompute_on_fit_scale for the scale the model is fitted on"
        )
    ya, ga = _as_arrays(y, groups)
    pure = pure_error_variance(ya, ga)
    total = float(np.var(ya, ddof=1))
    if total <= 0:
        raise ValueError("the target has no variance; there is nothing to bound")

    alpha = 1.0 - conf
    # chi-square interval on the pure-error variance, then mapped through 1 - s2/tot
    lo_var = pure.df * pure.variance / stats.chi2.ppf(1 - alpha / 2, pure.df)
    hi_var = pure.df * pure.variance / stats.chi2.ppf(alpha / 2, pure.df)
    return Ceiling(
        point=1.0 - pure.variance / total,
        lo=1.0 - hi_var / total,   # most noise -> lowest ceiling
        hi=1.0 - lo_var / total,
        df=pure.df,
        scale=scale,
        interval_method=f"chi-square on {pure.df} replicate df, {conf:.0%}",
    )


def report_ceiling(y, groups, scale: str, conf: float = 0.95) -> Ceiling:
    """What the report path calls. The variance ratio, always."""
    return noise_ceiling(y, groups, scale=scale, conf=conf)


def recompute_on_fit_scale(y, groups, transform, conf: float = 0.95) -> Ceiling:
    """The ceiling on the scale the model is fitted on. Test 13's path.

    Takes the transform rather than its output, so the scale label and the numbers
    cannot disagree.
    """
    from . import target as _target

    return noise_ceiling(transform(y), groups, scale=_target.FIT_SCALE, conf=conf)


def lack_of_fit_ss_ratio(y, groups) -> float:
    """The SS variant, `1 - SS_pure / SS_total`. **Not for the report.**

    This is the lack-of-fit denominator. It is near 1 here only because almost
    nothing is replicated -- 15 rows of 109 -- so it flatters the model by counting
    an absence of replication as explained variance. Kept so the distinction is a
    function somebody has to choose, not a formula somebody might misremember.
    """
    ya, ga = _as_arrays(y, groups)
    ss_pure = 0.0
    for g in np.unique(ga):
        vals = ya[ga == g]
        if len(vals) > 1:
            ss_pure += float(((vals - vals.mean()) ** 2).sum())
    ss_total = float(((ya - ya.mean()) ** 2).sum())
    return 1.0 - ss_pure / ss_total
