"""Jackknife+ interval test (PR3, test 16 of planning/workcycle.md).

The intervals are the part of the deliverable the client will actually act on, so
their coverage is checked on synthetic data where the truth is known -- never
inferred from the 109 rows they will be quoted on.

The synthetic law mimics the real design on purpose: replicate rows share a
composition but draw independent noise, so an interval for a *new* composition has
to carry the replicate spread. Leave-one-row-out would train on a point's own twin
and report intervals narrower than the data can support; that is what test 8 guards
structurally and what this one measures.
"""
import numpy as np
import pytest

from nemul import config, intervals, splits

ALPHA = 0.1
NOMINAL = 1.0 - ALPHA
TOL = 0.05              # moved only with the reason in the same commit
N_GROUPS, REPS, N_NEW = 24, 3, 60
BETA = np.array([2.0, -1.0, 0.5])
NOISE_SD = 1.0


def _synthetic(seed):
    """Grouped synthetic data: REPS rows per composition, independent noise each."""
    rng = np.random.default_rng(seed)
    Xg = rng.normal(size=(N_GROUPS, len(BETA)))
    X = np.repeat(Xg, REPS, axis=0)
    groups = np.repeat(np.arange(N_GROUPS), REPS)
    y = X @ BETA + rng.normal(scale=NOISE_SD, size=len(X))
    Xn = rng.normal(size=(N_NEW, len(BETA)))
    yn = Xn @ BETA + rng.normal(scale=NOISE_SD, size=N_NEW)
    return X, y, groups, Xn, yn


def test_jackknife_coverage():
    """Empirical coverage >= nominal - tolerance on synthetic data."""
    covered, total, widths = 0, 0, []
    for seed in range(config.SEED, config.SEED + 10):
        X, y, groups, Xn, yn = _synthetic(seed)
        band = intervals.jackknife_plus(
            X, y, Xn, splits.interval_folds(groups), alpha=ALPHA
        )
        assert band.alpha == ALPHA
        assert (band.lower <= band.upper).all()
        covered += int(((yn >= band.lower) & (yn <= band.upper)).sum())
        total += len(yn)
        widths.append(float(np.mean(band.upper - band.lower)))

    coverage = covered / total
    print()
    print(f"jackknife+ coverage over {total} synthetic points: {coverage:.3f} "
          f"(nominal {NOMINAL:.2f}, tolerance {TOL:.2f}), mean width "
          f"{np.mean(widths):.2f} against a noise SD of {NOISE_SD}")
    assert coverage >= NOMINAL - TOL


def test_jackknife_needs_grouped_folds():
    """The residuals come from leave-one-group-out, and the helper says so.

    Companion to test 8: there the fold maker was checked, here the interval builder
    refuses folds that test a row more than once -- the silent-narrowing path.
    """
    X, y, groups, Xn, _ = _synthetic(config.SEED)
    with pytest.raises(ValueError):
        intervals.jackknife_plus(
            X, y, Xn, splits.grouped_folds(groups, n_repeats=2), alpha=ALPHA
        )
