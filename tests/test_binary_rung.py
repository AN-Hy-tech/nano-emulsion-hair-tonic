"""Test 17: the binarised rung, the only externally comparable number we have.

Our regression R2 compares to nothing -- nobody has published a regression on
days-to-failure for these systems. A classifier on "stable past 30 days" compares
directly to the composition-only anchor in the literature report (AUC 0.87, n = 496).

That comparison is worth only as much as its honesty, so this test guards the two
ways it could be quietly manufactured:

- **the threshold.** 30 d was declared on 2026-10-08 before anything was scored, and
  is never searched over. A threshold read off the target distribution at scoring
  time would invent the benchmark it is meant to supply.
- **the folds and the features.** The classifier has to run on the identical grouped
  folds and the identical feature matrix as the regression ladder, or the two rungs
  are not on the same ladder at all.
"""
import numpy as np
import pytest

from nemul import binary, config, ladder, loader, splits


@pytest.fixture(scope="module")
def dataset():
    df = loader.load_analysis_set()
    X = loader.feature_matrix(df).to_numpy()
    y = loader.target(df).to_numpy()
    g = loader.composition_groups(df).to_numpy()
    return X, y, g


def test_binary_rung_is_comparable(dataset, capsys):
    X, y, g = dataset

    # -- the threshold is a named constant, not a statistic of the target ----------
    assert config.BINARY_THRESHOLD_DAYS == 30
    assert binary.THRESHOLD is config.BINARY_THRESHOLD_DAYS

    lab = binary.labels(y)
    assert lab.sum() == config.N_STABLE_PAST_THRESHOLD
    assert set(np.unique(lab)) == {0, 1}

    # The guard that catches a data-derived threshold. A quantile rule gives the same
    # labels under any monotone rescaling of the target; a fixed day count does not.
    assert not np.array_equal(lab, binary.labels(y * 2.0)), (
        "labels are invariant to rescaling the target, which is what a quantile "
        "threshold looks like -- the threshold must be a fixed number of days"
    )
    assert not np.array_equal(lab, binary.labels(y / 2.0))

    # -- same folds, same features as the regression ladder -----------------------
    folds = splits.grouped_folds(g, seed=config.SEED)
    res = binary.validate_binary(X, y, g, folds=folds)
    assert res.n_folds == config.N_SPLITS * config.N_REPEATS
    assert res.n_features == len(config.FEATURES)
    assert res.threshold_days == config.BINARY_THRESHOLD_DAYS

    reg = ladder.validate(X, y, g, ladder.RUNGS["ridge"](), "ridge", folds=folds)
    assert res.n_folds == reg.n_folds
    assert res.fold_signature == reg.fold_signature, (
        "the classifier and the regression ladder are not on the same folds"
    )

    assert len(res.auc_by_repeat) == config.N_REPEATS
    assert np.isfinite(res.auc_by_repeat).all()
    assert 0.0 <= res.auc <= 1.0

    with capsys.disabled():
        print()
        print(f"  threshold          : {res.threshold_days} d (declared, never searched)")
        print(f"  positives          : {lab.sum()} of {len(lab)} rows")
        print(f"  pooled OOF AUC     : {res.auc:.4f}  (spread {res.spread:.4f})")
        print(f"  accuracy           : {res.accuracy:.4f}  -- not comparable, classes lopsided")
        print(f"  literature anchor  : AUC {binary.ANCHOR_AUC} on n = {binary.ANCHOR_N}")
        print(f"  straddling groups  : {res.straddling_groups} replicate group(s) span the threshold")


def test_binary_rung_refuses_a_searched_threshold(dataset):
    """The API has no door for a threshold chosen at scoring time."""
    X, y, g = dataset
    with pytest.raises(ValueError, match="declared"):
        binary.labels(y, threshold=float(np.median(y)))
    with pytest.raises(ValueError, match="declared"):
        binary.validate_binary(X, y, g, threshold=12.0)
