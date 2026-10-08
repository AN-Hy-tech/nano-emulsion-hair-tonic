"""Leakage tests (PR2, tests 6-8 of planning/workcycle.md).

The single most important guard in the project. 109 rows hold only 101 distinct
compositions, so a row-wise split puts a replicate in train and its twin in test and
the model scores by recognising it. These three tests say: no composition spans a
fold, the leak is real and we measured it ourselves, and the interval path groups too.

A red test here is diagnosed in `nemul/splits.py` or the loader, never silenced.
"""
import numpy as np
import pytest

from nemul import config, loader, splits


@pytest.fixture(scope="module")
def data():
    df = loader.load_analysis_set()
    return loader.feature_matrix(df), loader.target(df), loader.composition_groups(df)


def test_no_group_spans_a_fold(data):
    """In every fold of every repeat, no composition appears in both train and test."""
    _, _, groups = data
    g = groups.to_numpy()
    folds = splits.grouped_folds(groups)

    assert len(folds) == config.N_SPLITS * config.N_REPEATS
    for f in folds:
        assert set(g[f.train]).isdisjoint(set(g[f.test])), (
            f"composition spans repeat {f.repeat} fold {f.fold}"
        )
        assert set(f.train).isdisjoint(set(f.test))

    # every row is tested exactly once per repeat, and nothing is dropped
    for r in range(config.N_REPEATS):
        tested = np.concatenate([f.test for f in folds if f.repeat == r])
        assert sorted(tested) == list(range(config.N_ANALYSIS_ROWS))


def test_random_split_leaks_and_the_delta_is_recorded(data):
    """Test 7, as the measurement forced it: the structural leak is asserted, the
    score delta is recorded and not asserted on.

    Planned as `test_random_split_scores_higher` -- a row-wise split was expected to
    out-score the grouped one, giving us our own leakage figure for a regime with no
    published one. **Measured 2026-10-08 (PR2): it does not.** The score delta is
    statistically zero (|t| < 0.3 at 10, 50 and 100 repeats, either sign), and on the
    15 replicate rows alone, where the leak must live if anywhere, a memorising 1-NN
    probe scores *worse* under the leaky split in 19 of 20 repeats. The reason is the
    noise floor already in `project.md`: within-replicate spreads run to 3.1 on
    `log1p` against a total SD of 1.1, so a twin seen in training is an anti-signal,
    not a shortcut. Full numbers in `pr-log.md`, PR2.

    So the sign assertion would be asserting a falsehood. What stays is the guard with
    teeth -- the leak is structurally present and grouping is what removes it -- on the
    precedent test 15 already sets: record the margin, do not fail on it. This does
    **not** relax the protocol; a GP or GBM rung in PR4 sees the same folds.
    """
    X, y, groups = data
    g = groups.to_numpy()

    # the structural leak, which is a fact about the folds and not about any score
    random = splits.random_folds(groups)
    spanning = [f for f in random if not set(g[f.train]).isdisjoint(set(g[f.test]))]
    assert spanning, "row-wise folds no longer span a composition; is the data still replicated?"
    assert all(set(g[f.train]).isdisjoint(set(g[f.test])) for f in splits.grouped_folds(groups))

    # and the delta, on the fit scale, recorded rather than asserted
    y_fit = np.log1p(y.to_numpy())          # PR3 owns the labelled transform
    grouped_r2 = splits.cv_r2_by_repeat(X, y_fit, splits.grouped_folds(groups))
    random_r2 = splits.cv_r2_by_repeat(X, y_fit, random)
    delta = random_r2 - grouped_r2
    print()
    print(f"leakage delta (random - grouped) pooled R2 over {len(delta)} repeats: "
          f"mean {delta.mean():+.4f}, SE {delta.std(ddof=1) / np.sqrt(len(delta)):.4f}, "
          f"positive in {(delta > 0).sum()}/{len(delta)} "
          f"[grouped {grouped_r2.mean():+.4f}, random {random_r2.mean():+.4f}]")
    assert len(delta) == config.N_REPEATS
    assert np.isfinite(delta).all()


def test_intervals_use_grouped_folds(data):
    """Jackknife+ residuals come from leave-one-*group*-out, not leave-one-row-out."""
    X, y, groups = data
    g = groups.to_numpy()
    folds = splits.interval_folds(groups)

    # one fold per composition, not one per row -- that difference is the whole test
    assert len(folds) == config.N_DISTINCT_COMPOSITIONS
    assert len(folds) != config.N_ANALYSIS_ROWS

    held_out = 0
    for f in folds:
        gids = set(g[f.test])
        assert len(gids) == 1, "a fold held out more than one composition"
        gid = gids.pop()
        assert gid not in set(g[f.train]), "the held-out composition stayed in train"
        assert sorted(f.test) == sorted(np.flatnonzero(g == gid)), (
            "a replicate row of the held-out composition was left in train"
        )
        if len(f.test) > 1:
            held_out += len(f.test)
    assert held_out == config.N_REPLICATE_ROWS

    # and the residuals the intervals are built from cover every row exactly once
    res = splits.oof_residuals(X, y, folds)
    assert len(res) == config.N_ANALYSIS_ROWS
    assert np.isfinite(res).all()
