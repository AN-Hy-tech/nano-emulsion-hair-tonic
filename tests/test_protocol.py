"""Tests 14 and 15: the protocol guarantees, not the model's quality.

Test 14 is the reproducibility gate -- a deliverable whose numbers move between runs
is not auditable, and the client's report quotes these numbers.

Test 15 is the pre-registered one. It **records** the margin over the mean baseline
and the across-repeat spread; it does not fail when the margin is small or negative.
`workcycle.md` pre-registered that outcome before anything was scored, and PR2 and
PR3 have since measured a grouped R2 near -0.3 and a fit-scale ceiling not above
zero. A test that failed on a small margin would be a test that fails on the
project's honest result.
"""
import numpy as np
import pytest

from nemul import ceiling, config, confound, ladder, loader, splits, target


@pytest.fixture(scope="module")
def dataset():
    df = loader.load_analysis_set()
    X = loader.feature_matrix(df).to_numpy()
    y = loader.target(df).to_numpy()
    g = loader.composition_groups(df).to_numpy()
    return X, y, g


# The cheap rungs. The GP and GBM are exercised in the ladder run below, but the
# determinism check does not need them to prove the point and they dominate runtime.
FAST_RUNGS = ("mean", "ridge", "elastic_net")


def test_run_is_deterministic(dataset):
    """Same seed, same metrics, twice, bit for bit.

    Compared on `tobytes()`, not `np.allclose`: the claim is bit-for-bit, and a
    tolerance here would hide exactly the drift this test exists to catch.
    """
    X, y, g = dataset

    first = ladder.run_ladder(X, y, g, rungs=FAST_RUNGS)
    second = ladder.run_ladder(X, y, g, rungs=FAST_RUNGS)

    assert [r.name for r in first] == [r.name for r in second]
    for a, b in zip(first, second):
        assert a.r2_by_repeat.tobytes() == b.r2_by_repeat.tobytes(), (
            f"{a.name} moved between runs: {a.r2_by_repeat} vs {b.r2_by_repeat}"
        )
        assert a.r2 == b.r2
        assert a.spearman == b.spearman

    # The folds themselves, which everything else rests on.
    f1 = splits.grouped_folds(g, seed=config.SEED)
    f2 = splits.grouped_folds(g, seed=config.SEED)
    assert [(x.repeat, x.fold, x.test.tobytes()) for x in f1] == [
        (x.repeat, x.fold, x.test.tobytes()) for x in f2
    ]


def test_beats_mean_baseline(dataset, capsys):
    """Records the margin over the mean baseline and the across-repeat spread.

    Asserts only what is true regardless of how the model scores:
    - the mean baseline is on the ladder, scored through the same function;
    - every rung is scored on the same folds, on the fit scale;
    - the margin is finite and the spread is reported beside it, because a margin
      quoted without its spread is unreadable on ten repeats.
    """
    X, y, g = dataset
    results = ladder.run_ladder(X, y, g)
    by_name = {r.name: r for r in results}

    assert "mean" in by_name, "the mean baseline is a rung, not a footnote"
    base = by_name["mean"]
    assert base.r2 <= 0.05, (
        "a mean baseline scoring above zero out-of-fold means the folds are leaking: "
        f"got {base.r2:.4f}"
    )

    for r in results:
        assert r.scale == target.FIT_SCALE
        assert r.n_folds == config.N_SPLITS * config.N_REPEATS
        assert len(r.r2_by_repeat) == config.N_REPEATS
        assert np.isfinite(r.margin_over_mean)
        assert np.isfinite(r.spread)

    # Both ceilings, never the flattering one alone (workcycle.md, risks).
    days = ceiling.report_ceiling(y, g, scale="days")
    fit = ceiling.recompute_on_fit_scale(y, g, target.to_fit_scale)
    assert days.scale == "days" and fit.scale == target.FIT_SCALE

    with capsys.disabled():
        print()
        print(f"  ceiling, days scale : {days.point:+.3f}  [{days.lo:+.3f}, {days.hi:+.3f}]")
        print(f"  ceiling, {fit.scale} scale: {fit.point:+.3f}  [{fit.lo:+.3f}, {fit.hi:+.3f}]")
        print(f"  {'rung':<14}{'R2':>9}{'spread':>9}{'margin':>9}{'spearman':>10}")
        for r in results:
            print(
                f"  {r.name:<14}{r.r2:>+9.4f}{r.spread:>9.4f}"
                f"{r.margin_over_mean:>+9.4f}{r.spearman:>+10.4f}"
            )
        best = max(results, key=lambda r: r.r2)
        print(f"  best rung: {best.name} (margin {best.margin_over_mean:+.4f} "
              f"vs spread {base.spread:.4f})")


def test_ladder_rungs_share_one_validation_function(dataset):
    """Every rung is scored by `ladder.validate` on folds it did not choose.

    The ladder's whole claim is comparability. If a rung could pass its own folds or
    its own scorer, the table would compare models to themselves.
    """
    X, y, g = dataset
    folds = splits.grouped_folds(g, seed=config.SEED)
    direct = ladder.validate(X, y, g, ladder.RUNGS["ridge"](), "ridge", folds=folds)
    viaRun = {r.name: r for r in ladder.run_ladder(X, y, g, rungs=("ridge",))}["ridge"]
    assert direct.r2_by_repeat.tobytes() == viaRun.r2_by_repeat.tobytes()


def test_nonlinear_gain_does_not_cross_run_order(dataset, capsys):
    """Records each rung's score on composition-grouped folds and on run-block folds.

    Added on build, 2026-10-09 (PR4), and the reason is the measurement itself. The
    GP and the GBM beat the mean baseline by about +0.33 under composition-grouped
    CV, which contradicted the pre-registered expectation -- so the gain was
    interrogated rather than reported. Re-cut the folds along run order and the whole
    advantage is gone, while the linear rungs barely move. The nonlinear rungs are
    interpolating inside dense run-order blocks, not learning composition physics.

    Asserted: the block folds are group-nested and cover every row exactly once, so
    the two columns are scored on the same rows. The scores themselves are recorded,
    on the test-7 and test-15 precedent -- the data gets the last word.
    """
    X, y, g = dataset
    df = loader.load_analysis_set()
    order = confound.run_order(df).to_numpy()

    grouped = splits.grouped_folds(g, seed=config.SEED)
    blocked = confound.block_folds(order, groups=g)

    seen = np.zeros(len(y), dtype=int)
    for f in blocked:
        seen[f.test] += 1
        assert not (set(g[f.train].tolist()) & set(g[f.test].tolist())), (
            "a composition group spans a run block, so a replicate twin crosses the "
            "split and the block-out score is inflated by the leak it must exclude"
        )
    assert (seen == 1).all(), f"block folds cover rows {sorted(set(seen.tolist()))} times"
    assert len(blocked) == config.N_RUN_BLOCKS

    rows = []
    for name in ladder.LADDER_ORDER:
        a = ladder.validate(X, y, g, ladder.RUNGS[name], name, folds=grouped)
        b = ladder.validate(X, y, g, ladder.RUNGS[name], name, folds=blocked)
        rows.append((name, a.r2, b.r2, a.r2 - b.r2))

    with capsys.disabled():
        print()
        print(f"  {'rung':<13}{'grouped':>10}{'block-out':>11}{'collapse':>10}")
        for name, a, b, d in rows:
            print(f"  {name:<13}{a:>+10.4f}{b:>+11.4f}{d:>+10.4f}")
        print("  the collapse is the confound, measured in R2")
