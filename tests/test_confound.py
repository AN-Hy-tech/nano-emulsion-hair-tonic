"""The confound triad, checked against a confound we built on purpose.

Not one of the eighteen. `test_confound_triad_all_emitted` was cut on the 2026-10-08
audit because it asserted that three things were emitted, which is a Done-gate item,
not behaviour. This test is a different thing and the reason it was added is in
`workcycle.md`: `confound.py` does real arithmetic, and the one claim it makes --
"run order as a covariate demotes lecithin" -- is checkable on synthetic data where
we know run order is the whole story and composition is a passenger.

If this test cannot tell a planted confound from a real effect, the triad cannot
either, and the limitations block would be resting on nothing.
"""
import numpy as np
import pytest

from nemul import config, confound, loader, target

N = 60
LECITHIN = "lecithin_pct"


def _planted(seed=0):
    """y is a function of run order alone; lecithin merely tracks run order.

    The honest answer on this data is "lecithin carries no independent signal". A
    naive composition-only fit cannot know that and will rank lecithin first.
    """
    rng = np.random.default_rng(seed)
    order = np.arange(N, dtype=float)
    scaled = order / N
    lec = scaled + rng.normal(0, 0.05, N)          # confounded with run order
    others = rng.normal(0, 1, (N, len(config.FEATURES) - 1))
    names = [LECITHIN] + [f for f in config.FEATURES if f != LECITHIN]
    X = np.column_stack([lec, others])
    # Offset to 1 d: a plain `10 * scaled + noise` put the first row at -0.06 d and
    # target.to_fit_scale rightly refused it. A negative day count is a data error,
    # so the fixture was wrong, not the guard.
    y = 1.0 + 10.0 * scaled + rng.normal(0, 0.1, N)   # run order only
    return X, y, order, names


def test_confound_views_measure_a_known_confound():
    """Renamed on measurement, 2026-10-09 (PR4). Planned as `..._detect_...`.

    The planned assertion was that the with-covariate view **demotes** a feature that
    only proxies run order. It does not, and cannot: ridge shares weight between two
    collinear predictors instead of handing it all to either, so on data where run
    order is literally the only cause, lecithin keeps rank 1 with run order sitting
    right beside it in the model. An unpenalised fit would demote it -- but
    composition closes to 100 wt%, the 11 features are linearly dependent, and OLS is
    rank-deficient here. The penalty is forced, so the demotion test was never
    available. See `workcycle.md` and `pr-log.md`, PR4.

    What survives, and is asserted: the naive fit is fooled (so the planted confound
    is real), and the delta in out-of-fold R2 from adding run order is **positive**
    here and negative on the real data, so that statistic does discriminate. The rank
    is recorded, not asserted on.
    """
    X, y, order, names = _planted()
    t = confound.triad(X, y, order, feature_names=names, feature=LECITHIN)

    assert t.naive.lecithin_rank == 1, (
        "the planted confound did not even fool the naive fit, so this test proves "
        f"nothing about the triad; got rank {t.naive.lecithin_rank}"
    )
    assert confound.delta_r2(X, order, target.to_fit_scale(y)) > 0.0, (
        "on data caused by run order alone, adding run order must buy out-of-fold "
        "predictive power -- if it does not, the statistic cannot detect a confound"
    )
    assert {v.view for v in t.views} == {"with-covariate", "within-block", "time-split"}
    for v in t.views:
        assert v.n_rows > 0
        assert v.lecithin_sign in (-1, 0, 1)

    lo, hi = t.rank_range()
    assert 1 <= lo <= hi <= len(names)
    assert "rank" in t.as_sentence()
    print()
    print(f"  planted: naive rank {t.naive.lecithin_rank}, "
          f"with-covariate rank {t.with_covariate.lecithin_rank}")
    print(f"  {t.with_covariate.note}")


def test_triad_reports_a_range_not_an_effect_size():
    """The Done gate in prose, as far as the API can carry it.

    The limitations block reads as a sensitivity range. So the objects the report
    path gets hold of expose ranks and a range, and no coefficient, standard error
    or interval that could be read as an effect size.
    """
    X, y, order, names = _planted()
    t = confound.triad(X, y, order, feature_names=names, feature=LECITHIN)
    banned = {"coef", "coefficient", "effect", "se", "stderr", "ci", "ci_low", "ci_high",
              "beta", "days_per_pct"}
    for v in t.views + (t.naive,):
        assert not (set(v._fields) & banned), f"{v.view} exposes {set(v._fields) & banned}"
    assert not (set(t._fields) & banned)


def test_triad_on_the_real_data_emits_all_three():
    """The real run, recorded. No assertion on which way it comes out."""
    df = loader.load_analysis_set()
    X = loader.feature_matrix(df).to_numpy()
    y = loader.target(df).to_numpy()
    order = confound.run_order(df).to_numpy()
    assert len(order) == config.N_ANALYSIS_ROWS

    t = confound.triad(X, y, order, groups=loader.composition_groups(df).to_numpy())
    assert len(t.views) == 3
    print()
    print(f"  naive          : {LECITHIN} rank {t.naive.lecithin_rank}, "
          f"sign {t.naive.lecithin_sign:+d}, n = {t.naive.n_rows}")
    for v in t.views:
        print(f"  {v.view:<15}: rank {v.lecithin_rank}, sign {v.lecithin_sign:+d}, "
              f"n = {v.n_rows}  {v.note}")
    print(f"  sensitivity    : {t.as_sentence()}")


def test_run_order_is_not_a_feature():
    """`num` is the confound, so it is a diagnostic input and never a model input."""
    df = loader.load_analysis_set()
    assert "num" in config.NEVER_A_FEATURE
    assert "num" not in loader.feature_matrix(df).columns
    with pytest.raises(ValueError, match="run order"):
        confound.triad(
            np.column_stack([loader.feature_matrix(df).to_numpy(),
                             confound.run_order(df).to_numpy()]),
            loader.target(df).to_numpy(),
            confound.run_order(df).to_numpy(),
        )
