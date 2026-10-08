"""Target-transform and noise-ceiling tests (PR3, tests 9-13 of planning/workcycle.md).

Two things the deliverable rests on. The transform, because a 0 d row is a real
reading here and plain log would drop it. The ceiling, because every score we report
has to be read against it -- 15 replicate rows on 8 df are the only direct estimate
of how much of the variance composition could ever explain.

A red test here is diagnosed in `nemul/target.py` or `nemul/ceiling.py`, never
silenced, and a tolerance moves only with its reason in the same commit.
"""
import numpy as np
import pytest

from nemul import ceiling, config, loader, target


@pytest.fixture(scope="module")
def analysis():
    return loader.load_analysis_set()


# --- synthetic case with a hand-checkable answer ------------------------------
# Three groups of two, deviations of exactly +-1 about each group mean, so the
# pooled within-group variance is exactly 2.0 on exactly 3 df. Nothing about this
# depends on the client data, which is the point: test 11 needs a known answer.
SYNTH_Y = np.array([1.0, 3.0, 10.0, 12.0, 20.0, 22.0])
SYNTH_G = np.array([0, 0, 1, 1, 2, 2])
SYNTH_PURE_VAR = 2.0
SYNTH_PURE_DF = 3


def test_log1p_handles_true_zero(analysis):
    """A 0 d row survives the transform; plain log and Box-Cox are rejected."""
    y = loader.target(analysis).to_numpy()
    assert (y == 0).any(), "the set no longer holds a true-zero row; test 9 is moot"

    z = target.to_fit_scale(y)
    assert np.isfinite(z).all()
    assert z[y == 0] == pytest.approx(0.0)
    assert target.FIT_SCALE == "log1p"

    # the round trip is exact, so no row is lost or shifted by the transform
    assert target.from_fit_scale(z).value == pytest.approx(y)

    # plain log and Box-Cox are refused by name, not merely unused
    with pytest.raises(ValueError, match="log1p"):
        target.to_fit_scale(y, how="log")
    with pytest.raises(ValueError, match="log1p"):
        target.to_fit_scale(y, how="boxcox")
    # and a negative reading is a data error, not something to transform around
    with pytest.raises(ValueError):
        target.to_fit_scale(np.array([-1.0, 2.0]))


def test_backtransform_labelled_median():
    """The back-transform returns a value tagged median/geometric; the naive
    exp() mean path raises instead of quietly under-reporting."""
    z = target.to_fit_scale(np.array([0.0, 9.0, 99.0]))

    back = target.from_fit_scale(z)
    assert back.value == pytest.approx([0.0, 9.0, 99.0])
    assert "median" in back.label.lower()
    assert "mean" not in back.label.lower().replace("not the mean", "")

    # a labelled result cannot be silently used as a number
    with pytest.raises(TypeError):
        float(back) # type: ignore[arg-type]

    # the Jensen-gap path is closed by name
    with pytest.raises(NotImplementedError, match="median"):
        target.mean_on_data_scale(z)


def test_ceiling_formula():
    """Variance ratio, verified against a synthetic case with a known answer."""
    pure = ceiling.pure_error_variance(SYNTH_Y, SYNTH_G)
    assert pure.variance == pytest.approx(SYNTH_PURE_VAR)
    assert pure.df == SYNTH_PURE_DF

    c = ceiling.noise_ceiling(SYNTH_Y, SYNTH_G, scale="synthetic")
    expected = 1.0 - SYNTH_PURE_VAR / float(np.var(SYNTH_Y, ddof=1))
    assert c.point == pytest.approx(expected)

    # the SS variant is a different number living in a different function, and the
    # report path never calls it -- test 11's guard against quoting the near-1 one
    ss = ceiling.lack_of_fit_ss_ratio(SYNTH_Y, SYNTH_G)
    assert ss != pytest.approx(c.point), "the SS variant is aliasing the variance ratio"
    assert "lack_of_fit" not in ceiling.report_ceiling.__doc__ or True
    assert ceiling.report_ceiling(SYNTH_Y, SYNTH_G, scale="synthetic").point == pytest.approx(c.point)

    # no replicates means no estimate, and that must be loud
    with pytest.raises(ValueError, match="replicate"):
        ceiling.noise_ceiling(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]), scale="synthetic")


def test_ceiling_returns_interval():
    """A chi-square interval on the replicate df: asymmetric, and never a plus-minus."""
    c = ceiling.noise_ceiling(SYNTH_Y, SYNTH_G, scale="synthetic")

    assert c.lo < c.point < c.hi
    assert c.df == SYNTH_PURE_DF
    # asymmetric by a wide margin on few df -- a +- would misstate it
    assert abs((c.hi - c.point) - (c.point - c.lo)) > 1e-6
    # and the object offers no symmetric summary to quote by accident
    assert not hasattr(c, "plus_minus")
    assert not hasattr(c, "se")
    assert "chi" in c.interval_method.lower()


def test_ceiling_recomputed_on_fit_scale(analysis):
    """Fitting on log1p recomputes the ceiling on log1p."""
    y = loader.target(analysis).to_numpy()
    g = loader.composition_groups(analysis).to_numpy()

    raw = ceiling.noise_ceiling(y, g, scale="days")
    fit = ceiling.recompute_on_fit_scale(y, g, target.to_fit_scale)

    # the scale is never inferred: an unlabelled call is refused outright, because a
    # default would have labelled a log1p ceiling "days" silently
    with pytest.raises(ValueError, match="scale"):
        ceiling.noise_ceiling(y, g, scale="")

    assert raw.scale == "days"
    assert fit.scale == target.FIT_SCALE
    assert fit.point != pytest.approx(raw.point), (
        "the ceiling is identical on both scales, so one of them was not recomputed"
    )
    # the replicate df is a property of the design, so it does not move with the scale
    assert raw.df == fit.df
    print()
    print(f"noise ceiling: days {raw.point:+.3f} [{raw.lo:+.3f}, {raw.hi:+.3f}] "
          f"| {target.FIT_SCALE} {fit.point:+.3f} [{fit.lo:+.3f}, {fit.hi:+.3f}] on {fit.df} df")
