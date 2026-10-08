"""The screening shortlist's guardrails, as tests.

Three were named in `workcycle.md` before any of this was built, and the third is
built as a **known-answer check**: a candidate planted far outside the data must come
back flagged, and that is verified before any real shortlist is believed. PR4's lesson
was that a diagnostic nobody has seen fire is not a diagnostic.

The fourth test here is the one that matters most for what the client is handed. PR4
measured the GP and the GBM ranking by run-order interpolation; using either to score
unseen candidates would export that artefact as advice. So the ranking model is
asserted to be the penalised linear one, by type.
"""
import numpy as np
import pandas as pd
import pytest

from nemul import config, loader, shortlist


@pytest.fixture(scope="module")
def analysis():
    return loader.load_analysis_set()


@pytest.fixture(scope="module")
def observed(analysis):
    return loader.feature_matrix(analysis)


@pytest.fixture(scope="module")
def groups(analysis):
    return loader.composition_groups(analysis)


@pytest.fixture(scope="module")
def result(analysis):
    # A small grid: these tests check structure and guardrails, not the ranking's
    # content, and building the full grid is the artefact script's job.
    return shortlist.shortlist(analysis, n_candidates=300, top_n=10, local=False)


@pytest.fixture(scope="module")
def local_result(analysis, result):
    """The delivered view: local candidates, restricted to the sampled density."""
    return shortlist.shortlist(
        analysis, n_candidates=300, top_n=10,
        max_distance=result.threshold_wt_pct,
    )


@pytest.mark.parametrize("generator", ["uniform", "local"])
def test_candidates_close_and_sit_inside_observed_ranges(observed, groups, generator):
    """Guardrail 1, on **both** generators. Closed to 100 wt%, no component outside range.

    The composition space is a simplex, so an unconstrained grid is mostly infeasible
    points. Generation renormalises to 100 wt% and then *rejects* anything the
    renormalisation pushed outside the observed per-component range, rather than
    clipping it -- clipping would silently reopen closure.
    """
    if generator == "uniform":
        cand = shortlist.generate_candidates(observed, n=400)
    else:
        cand = shortlist.generate_local_candidates(observed, groups, n=400)

    assert list(cand.columns) == config.FEATURES, "candidates must be the 11, in order"
    assert len(cand) == 400

    worst = float(np.abs(cand.to_numpy().sum(axis=1) - 100.0).max())
    assert worst < config.CLOSURE_TOL_PP, (
        f"{generator} candidates must close to 100 wt% as tightly as the real rows "
        f"do; worst was {worst:.2e} pp"
    )

    lo, hi = observed.min(), observed.max()
    for c in config.FEATURES:
        assert cand[c].min() >= lo[c] - 1e-9, f"{c} below the observed range"
        assert cand[c].max() <= hi[c] + 1e-9, f"{c} above the observed range"


def test_local_candidates_are_new_compositions_not_relabelled_twins(observed, groups):
    """A candidate must be distinguishable from the row it was perturbed from.

    The perturbation step is derived from the measured neighbour spacing, so it should
    land well clear of `GROUP_TOL_PP` -- the tolerance at which the project calls two
    rows the same composition. If it did not, the shortlist would be recommending
    formulations the client has already made, under new numbers.
    """
    step = shortlist.local_step(observed, groups)
    assert step > config.GROUP_TOL_PP, (
        f"perturbation step {step:.4f} wt% is inside the {config.GROUP_TOL_PP} wt% "
        "grouping tolerance: these would be relabelled twins, not new compositions"
    )

    cand = shortlist.generate_local_candidates(observed, groups, n=200)
    d = shortlist.nearest_observed_distance(cand, observed)
    assert (d > config.GROUP_TOL_PP).all(), "a candidate duplicates an observed row"


def test_a_uniform_grid_cannot_interpolate_in_eleven_dimensions(observed, groups):
    """Measured on build, 2026-10-09, and it is why the local generator exists.

    `workcycle.md` specified candidates drawn inside the observed per-component
    ranges, and a uniform draw over that box is the obvious reading. It does not work:
    the 109 observed rows occupy a thin shell of the box they span, so a uniform draw
    lands nowhere near them and **every** candidate is an extrapolation by
    construction. Recorded here, not asserted as a fixed count -- a future batch that
    fills the interior legitimately changes it.
    """
    thresh = shortlist.extrapolation_threshold(observed, groups)
    uni = shortlist.nearest_observed_distance(
        shortlist.generate_candidates(observed, n=300), observed
    )
    loc = shortlist.nearest_observed_distance(
        shortlist.generate_local_candidates(observed, groups, n=300), observed
    )
    print(
        f"\nnearest-observed distance, threshold {thresh:.3f} wt%:\n"
        f"  uniform box : median {np.median(uni):.3f}, "
        f"{(uni <= thresh).sum()}/300 inside the density\n"
        f"  local step  : median {np.median(loc):.3f}, "
        f"{(loc <= thresh).sum()}/300 inside the density"
    )
    assert np.median(loc) < np.median(uni), (
        "the local generator must put candidates nearer the data than a uniform box "
        "draw does, or it has no reason to exist"
    )


def test_shortlist_carries_no_day_count(result):
    """Guardrail 5. The output is an order and an interval width, never a forecast.

    Checked as a type, not as prose: no column name may read as a day count or a
    predicted stability, and the ranking score itself is not exposed at all. A score
    on the `log1p` scale is one `expm1` away from a number somebody would paste into a
    sentence as a shelf life.
    """
    cols = [c.lower() for c in result.table.columns]

    for forbidden in ("day", "stability", "shelf", "predict", "forecast", "score"):
        offenders = [c for c in cols if forbidden in c]
        assert not offenders, (
            f"{offenders} reads as a prediction; the shortlist ships a screening "
            "order, not a shelf-life forecast"
        )

    assert "screening_rank" in result.table.columns
    assert result.table["screening_rank"].tolist() == list(range(1, len(result.table) + 1))
    # The interval width is scale-labelled in its own name, so it cannot be read as days.
    assert "interval_width_log1p" in result.table.columns
    assert (result.table["interval_width_log1p"] > 0).all()
    assert "screening order" in result.label.lower()


def test_planted_far_candidate_is_flagged_as_extrapolation(observed, analysis):
    """Guardrail 3, as a known-answer check. Build the answer, then read the flag.

    Two planted rows: one copied from an observed formulation, which must come back
    unflagged, and one pushed to the far end of the widest observed ranges, which must
    come back flagged. A flag that never fires would let the whole shortlist ship as
    interpolation when it is not.
    """
    thresh = shortlist.extrapolation_threshold(observed, loader.composition_groups(analysis))
    assert thresh > 0

    near = observed.iloc[[0]].copy()
    far = observed.iloc[[0]].copy()
    # Move the two widest-range components to the opposite end of their observed span
    # and renormalise, so the point stays feasible and is merely remote.
    span = (observed.max() - observed.min()).sort_values(ascending=False)
    for c in span.index[:2]:
        far[c] = (
            observed[c].max() if far[c].iloc[0] < observed[c].mean() else observed[c].min()
        )
    far = shortlist.renormalise(far)

    planted = pd.concat([near, far], ignore_index=True)
    d = shortlist.nearest_observed_distance(planted, observed)

    assert d[0] < 1e-9, "a copy of an observed row is at distance zero from the data"
    assert d[1] > thresh, (
        f"the planted far candidate is {d[1]:.3f} wt% from the data against a "
        f"threshold of {thresh:.3f}; the known answer is 'flagged'"
    )

    flags = shortlist.extrapolation_flags(planted, observed, thresh)
    assert flags.tolist() == [False, True]


def test_shortlist_is_ranked_by_the_penalised_linear_model(result):
    """Guardrail 2. Never the GP, never the GBM.

    PR4 measured both nonlinear rungs losing their entire advantage under run-block
    folds: they rank by interpolating inside dense run-order blocks. Scoring unseen
    candidates with either would hand that artefact to the client as advice. Asserted
    on the fitted object, so swapping the rung trips it.
    """
    from sklearn.linear_model import Ridge

    model = result.model
    final = model.best_[-1] if hasattr(model, "best_") else model[-1]
    assert isinstance(final, Ridge), (
        f"the shortlist is ranked by {type(final).__name__}; it must be the "
        "penalised linear model (workcycle.md, Decisions)"
    )
    assert "alpha" in result.model_detail


def test_unconstrained_shortlist_is_entirely_extrapolation(result):
    """Measured on build, 2026-10-09, and it is the shortlist's main finding.

    A penalised *linear* model is monotone in every component, so its maximum over a
    box-constrained grid always sits at a corner of that box -- the most extreme
    feasible composition, which is by construction the furthest from the data. So the
    unconstrained top of the ranking is not merely sometimes flagged, it is
    **always** flagged, and a deliverable made of it recommends nothing.

    This test records that, it does not assert a count: if a future batch fills the
    corners the flags legitimately clear. What it asserts is the structural fact that
    makes the constrained shortlist below necessary -- the top-ranked candidate is
    further from the data than the threshold.
    """
    worst = result.table["wt_pct_to_nearest_observed"].max()
    print(
        f"\nunconstrained shortlist: {result.n_flagged} of {len(result.table)} rows "
        f"flagged, furthest {worst:.3f} wt% against a threshold of "
        f"{result.threshold_wt_pct:.3f} wt%"
    )
    assert result.n_flagged > 0, (
        "the linear ranking's top candidates sit at the corners of the feasible box; "
        "if none is flagged, either the threshold or the distance metric has changed"
    )


def test_constrained_shortlist_stays_inside_the_sampled_density(local_result, result):
    """The shortlist that can actually be acted on: ranked, but inside the data.

    Same model, same guardrails -- the pool is local candidates restricted to points
    within the extrapolation threshold of an observed composition. Without this the
    delivered table is twelve rows that all say "do not trust me", which answers the
    client's question with nothing.
    """
    inside = local_result

    assert inside.n_flagged == 0, "a constrained shortlist must carry no flagged row"
    assert (
        inside.table["wt_pct_to_nearest_observed"] <= result.threshold_wt_pct + 1e-9
    ).all()
    assert len(inside.table) > 0, "no feasible candidate sits inside the density"
    # Same guardrails as the unconstrained table: it is the pool that narrowed.
    assert "interval_width_log1p" in inside.table.columns
    assert "wt_pct_to_early_block" in inside.table.columns
    assert inside.n_candidates <= inside.n_generated


def test_delivered_shortlist_is_twelve_leads_and_not_one_lead_twelve_times(local_result):
    """Measured on build, 2026-10-09, and it changed the deliverable.

    The first constrained shortlist returned twelve rows that were all perturbations of
    **one** observed formulation. The generator was not at fault -- it drew candidates
    near all 109 rows in proportion to their number -- the ranking simply collapses
    onto whichever observed row sits furthest in the model's preferred direction, and a
    monotone model has exactly one of those.

    A client handed twelve rows reads twelve options. Twelve roundings of one
    composition is a presentation that misleads by its shape, whatever the caveats say.
    So the delivered shortlist keeps at most one candidate per observed neighbourhood,
    and this asserts it: as many distinct neighbourhoods as rows.
    """
    runs = local_result.table["nearest_observed_run"]
    assert runs.nunique() == len(local_result.table), (
        f"{len(local_result.table)} rows but only {runs.nunique()} distinct "
        f"neighbourhood(s) ({sorted(set(runs))}): that is one lead repeated, not a "
        "shortlist"
    )


def test_a_threshold_that_admits_nothing_raises(analysis):
    """An empty pool is an error, not an empty table.

    A silently empty shortlist would read in the deliverable as "the model found no
    good candidates", which is a different and much stronger claim than "the filter
    was set tighter than the grid's resolution".
    """
    with pytest.raises(ValueError, match="no candidate"):
        shortlist.shortlist(analysis, n_candidates=50, top_n=5, max_distance=1e-6)


def test_shortlist_reports_distance_to_the_early_high_lecithin_block(result):
    """Guardrail 4 -- the column that makes this better than a model score.

    All nine of the client's product-ready picks sit in runs 16-34. A candidate that
    ranks well *because* it resembles that block is ranking on the confound; one that
    ranks well and sits away from it is the one worth making.
    """
    col = "wt_pct_to_early_block"
    assert col in result.table.columns
    assert (result.table[col] >= 0).all()
    assert result.table[col].notna().all()
