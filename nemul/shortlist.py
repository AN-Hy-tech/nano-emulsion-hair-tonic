"""The screening shortlist: generated candidate compositions, ranked and caveated.

The client's question is which formulations to make next, so the answer has to be a
set of compositions that do not exist yet. That makes it an extrapolation, and the
safer option -- re-ranking the 101 compositions already made -- was considered and
rejected (Albert, 2026-10-09) because it does not answer the question. So it ships
with guardrails rather than without them, and the guardrails are the deliverable as
much as the ranking is:

1. **Candidates are generated inside the observed per-component ranges and closed to
   100 wt%.** The composition space is a simplex; an unconstrained grid is almost
   entirely infeasible. A candidate a renormalisation pushed outside an observed
   range is *rejected*, never clipped -- clipping reopens closure silently.
2. **Ranked by the penalised linear model, never by the GP or the GBM.** PR4 measured
   both nonlinear rungs losing their whole advantage under run-block folds: they rank
   by interpolating inside dense run-order blocks. Scoring unseen candidates with
   either would export that artefact to the client as advice.
3. **Every candidate carries its distance to the nearest observed composition and an
   extrapolation flag.** The threshold is computed from the design's own density, not
   typed in.
4. **Every candidate carries its distance to the early high-lecithin run block**
   (runs 16-34, where all nine product-ready picks sit). A candidate that ranks well
   *because* it resembles that block is ranking on the confound. One that ranks well
   and sits away from it is the one worth making, and that column is what makes this
   better than a model score.
5. **The output is an order plus an interval width, never a predicted day count.**
   The ranking score is not exposed at all: a `log1p`-scale score is one `expm1` away
   from a number somebody would paste into a sentence as a shelf life.
6. **The honest claim travels with the table** (`CLAIM` below).

Distances are Euclidean in wt% over the 11 components -- the same metric
`scripts/stats_report.py` uses for the median nearest-different-composition distance,
so the threshold in guardrail 3 and the figure in `planning/project.md` sit on one
scale by construction.
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import config, intervals, ladder, loader, splits, target

LABEL = "screening order, not a shelf-life forecast"

CLAIM = (
    "These are the candidate compositions most consistent with the ordering this "
    "dataset supports. The ranking model's ordering holds up better than its "
    "magnitude, and no evidence says the ordering holds outside the sampled region "
    "-- which is what the extrapolation column is for. This is a screening aid for "
    "choosing what to prepare next, not a prediction of shelf life for any candidate."
)


class Shortlist(NamedTuple):
    table: pd.DataFrame        # screening_rank, the 11 components, width, two distances
    n_candidates: int          # feasible candidates ranked
    n_generated: int           # candidates drawn before the feasibility rejection
    threshold_wt_pct: float    # the extrapolation threshold, computed from the data
    n_flagged: int             # of the delivered rows
    model: object              # the fitted ranking model; a test asserts it is linear
    model_detail: str
    label: str = LABEL
    claim: str = CLAIM


def renormalise(X) -> pd.DataFrame:
    """Scale every row to close to 100 wt%.

    The same operation `scripts/build_clean_dataset.py` applies to the real masses --
    each component over the row total, times 100 -- so a candidate and an observed row
    mean the same thing. Written here as one expression rather than imported, because
    that script is a top-to-bottom data rebuild with no function to call.
    """
    df = pd.DataFrame(X).astype(float)
    totals = df.to_numpy().sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("a candidate with a zero total has no composition to normalise")
    return pd.DataFrame(100.0 * df.to_numpy() / totals, columns=df.columns, index=df.index)


def generate_candidates(X_observed, n=None, seed=None) -> pd.DataFrame:
    """`n` feasible candidate compositions: in range, and closed to 100 wt%.

    Draws each component uniformly inside its observed range, renormalises, then
    rejects any row the renormalisation pushed outside a range and draws again. Water
    carries most of the mass, so most draws survive. The loop is bounded and raises
    rather than returning a short table: a silently short grid would make the
    shortlist's length a function of the seed.
    """
    obs = pd.DataFrame(X_observed, columns=config.FEATURES).astype(float)
    n = config.SHORTLIST_N_CANDIDATES if n is None else n
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    lo, hi = obs.min().to_numpy(), obs.max().to_numpy()

    kept, drawn = [], 0
    while sum(len(k) for k in kept) < n:
        if drawn > 200 * n:
            raise ValueError(
                f"drew {drawn} candidates for {n} feasible ones: the observed ranges "
                "and the closure constraint are barely compatible, which is a finding "
                "about the design space and not something to loosen quietly"
            )
        batch = rng.uniform(lo, hi, size=(max(n, 256), len(config.FEATURES)))
        drawn += len(batch)
        cand = renormalise(pd.DataFrame(batch, columns=config.FEATURES))
        ok = ((cand.to_numpy() >= lo - 1e-9) & (cand.to_numpy() <= hi + 1e-9)).all(axis=1)
        kept.append(cand.loc[ok])

    out = pd.concat(kept, ignore_index=True).iloc[:n].reset_index(drop=True)
    return out[config.FEATURES]


def local_step(X_observed, groups) -> float:
    """The per-component perturbation size that lands a candidate at the design's own
    density: the median nearest-different-composition distance, spread over the 11
    components.

    A Gaussian step of `s` on each of `p` independent components moves a point about
    `s * sqrt(p)` in Euclidean distance, so `s = median_nn / sqrt(p)` puts a candidate
    roughly a typical neighbour-spacing away from the row it came from. Derived from
    the data for the same reason `extrapolation_threshold` is: a pinned step would stop
    matching the design the moment a batch is added.

    It is comfortably above `config.GROUP_TOL_PP`, so a candidate is a genuinely new
    composition by the project's own grouping contract and not a relabelled twin.
    """
    obs = pd.DataFrame(X_observed, columns=config.FEATURES).to_numpy()
    ga = np.asarray(groups)
    d = np.sqrt(((obs[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2))
    d[ga[:, None] == ga[None, :]] = np.inf
    nn = d.min(axis=1)
    return float(np.median(nn[np.isfinite(nn)]) / np.sqrt(len(config.FEATURES)))


def generate_local_candidates(X_observed, groups, n=None, seed=None) -> pd.DataFrame:
    """`n` candidates drawn as local perturbations of the observed compositions.

    **This is the generator the delivered shortlist uses, and the reason is measured.**
    `generate_candidates` draws uniformly inside the observed per-component ranges,
    which is what `workcycle.md` specified and the obvious reading of it. On 2026-10-09
    that was measured: in 11 dimensions the 109 observed rows occupy a thin shell of
    the box they span, and **not one of 300 uniform draws landed within 3.11 wt% of any
    observed composition** -- so every uniform candidate is an extrapolation by
    construction and the flag fires on all of them. The box is not the design space;
    the neighbourhood of the design is.

    So each candidate here starts from a real formulation, takes a Gaussian step of
    `local_step` per component, and is renormalised and range-checked exactly as the
    uniform ones are. A step is rejected rather than clipped, same as there.
    """
    obs = pd.DataFrame(X_observed, columns=config.FEATURES).astype(float)
    n = config.SHORTLIST_N_CANDIDATES if n is None else n
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    lo, hi = obs.min().to_numpy(), obs.max().to_numpy()
    base = obs.to_numpy()
    s = local_step(obs, groups)

    kept, drawn = [], 0
    while sum(len(k) for k in kept) < n:
        if drawn > 200 * n:
            raise ValueError(
                f"drew {drawn} perturbations for {n} feasible candidates: the step "
                f"{s:.3f} wt% pushes almost everything out of range, which is a finding "
                "about the design space and not something to loosen quietly"
            )
        seeds = base[rng.integers(0, len(base), size=max(n, 256))]
        batch = seeds + rng.normal(0.0, s, size=seeds.shape)
        batch = np.clip(batch, 0.0, None)      # a negative wt% is not a composition
        drawn += len(batch)
        cand = renormalise(pd.DataFrame(batch, columns=config.FEATURES))
        ok = ((cand.to_numpy() >= lo - 1e-9) & (cand.to_numpy() <= hi + 1e-9)).all(axis=1)
        kept.append(cand.loc[ok])

    out = pd.concat(kept, ignore_index=True).iloc[:n].reset_index(drop=True)
    return out[config.FEATURES]


def _pairwise_min(A, B) -> np.ndarray:
    """Euclidean distance from each row of A to its nearest row of B, in wt%."""
    a = np.asarray(A, dtype=float)
    b = np.asarray(B, dtype=float)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def nearest_observed_distance(candidates, X_observed) -> np.ndarray:
    """How far each candidate sits from the nearest formulation actually made."""
    return _pairwise_min(
        pd.DataFrame(candidates, columns=config.FEATURES).to_numpy(),
        pd.DataFrame(X_observed, columns=config.FEATURES).to_numpy(),
    )


def extrapolation_threshold(X_observed, groups) -> float:
    """The flag's threshold, computed from the design's own density.

    The distribution is each observed row's distance to the nearest row of a
    *different* composition -- the same quantity `stats_report.py` takes the median of
    (0.762 wt%, owned by `project.md`). A candidate further from the data than the
    `EXTRAPOLATION_PERCENTILE`th percentile of that distribution sits outside the
    density the model was fitted in, so it is flagged.

    Computed rather than pinned: a hardcoded threshold goes stale the moment a batch
    is added, and the one thing this flag must not be is quietly wrong.
    """
    obs = pd.DataFrame(X_observed, columns=config.FEATURES).to_numpy()
    ga = np.asarray(groups)
    d = np.sqrt(((obs[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2))
    d[ga[:, None] == ga[None, :]] = np.inf      # a twin is not a different composition
    nn = d.min(axis=1)
    return float(np.percentile(nn[np.isfinite(nn)], config.EXTRAPOLATION_PERCENTILE))


def extrapolation_flags(candidates, X_observed, threshold) -> np.ndarray:
    return nearest_observed_distance(candidates, X_observed) > threshold


def early_block_distance(candidates, df) -> np.ndarray:
    """Distance to the nearest row of the early high-lecithin block (runs 16-34).

    Low means the candidate resembles the block the run-order confound lives in, and
    where every one of the client's product-ready picks sits. It is reported and never
    acted on: the shortlist does not filter on it, because a filter would be us
    deciding how much of the confound the client should tolerate.
    """
    lo, hi = config.EARLY_BLOCK_RUNS
    block = df.loc[(df["num"] >= lo) & (df["num"] <= hi)]
    if block.empty:
        raise ValueError(f"no rows in runs {lo}-{hi}; the early block is empty")
    return _pairwise_min(
        pd.DataFrame(candidates, columns=config.FEATURES).to_numpy(),
        loader.feature_matrix(block).to_numpy(),
    )


def ranking_model():
    """The penalised linear rung, tuned exactly as the ladder tunes it. Guardrail 2."""
    return ladder.ridge_tuned()


def nearest_observed_index(candidates, X_observed) -> np.ndarray:
    """Which observed row each candidate is nearest to. Its *neighbourhood*."""
    a = pd.DataFrame(candidates, columns=config.FEATURES).to_numpy()
    b = pd.DataFrame(X_observed, columns=config.FEATURES).to_numpy()
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).argmin(axis=1)


def shortlist(df, n_candidates=None, top_n=None, alpha: float = 0.1, seed=None,
              max_distance=None, local: bool = True,
              one_per_neighbourhood: bool = True) -> Shortlist:
    """Rank a generated candidate grid and return the top `top_n` with their caveats.

    Fitted on `log1p(stability_days)` over every analysis row with the alpha tuned in
    grouped inner folds, then used to order the candidates. The jackknife+ band comes
    from leave-one-group-out folds on the observed data, so the width each candidate
    carries is the width this dataset can actually support there -- which is the number
    that makes the ranking readable as a screening aid instead of a forecast.

    `local` picks the generator: perturbations of observed compositions (the default,
    and what the delivered table uses) or a uniform draw inside the observed ranges.
    See `generate_local_candidates` for the measurement that settled it.

    `max_distance` restricts the pool to candidates within that wt% of an observed
    composition, and **both views are delivered**. Measured 2026-10-09: a penalised
    linear model is monotone in every component, so its maximum over a box-constrained
    grid always sits at a corner of the box -- the furthest feasible point from the
    data. The unconstrained top of a uniform ranking is therefore flagged in every
    row, and a deliverable made only of it recommends nothing. Pass the extrapolation
    threshold here for the view that can be acted on; pass nothing for the view that
    shows why the constraint is needed.

    `one_per_neighbourhood` keeps at most one candidate per observed formulation, and
    it is on by default because of a second measurement on 2026-10-09: without it the
    delivered twelve rows were **all perturbations of one observed row**. The generator
    was not at fault -- it sampled near all 109 rows in proportion -- the ranking
    collapses onto whichever row sits furthest in the model's preferred direction, and
    a monotone model has exactly one of those. Twelve roundings of one composition
    presented as a shortlist misleads by its shape, whatever the caveats say.
    """
    top_n = config.SHORTLIST_TOP_N if top_n is None else top_n
    X = loader.feature_matrix(df)
    y = loader.target(df)
    groups = loader.composition_groups(df)
    z = target.to_fit_scale(y.to_numpy())

    if local:
        cand = generate_local_candidates(X, groups, n=n_candidates, seed=seed)
    else:
        cand = generate_candidates(X, n=n_candidates, seed=seed)
    n_generated = len(cand)

    if max_distance is not None:
        keep = nearest_observed_distance(cand, X) <= max_distance
        cand = cand.loc[keep].reset_index(drop=True)
        if cand.empty:
            raise ValueError(
                f"no candidate of {n_generated} sits within {max_distance:g} wt% of an "
                "observed composition. An empty table would read as 'the model found "
                "no good candidates', which is a far stronger claim than 'the filter "
                "was tighter than the grid's resolution' -- widen one or the other"
            )

    model = ranking_model()
    model.fit(X.to_numpy(), z, groups=groups.to_numpy())
    order = np.asarray(model.predict(cand.to_numpy()), dtype=float)

    # Longest-lived first on the fit scale. These values do not leave this function:
    # guardrail 5 is that the client gets an order, not a score to back-transform.
    ranked = np.argsort(-order)
    nbr = nearest_observed_index(cand, X)
    if one_per_neighbourhood:
        seen, rank_idx = set(), []
        for i in ranked:
            if nbr[i] in seen:
                continue
            seen.add(nbr[i])
            rank_idx.append(i)
            if len(rank_idx) == top_n:
                break
        rank_idx = np.asarray(rank_idx, dtype=int)
    else:
        rank_idx = ranked[:top_n]
    top = cand.iloc[rank_idx].reset_index(drop=True)
    base_runs = df["num"].to_numpy()[nbr[rank_idx]]

    band = intervals.jackknife_plus(
        X.to_numpy(), z, top.to_numpy(),
        folds=splits.interval_folds(groups.to_numpy()),
        estimator=splits.probe_estimator(), alpha=alpha,
    )

    thresh = extrapolation_threshold(X, groups)
    d_obs = nearest_observed_distance(top, X)

    table = pd.DataFrame({"screening_rank": np.arange(1, len(top) + 1)})
    for c in config.FEATURES:
        table[c] = top[c].round(3)
    table["interval_width_log1p"] = (band.upper - band.lower).round(3)
    table["nearest_observed_run"] = base_runs
    table["wt_pct_to_nearest_observed"] = d_obs.round(3)
    table["extrapolation_flag"] = d_obs > thresh
    table["wt_pct_to_early_block"] = early_block_distance(top, df).round(3)

    return Shortlist(
        table=table,
        n_candidates=len(cand),
        n_generated=n_generated,
        threshold_wt_pct=thresh,
        n_flagged=int(table["extrapolation_flag"].sum()),
        model=model,
        model_detail=", ".join(
            f"{k.split('__')[-1]}={v:g}" for k, v in model.best_params_.items()
        ),
    )
