"""The confound triad: three ways of asking whether lecithin survives run order.

Composition and run order move together in this dataset -- the high-lecithin
formulations are also the early runs, and all nine of the client's product-ready
picks sit inside that block. The protocol was constant for every sample and nothing
was logged per batch, so there is no covariate that separates them and no analysis
that can. That is permanent: see `planning/project.md`.

What is still possible is a **sensitivity statement**. Fit the same penalised model
three ways, each handling run order differently, and report how far the lecithin
result moves:

- **with-covariate** -- run order enters as a 12th predictor. If lecithin were only a
  proxy for run order, this demotes it.
- **within-block** -- fit inside each run-order block, where run order barely varies,
  so any surviving ordering is not carried by it.
- **time-split** -- fit on the early runs, predict the late ones, and the reverse. A
  model that only learned the run-order gradient does not transfer.

**What this module reports is a rank and a sign, never a magnitude.** The outputs
carry no coefficient, no standard error and no interval, because the Done gate in
`workcycle.md` requires the limitations block to read as a sensitivity range and
because a number like "days per wt%" is exactly what this design cannot identify.
The agreed wording is: direction robust, magnitude not separately identifiable,
randomised follow-up required.

Run order is `num`, which is in `config.NEVER_A_FEATURE`. It is a diagnostic input
here and never a model input; `triad` adds it itself and refuses an `X` that already
carries it.
"""
from typing import NamedTuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, splits, target
from .splits import Fold

LEVER = "lecithin_pct"


class Sensitivity(NamedTuple):
    """One view. A rank among the composition components, and a direction.

    Deliberately no coefficient field: see the module docstring.
    """
    view: str
    lecithin_rank: int      # 1 = largest standardised |coefficient| of the 11
    lecithin_sign: int      # +1 longer-lived, -1 shorter-lived, 0 shrunk to zero
    n_rows: int
    note: str = ""


class Triad(NamedTuple):
    naive: Sensitivity      # composition only, run order ignored -- the reference
    views: tuple            # the three sensitivity views
    feature: str
    n_features: int

    def _view(self, name: str) -> Sensitivity:
        for v in self.views:
            if v.view == name:
                return v
        raise KeyError(name)

    # Named access, so a caller never indexes `views[1]` and silently read the wrong
    # view when the order changes. Properties, not fields: `views` stays the one list.
    @property
    def with_covariate(self) -> Sensitivity:
        return self._view("with-covariate")

    @property
    def within_block(self) -> Sensitivity:
        return self._view("within-block")

    @property
    def time_split(self) -> Sensitivity:
        return self._view("time-split")

    def rank_range(self) -> tuple:
        """The span of the lever's rank across the reference and the three views."""
        ranks = [self.naive.lecithin_rank] + [v.lecithin_rank for v in self.views]
        return min(ranks), max(ranks)

    def signs_agree(self) -> bool:
        signs = [self.naive.lecithin_sign] + [v.lecithin_sign for v in self.views]
        nonzero = [s for s in signs if s != 0]
        return bool(nonzero) and len(set(nonzero)) == 1

    def as_sentence(self) -> str:
        """The sensitivity range in one line, in the agreed wording."""
        lo, hi = self.rank_range()
        span = f"{lo}" if lo == hi else f"{lo}-{hi}"
        direction = "consistent" if self.signs_agree() else "not consistent"
        return (
            f"{self.feature} ranks {span} of {self.n_features} among composition "
            "components by standardised coefficient magnitude (a rank, not an effect "
            f"size) across the naive, with-covariate, within-block and time-split "
            f"views; direction {direction}. Magnitude is not separately identifiable "
            "from run order; a randomised follow-up is required."
        )


def run_order(df):
    """The run-order column, `num`. A diagnostic, never a feature.

    Returned from the loader's dataframe rather than added to `FEATURES`, so there is
    no path by which it reaches `loader.feature_matrix`.
    """
    return df["num"].astype(float)


def _probe():
    """The same penalised linear probe everywhere, so views differ only in their rows."""
    return make_pipeline(StandardScaler(), Ridge(alpha=config.LEAKAGE_PROBE_ALPHA))


def _rank_and_sign(X, z, col: int, n_rank: int):
    """Fit the probe and return the column's |coef| rank and sign.

    Only the first `n_rank` columns are ranked, so a run-order covariate appended at
    the end can influence the fit without competing for a rank.
    """
    est = _probe()
    est.fit(X, z)
    coef = np.asarray(est[-1].coef_, dtype=float).ravel()[:n_rank]
    order = np.argsort(-np.abs(coef))
    rank = int(np.where(order == col)[0][0]) + 1
    c = coef[col]
    return rank, (0 if c == 0 else int(np.sign(c)))


DELTA_R2_REPEATS = 3


def delta_r2(X, order, z, groups=None) -> float:
    """Out-of-fold R2 with run order as a covariate, minus without it.

    The one deconfounding measurement a penalised fit can actually make: if the
    composition signal were only run order's shadow, adding run order would buy
    predictive power. If it buys none, composition is not standing in for it.

    Grouped folds when `groups` is given, so a replicate twin never crosses the
    split; row-wise folds otherwise, which is correct when there are no replicates.
    """
    ga = np.arange(len(z)) if groups is None else np.asarray(groups)
    folds = splits.grouped_folds(ga, n_repeats=DELTA_R2_REPEATS, seed=config.SEED)
    est = _probe
    with_ = splits.cv_r2(np.column_stack([X, order]), z, folds, estimator=est())
    without = splits.cv_r2(X, z, folds, estimator=est())
    return float(with_ - without)


def triad(X, y, order, feature_names=None, feature: str = LEVER, groups=None) -> Triad:
    """The three views plus the naive reference. Fitted on the `log1p` scale.

    `groups` is the composition grouping, used only by the out-of-fold delta in the
    with-covariate view. Omit it on synthetic data with no replicates.
    """
    names = list(config.FEATURES if feature_names is None else feature_names)
    Xa = np.asarray(X, dtype=float)
    if Xa.shape[1] != len(names):
        raise ValueError(
            f"X has {Xa.shape[1]} columns for {len(names)} feature names: the triad "
            "adds run order itself, so pass the composition matrix only. Run order "
            "is a diagnostic input and never a model input."
        )
    if feature not in names:
        raise ValueError(f"{feature!r} is not among the features")
    col = names.index(feature)
    n = len(names)
    z = target.to_fit_scale(np.asarray(y, dtype=float))
    ordr = np.asarray(order, dtype=float)

    rank, sign = _rank_and_sign(Xa, z, col, n)
    naive = Sensitivity("naive", rank, sign, len(z), "run order ignored; the reference")

    # -- with-covariate -----------------------------------------------------------
    # Read the note, not the rank. A penalised fit **cannot** demote a confounded
    # feature: L2 shares weight between two collinear predictors rather than giving
    # it all to either, so lecithin keeps its rank even on synthetic data where run
    # order is the only cause (`test_confound_views_detect_a_known_confound`). An
    # unpenalised fit would demote it, but composition closes to 100 wt%, so the 11
    # features are linearly dependent and OLS is rank-deficient here. The penalty is
    # not a choice. What this view can measure is whether run order *adds predictive
    # power* over composition alone -- that delta is the deconfounding evidence.
    Xc = np.column_stack([Xa, ordr])
    rank_c, sign_c = _rank_and_sign(Xc, z, col, n)
    d = delta_r2(Xa, ordr, z, groups)
    with_cov = Sensitivity(
        "with-covariate", rank_c, sign_c, len(z),
        f"rank {rank} -> {rank_c} (a penalty cannot demote a collinear proxy); "
        f"run order adds dR2 {d:+.3f} out-of-fold",
    )

    # -- within-block -------------------------------------------------------------
    edges = np.quantile(ordr, np.linspace(0, 1, config.N_RUN_BLOCKS + 1))
    block_ranks, block_signs, used = [], [], 0
    for i in range(config.N_RUN_BLOCKS):
        lo, hi = edges[i], edges[i + 1]
        m = (ordr >= lo) & (ordr <= hi if i == config.N_RUN_BLOCKS - 1 else ordr < hi)
        if m.sum() <= n:          # fewer rows than features: no usable fit
            continue
        r, s = _rank_and_sign(Xa[m], z[m], col, n)
        block_ranks.append(r)
        block_signs.append(s)
        used += int(m.sum())
    if not block_ranks:
        raise ValueError(
            f"no run-order block has more than {n} rows, so the within-block view is "
            "not computable; reduce config.N_RUN_BLOCKS or say so in the report"
        )
    # The worst rank is reported: the view's job is to show how far the lever can
    # fall, not to find the block that flatters it.
    nz = [s for s in block_signs if s != 0]
    within = Sensitivity(
        "within-block", max(block_ranks),
        (nz[0] if nz and len(set(nz)) == 1 else 0), used,
        f"{len(block_ranks)} block(s), ranks {block_ranks}, signs {block_signs}",
    )

    # -- time-split ---------------------------------------------------------------
    cut = np.median(ordr)
    early, late = ordr <= cut, ordr > cut
    rank_t, sign_t = _rank_and_sign(Xa[early], z[early], col, n)
    est = _probe().fit(Xa[early], z[early])
    fwd = float(r2_score(z[late], est.predict(Xa[late])))
    est2 = _probe().fit(Xa[late], z[late])
    back = float(r2_score(z[early], est2.predict(Xa[early])))
    time_split = Sensitivity(
        "time-split", rank_t, sign_t, int(early.sum()),
        f"early->late R2 {fwd:+.3f}, late->early R2 {back:+.3f} "
        f"(transfer, not a fit quality claim)",
    )

    return Triad(naive, (with_cov, within, time_split), feature, n)

def block_folds(order, groups=None, n_blocks=None) -> list:
    """Leave-one-run-block-out folds: the ladder's folds, re-cut along run order.

    `splits.grouped_folds` keeps a composition out of its own training set. These
    folds keep a whole *era* out, so a model cannot interpolate between neighbouring
    runs. The difference between a rung's score on the two is a direct measure of how
    much of its performance is local interpolation inside a run-order block -- which
    is what PR4 found the nonlinear rungs to be living on. See `pr-log.md`, PR4.

    Raises if a composition group spans two blocks, because then a replicate twin
    would cross the boundary and the comparison would be contaminated by the thing
    `grouped_folds` exists to prevent.
    """
    ordr = np.asarray(order, dtype=float)
    k = config.N_RUN_BLOCKS if n_blocks is None else n_blocks
    edges = np.quantile(ordr, np.linspace(0, 1, k + 1))
    block = np.digitize(ordr, edges[1:-1])

    if groups is not None:
        # Six of the seven replicate groups span a run block -- the replicates were
        # spread across run order, not run back to back (PR4 measurement). So the
        # blocks are *nested on the groups*: a whole composition goes to the block of
        # its median run. Without this a twin would cross the split and inflate the
        # block-out score, which is the number this comparison exists to depress.
        ga = np.asarray(groups)
        for g in np.unique(ga):
            m = ga == g
            block[m] = int(np.bincount(block[m]).argmax())
        counts = np.bincount(block, minlength=k)
        if (counts == 0).any():
            raise ValueError(
                f"nesting blocks on groups emptied a block: sizes {counts.tolist()}"
            )

    idx = np.arange(len(ordr))
    return [
        Fold(repeat=0, fold=int(b), train=idx[block != b], test=idx[block == b])
        for b in np.unique(block)
    ]
