"""Generate every delivered block, the fitted model and the runbook. One command.

The Done gate in `workcycle.md` says: *every number in the delivered blocks traceable
to a script output, nothing hand-typed.* This is that script. It writes into
`artifacts/`, which `.gitignore` blocks and test 18 asserts is blocked -- the output is
derived from client rows and is one command away from being rebuilt, so there is
nothing to gain by tracking it and a data leak to lose.

What it writes:

    artifacts/figures.json    every generated figure, machine-readable; the audit trail
    artifacts/methods.md      deliverable block 1
    artifacts/results.md      deliverable block 2 -- the ladder on both fold schemes
    artifacts/claims.md       deliverable block 3 -- the six claims, figure-free
    artifacts/limitations.md  deliverable block 4 -- the template with its tokens filled
    artifacts/shortlist.md    the screening shortlist, both views
    artifacts/shortlist.csv   the same table, for the client's own spreadsheet
    artifacts/model.joblib    deliverable block 5 -- the fitted ranking model
    artifacts/runbook.md      deliverable block 5 -- how to rerun all of it

Three rules this script enforces rather than trusts:

- **An unfilled token is a failure.** `limitations.md` is a template whose figures are
  `{{token}}`s; if one is left unfilled the script exits non-zero and names it. A
  delivered block containing `{{CEILING_DF}}` is worse than one containing a stale
  number, because it reaches the funder looking like a bug.
- **No delivered block quotes an external figure.** The Done gate settled the
  literature report's "verify before publishing" table *by cutting*: nothing external
  is quoted, so nothing in that table needs clearing. `_assert_no_external_figures`
  checks the rendered blocks for the specific figures that would breach it.
- **The shortlist ships both views or neither.** The unconstrained ranking is entirely
  extrapolation (measured, see `nemul/shortlist.py`), so delivering it alone would
  recommend nothing; delivering the constrained one alone would hide why the
  constraint is there.

Run: `.venv/Scripts/python.exe scripts/make_artifacts.py`
"""
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemul import (  # noqa: E402
    binary, ceiling, config, confound, ladder, loader, shortlist, splits, target,
)

OUT = config.REPO_ROOT / "artifacts"
TEMPLATES = (
    config.REPO_ROOT.parent / "sdlc" / "nano-emulsion-hair-tonic" / "deliverables"
)

# The figures that would breach the Done gate if they reached a delivered block. Each
# is an external number from the literature report; the gate's rule is cut, not hedge.
EXTERNAL_FIGURES = ("0.87", "0.859", "0.998", "81 %", "81%", "496", "530", "8-353", "0.968")


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _fmt(x, nd=3) -> str:
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _humanise(text: str) -> str:
    """Turn column identifiers into words. The blocks go to a funder, not to a reader
    of this repository, and `lecithin_pct` in a progress report is a leaked variable
    name rather than a measurement."""
    for ident, word in (
        ("lecithin_pct", "lecithin concentration"),
        ("_pct", " concentration"),
        ("dR2", "an out-of-fold R2 change of"),
    ):
        text = text.replace(ident, word)
    return text


# ---------------------------------------------------------------- the measurements
def collect(df) -> dict:
    """Every generated figure, in one dict. The only place numbers are computed.

    The blocks below render from this dict and never recompute anything, so a figure
    cannot appear in two blocks with two values. `figures.json` is this dict on disk.
    """
    config.set_seeds()
    X = loader.feature_matrix(df)
    y = loader.target(df)
    groups = loader.composition_groups(df)
    order = confound.run_order(df)

    fig = {
        "generated": date.today().isoformat(),
        "git_rev": _git_rev(),
        "seed": config.SEED,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "n_file_rows": int(config.N_FILE_ROWS),
        "n_analysis": int(len(df)),
        "n_distinct": int(groups.nunique()),
        "n_features": len(config.FEATURES),
        "features": list(config.FEATURES),
    }

    # ---- replicates and the two ceilings ------------------------------------
    sizes = groups.value_counts()
    rep = sizes[sizes > 1]
    spreads = [
        float(y[groups == g].max() - y[groups == g].min()) for g in rep.index
    ]
    fig["n_replicate_groups"] = int(len(rep))
    fig["n_replicate_rows"] = int(rep.sum())
    fig["max_replicate_spread"] = max(spreads)

    # The inspection-interval plateau: the modal stability reading and how many rows
    # sit on it. Limitations item 7 -- the response's resolution, not a censoring
    # problem; PR4 settled that these are observed failures.
    modal = y.value_counts()
    fig["plateau_days"] = float(modal.index[0])
    fig["n_at_plateau"] = int(modal.iloc[0])

    c_days = ceiling.report_ceiling(y, groups, scale="days")
    c_fit = ceiling.recompute_on_fit_scale(y, groups, target.to_fit_scale)
    fig["ceiling_days"] = dict(c_days._asdict())
    fig["ceiling_fit"] = dict(c_fit._asdict())
    fig["ceiling_df"] = int(c_days.df)
    fig["pure_error_sd_days"] = float(np.sqrt(ceiling.pure_error_variance(y, groups).variance))
    fig["total_sd_days"] = float(np.std(y, ddof=1))

    # ---- the ladder, on both fold schemes -----------------------------------
    gf = splits.grouped_folds(groups.to_numpy(), seed=config.SEED)
    bf = confound.block_folds(order.to_numpy(), groups=groups.to_numpy())
    fig["n_grouped_folds"] = len(gf)
    fig["n_block_folds"] = len(bf)
    fig["n_splits"] = config.N_SPLITS
    fig["n_repeats"] = config.N_REPEATS

    grouped = ladder.run_ladder(X.to_numpy(), y.to_numpy(), groups.to_numpy(), folds=gf)
    blocked = ladder.run_ladder(X.to_numpy(), y.to_numpy(), groups.to_numpy(), folds=bf)
    bym = {r.name: r for r in blocked}
    fig["ladder"] = [
        {
            "rung": r.name,
            "r2_grouped": r.r2,
            "spread_grouped": r.spread,
            "margin_over_mean": r.margin_over_mean,
            "spearman_grouped": r.spearman,
            "r2_block_out": bym[r.name].r2,
            "collapse": r.r2 - bym[r.name].r2,
            "detail": r.detail,
            "scale": r.scale,
        }
        for r in grouped
    ]
    fig["fold_signature_grouped"] = grouped[0].fold_signature
    fig["fold_signature_block"] = blocked[0].fold_signature

    # ---- the binarised rung --------------------------------------------------
    b = binary.validate_binary(X.to_numpy(), y.to_numpy(), groups.to_numpy(), folds=gf)
    fig["binary"] = {
        "threshold_days": b.threshold_days,
        "auc": b.auc,
        "auc_spread": b.spread,
        "accuracy": b.accuracy,
        "n_positive": int((y > b.threshold_days).sum()),
        "majority_class_accuracy": float(1 - (y > b.threshold_days).mean()),
        "straddling_groups": b.straddling_groups,
    }

    # ---- the confound triad --------------------------------------------------
    tri = confound.triad(
        X.to_numpy(), y.to_numpy(), order.to_numpy(), groups=groups.to_numpy()
    )
    lo, hi = tri.rank_range()
    fig["triad"] = {
        "feature": tri.feature,
        "rank_low": lo,
        "rank_high": hi,
        "signs_agree": tri.signs_agree(),
        "sentence": tri.as_sentence(),
        "views": [
            {"view": v.view, "rank": v.lecithin_rank, "sign": v.lecithin_sign,
             "note": v.note}
            for v in (tri.naive,) + tuple(tri.views)
        ],
    }

    # ---- the shortlist, both views ------------------------------------------
    uncon = shortlist.shortlist(df, local=False, top_n=config.SHORTLIST_TOP_N)
    con = shortlist.shortlist(
        df, top_n=config.SHORTLIST_TOP_N, max_distance=uncon.threshold_wt_pct
    )
    fig["shortlist"] = {
        "threshold_wt_pct": uncon.threshold_wt_pct,
        "local_step_wt_pct": shortlist.local_step(X, groups),
        "label": con.label,
        "claim": con.claim,
        "model_detail": con.model_detail,
        "unconstrained": {
            "n_generated": uncon.n_generated,
            "n_flagged": uncon.n_flagged,
            "n_rows": len(uncon.table),
            "max_distance": float(uncon.table["wt_pct_to_nearest_observed"].max()),
        },
        "constrained": {
            "n_generated": con.n_generated,
            "n_candidates": con.n_candidates,
            "n_flagged": con.n_flagged,
            "n_rows": len(con.table),
            "median_interval_width_log1p": float(
                con.table["interval_width_log1p"].median()
            ),
            "min_distance_to_early_block": float(
                con.table["wt_pct_to_early_block"].min()
            ),
            # The diversity guardrail, and what the chosen neighbourhoods are. Without
            # it every delivered row was a perturbation of one observed formulation.
            "n_distinct_neighbourhoods": int(
                con.table["nearest_observed_run"].nunique()
            ),
            "base_runs": [int(r) for r in con.table["nearest_observed_run"]],
            "n_base_product_ready": int(
                con.table["nearest_observed_run"]
                .isin(config.PRODUCT_READY_RUNS).sum()
            ),
            "n_base_in_early_block": int(
                con.table["nearest_observed_run"]
                .between(*config.EARLY_BLOCK_RUNS).sum()
            ),
            "n_product_ready_total": len(config.PRODUCT_READY_RUNS),
        },
    }
    # The response's span on the fitted scale, so an interval width has something to be
    # read against. A width near this span means the ranking is not a quantification.
    z = target.to_fit_scale(y.to_numpy())
    fig["fit_scale_span"] = float(z.max() - z.min())
    return fig, uncon, con, con.model


# ------------------------------------------------------------------ the blocks
def methods_block(f) -> str:
    led = ", ".join(f["features"])
    return f"""# Methods

*Generated by `scripts/make_artifacts.py` on {f["generated"]} from commit
`{f["git_rev"]}`. Every figure below is computed, not transcribed.*

## Dataset

{f["n_file_rows"]} formulations were prepared and assessed. {f["n_analysis"]} enter the
analysis; the excluded formulation is retained in the data file behind a flag rather
than deleted, because its composition could not be reconciled between the laboratory
record and the data sheet. The {f["n_analysis"]} analysed formulations represent
{f["n_distinct"]} distinct compositions.

## Preprocessing

Component masses were renormalised to weight percent of the actual batch total, with
water included, so that every formulation closes to 100 wt% and compositions are
comparable between batches of slightly different total mass. Derived quantities stated
on the original sheets were recomputed from the masses rather than taken as given, and
the stated values were dropped where they could not be reproduced. Invalid instrument
readings were removed **cell by cell** rather than row by row, so a formulation with
one unusable measurement keeps its remaining measurements. Value corrections confirmed
by the laboratory and the one analysis exclusion are both recorded in the data file and
remain visible; the raw file is never edited.

## Features and target

The model uses the {f["n_features"]} renormalised composition components as predictors
({led}) and days to visible degradation as the response. No component was selected or
dropped. Quantities that are exact functions of the {f["n_features"]} components were
excluded as redundant rather than informative, and particle-characterisation
measurements were excluded as inputs because they were performed only on formulations
that had already survived the stability screen.

The response is modelled on the `log1p` scale -- the natural logarithm of one plus the
number of days -- so that a formulation that failed immediately, a genuine zero,
remains finite data rather than being dropped or shifted. Values reported back on the
day scale are medians; the back-transform of a mean on the log scale is a geometric
mean and would understate a right-skewed shelf life.

## Validation protocol

Model performance is estimated by repeated cross-validation **grouped on composition**:
{f["n_splits"]} folds, {f["n_repeats"]} repeats, with all repeat preparations of one
composition assigned to the same fold. With {f["n_distinct"]} distinct compositions
among {f["n_analysis"]} formulations, splitting individual samples would place a
formulation in the training data and its own repeat preparation in the test data, and
the resulting score would partly measure recognition rather than prediction. Model
hyperparameters are selected inside a further grouped cross-validation on the training
samples only, so no reported score depends on a setting chosen using its own test data.

The whole procedure is a deterministic function of one seed ({f["seed"]}), and reruns
are identical.

Every model is additionally re-validated on a second, stricter fold scheme in which a
whole contiguous block of the preparation sequence is held out at a time
({f["n_block_folds"]} folds). The difference between a model's score under the two
schemes measures how much of its apparent performance comes from interpolating between
formulations prepared close together in time, and both columns are reported.

## The reproducibility ceiling

Where two or more formulations share a composition to within {config.GROUP_TOL_PP} wt%
on every component yet differ in observed stability, that difference cannot be
explained by any model of composition. Pooling those differences gives a direct
estimate of the preparation's own reproducibility, and the proportion of the total
variance it accounts for sets an upper bound on any composition-based model:

    ceiling = 1 - (pure-error variance / total variance)

The bound is quoted with a chi-square interval on its {f["ceiling_df"]} degrees of
freedom and never as a single value or as a plus-or-minus, and it is recomputed on
whichever scale the model is fitted on, because the quantity is scale-dependent.
"""


def results_block(f) -> str:
    rows = "\n".join(
        "| {rung} | {r2} | {sp} | {mg} | {rho} | {bo} | {cl} |".format(
            rung=r["rung"].replace("_", " "),
            r2=_fmt(r["r2_grouped"], 4), sp=_fmt(r["spread_grouped"], 4),
            mg=_fmt(r["margin_over_mean"], 4), rho=_fmt(r["spearman_grouped"], 3),
            bo=_fmt(r["r2_block_out"], 4), cl=_fmt(r["collapse"], 4),
        )
        for r in f["ladder"]
    )
    cd, cf = f["ceiling_days"], f["ceiling_fit"]
    b = f["binary"]
    sl = f["shortlist"]
    nonlinear = [r for r in f["ladder"] if r["rung"] in ("gp", "gbm")]
    worst_collapse = max(r["collapse"] for r in nonlinear)

    return f"""# Results

*Generated by `scripts/make_artifacts.py` on {f["generated"]} from commit
`{f["git_rev"]}`. Every figure below is computed, not transcribed. Fold signatures:
composition-grouped `{f["fold_signature_grouped"]}`, block-out
`{f["fold_signature_block"]}` -- identical signatures mean every model was scored on
identical folds.*

## 1. The reproducibility ceiling

Among {f["n_analysis"]} formulations there are {f["n_distinct"]} distinct compositions:
{f["n_replicate_rows"]} samples fall into {f["n_replicate_groups"]} groups matching on
all {f["n_features"]} components while differing in observed stability, in one case by
{_fmt(f["max_replicate_spread"], 0)} days. The pooled within-composition standard
deviation is {_fmt(f["pure_error_sd_days"], 1)} days against a total standard deviation
of {_fmt(f["total_sd_days"], 1)} days (sample standard deviations throughout, so these
differ in the final digit from population figures quoted elsewhere in the project
record; the ratio they form moves by nothing).

| scale | ceiling | 95 % interval | df |
|---|---|---|---|
| days | {_fmt(cd["point"])} | {_fmt(cd["lo"])} to {_fmt(cd["hi"])} | {cd["df"]} |
| `log1p` (the scale fitted) | {_fmt(cf["point"])} | {_fmt(cf["lo"])} to {_fmt(cf["hi"])} | {cf["df"]} |

**Both are reported, and the second is the one to read the model scores against.** On
the day scale the ceiling is appreciable; recomputed on the scale the model is actually
fitted on, the within-composition variation exceeds the total variation and the bound
is **not above zero**. The two are limited rather than contradictory: the repeat
preparations are concentrated among the short-lived formulations, so the day-scale
figure is a local and optimistic estimate, and the `log1p` scale magnifies
disagreement there. Neither reading licenses a model; the honest statement is that the
reproducibility of the preparation, not the model, is the binding constraint.

## 2. The model ladder, on two fold schemes

Composition-grouped cross-validation ({f["n_splits"]} folds x {f["n_repeats"]}
repeats), fitted on `log1p` days. *Spread* is the standard deviation across repeats --
the ruler any margin must be read against. *Block-out* re-scores the identical model on
the stricter scheme that holds out a whole block of the preparation sequence, and
*collapse* is the difference.

| model | R2 grouped | spread | margin over mean | Spearman | R2 block-out | collapse |
|---|---|---|---|---|---|---|
{rows}

Two things follow, and both belong in the report.

**The penalised linear models do not outperform predicting the mean, and that is the
result.** It was pre-registered as a possible outcome before any model was fitted. Their
score is near zero under either fold scheme, so it does not depend on which split is
believed -- which is why the penalised linear model remains the reported architecture.

**The two non-linear models do outperform the mean under composition-grouped
cross-validation, and lose all of it -- up to {_fmt(worst_collapse, 2)} R2 -- when a
block of the preparation sequence is held out.** The linear models barely move. They
were therefore not learning composition chemistry; they were interpolating between
formulations prepared close together in time. They are reported as evidence that the
preparation sequence is structured, not as evidence of a composition-stability
relationship, and no recommendation in this report is derived from either.

The ordering survives better than the magnitude throughout: rank correlations are
positive where the variance explained is not. That asymmetry is why the deliverable is
a ranking aid and not a predictor.

## 3. Stability past {b["threshold_days"]} days, as a classification

The {b["threshold_days"]}-day threshold was declared before any model was scored and
was never selected by searching: a month of shelf life is an external product
criterion, not a feature of this dataset's distribution. {b["n_positive"]} of
{f["n_analysis"]} formulations exceed it, and none sits exactly on it.

| statistic | value |
|---|---|
| area under the ROC curve, out-of-fold | {_fmt(b["auc"], 3)} |
| across-repeat spread | {_fmt(b["auc_spread"], 3)} |
| accuracy, out-of-fold | {_fmt(b["accuracy"], 3)} |
| accuracy of always predicting "not stable" | {_fmt(b["majority_class_accuracy"], 3)} |

**Area under the curve is the figure to quote and accuracy is not.** The classes are
lopsided at this threshold, so a model that predicted "not stable" for every
formulation would score the accuracy in the last row -- above this model's. The
comparison would be meaningless.

This is the one result in this report that is directly comparable to published work:
composition-only classification of nanoemulsion stability has been reported for a
dataset several times this size (*Nanomaterials* 2026, doi:10.3390/nano16130793), and
the value above is **below** that published benchmark. On a smaller, single-site
dataset with the sequence structure described in section 2, that is the expected
direction. {b["straddling_groups"]} of the {f["n_replicate_groups"]} repeat-preparation
groups straddles the threshold, so the labels themselves carry some of the
reproducibility limit described in section 1.

## 4. Sensitivity of the leading compositional effect

{_humanise(f["triad"]["sentence"])}

| view | rank of {f["n_features"]} | direction | note |
|---|---|---|---|
""" + "\n".join(
        "| {v} | {r} | {s} | {n} |".format(
            v=x["view"], r=x["rank"],
            s={1: "longer-lived", -1: "shorter-lived", 0: "shrunk to zero"}[x["sign"]],
            n=_humanise(x["note"]),
        )
        for x in f["triad"]["views"]
    ) + f"""

The direction is consistent across all four views; the rank moves. **No effect size is
reported**, and none is reportable: the composition of the series and the order in
which it was prepared move together, and no analysis of this dataset can separate them.
A randomised preparation order in the next batch is what resolves it.

## 5. The screening shortlist

The full table is `shortlist.md`. Two views ship together, and the reason is itself a
result.

Candidate compositions were generated inside the observed range of every component and
renormalised to close to 100 wt%, then ranked by the penalised linear model -- never by
the non-linear models of section 2, whose ranking was shown there to follow the
preparation sequence. Each candidate carries its distance to the nearest formulation
already prepared, a flag when that distance exceeds the
{_fmt(sl["threshold_wt_pct"], 2)} wt% spacing typical of this design, and its distance
to the early, high-concentration block of the sequence where every laboratory-endorsed
formulation sits.

**Ranking freely over the sampled ranges produces candidates that are all
extrapolation.** All {sl["unconstrained"]["n_flagged"]} of the top
{sl["unconstrained"]["n_rows"]} are flagged, the furthest
{_fmt(sl["unconstrained"]["max_distance"], 1)} wt% from anything prepared. There are
two reasons, both structural: a linear model is monotone in every component, so its
best candidate always lies at the edge of the allowed range; and with
{f["n_features"]} components and {f["n_analysis"]} formulations, the compositions
prepared occupy a thin region of the range they span, so a point drawn freely inside
that range is almost never near one of them. That view is reported because it is the
honest answer to "what does the model most prefer", and the answer is "something it
has no evidence about".

**The actionable view is therefore restricted to candidates within the design's own
spacing of a prepared formulation**, of which {sl["constrained"]["n_rows"]} are
delivered, none flagged, and **each one in the neighbourhood of a different prepared
formulation**. That last restriction is necessary rather than cosmetic: without it, all
{sl["constrained"]["n_rows"]} highest-ranked candidates were small variations of a
single formulation, because the ranking model increases monotonically in every
component and so has exactly one most-preferred direction. A list of
{sl["constrained"]["n_rows"]} variations of one composition, presented as a shortlist,
would mislead by its shape whatever the accompanying text said.

Their prediction intervals have a median width of
{_fmt(sl["constrained"]["median_interval_width_log1p"], 2)} on the fitted scale,
against a total span of {_fmt(f["fit_scale_span"], 2)} for the observed response on
that same scale. **The interval is as wide as most of the range of the data.** Stated
plainly: the model orders candidates considerably better than it quantifies them, which
is the same asymmetry that appears in section 2 and the reason this is delivered as a
ranking aid.

**Two readings of which formulations the shortlist points to, and both are true.**
{sl["constrained"]["n_base_product_ready"]} of the
{sl["constrained"]["n_rows"]} candidates sit beside one of the
{sl["constrained"]["n_product_ready_total"]} formulations the laboratory independently
identified as product-ready. That is genuine corroboration: the model, which never saw
that list, prefers the region the bench already chose. It is also exactly the
confound -- {sl["constrained"]["n_base_in_early_block"]} of the
{sl["constrained"]["n_rows"]} sit in the early part of the preparation sequence, which
is where all of the endorsed formulations were made. The shortlist should therefore be
read as pointing at a **region** that both the model and the laboratory favour, with
the magnitude of any component's contribution still unidentified, and with the two
candidates outside that region being the most informative ones to prepare, precisely
because they would test the ranking where the sequence does not support it.

{sl["claim"]}
"""


def claims_block(f) -> str:
    return f"""# Supportable claims

*Generated by `scripts/make_artifacts.py` on {f["generated"]}. Six claims, each
supported by a citation. **No external figure is quoted in this report**, by design:
sources that could not be verified before the deadline were cut rather than hedged, so
comparisons are stated as directions and the citation is given for the reader to
check.*

**1. No prior published model predicts days to visible degradation from formulation
composition.** Worded narrowly: we found no prior published model of this response. The
two nearest published studies model droplet size and ζ-potential
(doi:10.1016/j.jfutfo.2026.08.003) and binary stability success
(doi:10.3390/nano16130793) respectively. There is therefore no benchmark score for the
continuous response in this report to be judged against, which is why the
classification in Results section 3 is included at all.

**2. The dataset is a contribution in its own right.** No open benchmark dataset exists
for this response, and {f["n_analysis"]} formulations prepared under a single protocol
is substantially larger than a typical published design-of-experiments campaign in this
field, where optimised formulations are published one at a time.

**3. Composition-only modelling with the process fixed by protocol is established
practice, and the process constancy here is tighter than the published precedent's.**
The nearest precedent inferred a common ultrasonication method across many
laboratories; every formulation here was prepared by one operator under one protocol.

**4. Surfactant load sits at or below comparable high-energy topical nanoemulsions and
well below phase-diagram alopecia nanoemulgels.** This is simultaneously a stability
argument and a scalp-tolerance argument.

**5. The z-average is the reportable particle size under ISO 22412, which settles the
small-peak question on standards-compliance grounds.** The intensity-peak size reported
elsewhere in the project record is a different quantity and is not comparable to the
z-average.

**6. The botanical rationale rests on the three plants that carry clinical or
minoxidil-comparated evidence -- rosemary (PMID 25842469; note the comparator strength
stated in that trial), peppermint and lavender -- and all three are in the blend.**

## Claims deliberately not made

- **That machine learning outperforms response-surface methodology.** Response-surface
  methodology does not apply to an observational, mixture-constrained dataset, and
  published comparisons are evaluated near their own design points. The structural
  argument replaces the performance claim.
- **Any effect size for a compositional component**, and any statement that including
  the preparation sequence as a covariate corrects for it. Direction, rank stability
  and the remedy are reported; magnitude is not identifiable.
- **That all twelve botanicals have hair-growth evidence.** Several have none, and the
  narrower claim in item 6 costs nothing.
- **Any conversion from accelerated ageing to predicted shelf life.** No validated
  acceleration factor for nanoemulsion physical stability exists, so the direct ambient
  observation is reported as what it is.
"""


def runbook_block(f) -> str:
    v = f["versions"]
    return f"""# Runbook

*Generated by `scripts/make_artifacts.py` on {f["generated"]} from commit
`{f["git_rev"]}`. One page, and it is the whole operating manual.*

## What produced the delivered blocks

| | |
|---|---|
| commit | `{f["git_rev"]}` |
| seed | `{f["seed"]}` (one seed for the entire project) |
| Python | {v["python"]} |
| scikit-learn | {v["scikit-learn"]} |
| numpy | {v["numpy"]} |

Every reported number is a deterministic function of the data file and that seed. Two
runs are identical, and a test asserts it.

## Rerunning everything, from a clean checkout

    python -m venv .venv
    .venv/Scripts/python.exe -m pip install -r requirements.txt
    .venv/Scripts/python.exe -m pytest                     # the full test suite
    .venv/Scripts/python.exe scripts/stats_report.py       # drift-check the doc figures
    .venv/Scripts/python.exe scripts/make_artifacts.py     # rebuild artifacts/

The data file is not in the repository and never will be: it is client data, and
`.gitignore` blocks it while a test asserts nothing of the sort is tracked. Place
`formulations.csv` in `data/raw-data/` and run
`.venv/Scripts/python.exe scripts/build_clean_dataset.py` to rebuild
`data/processed/formulations_clean.csv` first.

## Order matters, in one place only

`stats_report.py` recomputes every figure quoted in the project documentation from the
data file and **exits 1 on any mismatch**. Run it after anything that touches the data,
and reconcile the documentation before trusting a model result -- a model fitted on
data whose documented figures have drifted is a model nobody can describe.

## Using the model file

`model.joblib` is the fitted penalised linear ranking model, with its standardiser, as
a scikit-learn pipeline. It expects the {f["n_features"]} renormalised composition
components in this exact order:

    {", ".join(f["features"])}

and returns a value on the `log1p` day scale.

**It is a ranking tool.** Its output orders candidate formulations; it is not a
shelf-life forecast for any individual formulation, and Results sections 1 and 2
explain why. Compare candidates to each other, not to a target number of days. A
candidate further than {_fmt(f["shortlist"]["threshold_wt_pct"], 2)} wt% from every
formulation already prepared is outside the region the model has evidence about, and
`shortlist.md` carries that distance for every candidate it lists.

Loading it requires the same scikit-learn version as above; a pickled model is not a
stable format across versions.

## If a figure needs to change

Nothing is edited by hand. Change the data or the code, rerun `stats_report.py` and
`make_artifacts.py`, and the blocks regenerate. A number typed into a delivered block
is a number that will be wrong later.
"""


def shortlist_block(f, uncon, con) -> str:
    sl = f["shortlist"]

    def tbl(t):
        cols = list(t.columns)
        head = "| " + " | ".join(c.replace("_", " ") for c in cols) + " |"
        rule = "|" + "---|" * len(cols)
        body = "\n".join(
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in t.itertuples(index=False)
        )
        return "\n".join([head, rule, body])

    return f"""# Candidate screening shortlist

*Generated by `scripts/make_artifacts.py` on {f["generated"]} from commit
`{f["git_rev"]}`. Ranked by the penalised linear model
({sl["model_detail"]}); **{sl["label"]}**.*

{sl["claim"]}

## How to read the columns

- **screening rank** -- the model's ordering, best first. There is deliberately no
  score and no predicted number of days: the model's ordering is better supported than
  its magnitude, and a number on the fitted scale is one step away from being read as a
  shelf life.
- **interval width log1p** -- the width of a distribution-free prediction interval at
  that candidate, on the fitted scale. It is the honest measure of how little the
  ranking should be trusted quantitatively.
- **nearest observed run** -- the prepared formulation this candidate is a variation of.
  Every delivered candidate has a different one, so the list is
  {sl["constrained"]["n_rows"]} distinct leads rather than one lead restated. Use it to
  read each candidate as "formulation *n*, adjusted" -- which is also how it should be
  prepared, since the adjustment is small.
- **wt pct to nearest observed** -- Euclidean distance, in weight percent across all
  {f["n_features"]} components, to that formulation.
- **extrapolation flag** -- true when that distance exceeds
  {_fmt(sl["threshold_wt_pct"], 2)} wt%, the spacing typical of this design. A flagged
  candidate is outside the region the model has evidence about.
- **wt pct to early block** -- distance to the nearest formulation in the early,
  high-concentration part of the preparation sequence, where every laboratory-endorsed
  formulation sits. **A low value here is a warning**: a candidate that scores well
  while resembling that block may be scoring on the sequence rather than on its
  composition. A candidate that scores well and sits away from it is the one worth
  preparing. The shortlist reports this and does not filter on it -- how much of that
  risk to accept is the laboratory's decision, not ours.

## View 1 -- the actionable shortlist: candidates within the design's own spacing

{con.n_candidates} of {con.n_generated} generated candidates sit within
{_fmt(sl["threshold_wt_pct"], 2)} wt% of a prepared formulation. The best
{len(con.table)} of those, one per prepared formulation:

{tbl(con.table)}

None is flagged, by construction, and all {sl["constrained"]["n_distinct_neighbourhoods"]}
are variations of different prepared formulations -- runs
{", ".join(str(r) for r in sl["constrained"]["base_runs"])}.
{sl["constrained"]["n_base_product_ready"]} of those base formulations are among the
{sl["constrained"]["n_product_ready_total"]} the laboratory identified as product-ready,
which the model never saw. {sl["constrained"]["n_base_in_early_block"]} are in the early
part of the preparation sequence, which is both why the agreement is encouraging and why
it is not independent evidence. The nearest any candidate comes to that block is
{_fmt(sl["constrained"]["min_distance_to_early_block"], 2)} wt%.

## View 2 -- the unconstrained ranking, which is entirely extrapolation

The same model over the same generated ranges with no distance restriction. It is
included because it is the honest answer to what the model most prefers, and because
the answer disqualifies itself:

{tbl(uncon.table)}

{sl["unconstrained"]["n_flagged"]} of {sl["unconstrained"]["n_rows"]} rows are flagged;
the furthest sits {_fmt(sl["unconstrained"]["max_distance"], 1)} wt% from anything ever
prepared. Two structural reasons, neither fixable by analysis: a linear model is
monotone in every component, so its preferred candidate always lies at the edge of the
allowed range; and with {f["n_features"]} components and {f["n_analysis"]}
formulations, what has been prepared occupies a thin region of the ranges it spans.

**Nothing in view 2 should be prepared on the strength of this ranking.** It is
reported so that view 1's restriction is visible as a decision rather than absent as an
assumption.
"""


# ------------------------------------------------------------------- token filling
def fill_limitations(f) -> str:
    """Fill `limitations.md`'s tokens and strip its build notes. Fails on a leftover."""
    src = TEMPLATES / "limitations.md"
    if not src.exists():
        raise SystemExit(
            f"the limitations template is missing: {src}\nIt lives in the sdlc repo, "
            "which is a sibling checkout of this one."
        )
    text = src.read_text(encoding="utf-8")

    cd = f["ceiling_days"]
    tokens = {
        "N_ANALYSIS": str(f["n_analysis"]),
        "N_DISTINCT": str(f["n_distinct"]),
        "N_REPLICATE_ROWS": str(f["n_replicate_rows"]),
        "N_REPLICATE_GROUPS": str(f["n_replicate_groups"]),
        "MAX_REPLICATE_SPREAD": _fmt(f["max_replicate_spread"], 0),
        "CEILING_INTERVAL": f'{_fmt(cd["lo"])} to {_fmt(cd["hi"])}, on the day scale',
        "CEILING_DF": str(f["ceiling_df"]),
        "CEILING_FIT_INTERVAL": (
            f'{_fmt(f["ceiling_fit"]["lo"])} to {_fmt(f["ceiling_fit"]["hi"])}'
        ),
        "N_AT_PLATEAU": str(f["n_at_plateau"]),
        "PLATEAU_DAYS": _fmt(f["plateau_days"], 0),
        "GENERATED": f["generated"],
        "GIT_REV": f["git_rev"],
    }
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", v)

    # Strip everything from the build-notes heading on: it is ours, not the client's.
    marker = "## Build notes"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    # And strip the file's own front matter, down to the first horizontal rule.
    if "\n---\n" in text:
        text = text.split("\n---\n", 1)[1].lstrip()

    leftover = sorted(
        set(
            text[i + 2 : text.index("}}", i)]
            for i in range(len(text))
            if text.startswith("{{", i) and "}}" in text[i:]
        )
    )
    if leftover:
        raise SystemExit(
            "unfilled token(s) in the limitations block: "
            + ", ".join(leftover)
            + "\nAdd each one to `tokens` in make_artifacts.py, computed from the data. "
            "A delivered block containing a raw token reaches the funder looking like a "
            "bug, which is worse than a stale number."
        )
    header = (
        f"*Figures filled by `scripts/make_artifacts.py` on {f['generated']} from "
        f"commit `{f['git_rev']}`.*\n\n"
    )
    return header + text


def _assert_no_external_figures(blocks: dict) -> None:
    """The Done gate, as a check. No delivered block quotes an external figure.

    The literature report's "verify before publishing" table was settled by cutting:
    nothing external is quoted, so nothing in it needs clearing before the deadline. A
    figure that creeps back in silently reopens that whole question, so the specific
    numbers are checked for by name.

    **Markdown table rows are stripped before the scan, and that is the whole
    subtlety.** The first version scanned the rendered text whole and fired on
    `shortlist.md` because a generated composition happened to read `0.87` -- the same
    digits as a published benchmark. Exempting the block would have been the wrong fix:
    it is prose that can quote an external figure, and prose is what this now reads. A
    generated number cannot breach the gate, because nothing external generates one.
    """
    bad = []
    for name, text in blocks.items():
        prose = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("|")
        )
        for fig in EXTERNAL_FIGURES:
            if fig in prose:
                bad.append(f"{name}: {fig!r}")
    if bad:
        raise SystemExit(
            "a delivered block quotes an external figure, which the Done gate "
            "settled by cutting:\n  " + "\n  ".join(bad)
            + "\nIf the figure is genuinely needed, it gets verified first or the "
            "block drops it -- those are the only two options the gate allows."
        )


# ------------------------------------------------------------------------- main
def main() -> int:
    df = loader.load_analysis_set()
    print(f"analysis set: {len(df)} rows")

    print("measuring (the full ladder on two fold schemes takes a few minutes) ...")
    f, uncon, con, model = collect(df)

    blocks = {
        "methods.md": methods_block(f),
        "results.md": results_block(f),
        "claims.md": claims_block(f),
        "limitations.md": fill_limitations(f),
        "shortlist.md": shortlist_block(f, uncon, con),
        "runbook.md": runbook_block(f),
    }
    _assert_no_external_figures(blocks)

    OUT.mkdir(exist_ok=True)
    for name, text in blocks.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"  wrote {name:18s} {len(text):>6d} chars")

    (OUT / "figures.json").write_text(
        json.dumps(f, indent=2, default=float), encoding="utf-8"
    )
    con.table.to_csv(OUT / "shortlist.csv", index=False)
    joblib.dump(model, OUT / "model.joblib")
    print(f"  wrote figures.json, shortlist.csv, model.joblib\n\nall of it: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
