"""The binarised rung: "stable past 30 days", the one externally comparable number.

Our regression R2 compares to nothing -- no published work regresses days-to-failure
for these systems. A composition-only classifier does have an anchor: AUC 0.87 on
n = 496 in the literature report. So this rung exists to be compared, and it is a
comparator and not the headline.

Two rules, both of them about not manufacturing the benchmark:

- **The threshold is 30 days, declared before anything was scored, never searched.**
  `labels` refuses any other value by name. A threshold chosen after seeing the
  scores would invent the number it is supposed to supply.
- **Compare on AUC, not accuracy.** 19 of 109 rows are positive, so a model that
  predicts "unstable" for everything scores 83 % accuracy. Accuracy is computed here
  only so the report can say why it is not the figure quoted.

It runs on the identical grouped folds and the identical feature matrix as the
regression ladder (test 17), and the AUC is pooled within a repeat for the same
reason the R2 is -- about four positives land in a test fold, and a per-fold AUC on
four positives is noise.

One honest wrinkle worth carrying into the report: one replicate group straddles the
threshold, so even a well-placed threshold inherits some label noise from the same
replicate spread the noise ceiling measures.
"""
from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config, ladder, splits

THRESHOLD = config.BINARY_THRESHOLD_DAYS

# The external anchor from the literature report, for the comparison line only. Not
# a figure we computed, and not recomputable here; it is cited, never recalculated.
ANCHOR_AUC = 0.87
ANCHOR_N = 496


class BinaryResult(NamedTuple):
    auc: float
    auc_by_repeat: np.ndarray
    spread: float
    accuracy: float
    threshold_days: int
    n_folds: int
    n_features: int
    fold_signature: str
    straddling_groups: int
    detail: str = ""


def labels(y, threshold=None) -> np.ndarray:
    """1 if the formulation outlived the declared threshold, else 0.

    Strictly greater than: "stable past 30 days". No row sits exactly at 30 d, so the
    boundary convention changes nothing here -- it is fixed anyway, because a
    convention settled by the data is a threshold chosen by the data.
    """
    if threshold is not None and threshold != THRESHOLD:
        raise ValueError(
            f"the threshold is declared, not passed: {THRESHOLD} d, fixed on "
            "2026-10-08 before anything was scored. A threshold read off the target "
            "at scoring time manufactures the benchmark it is meant to supply."
        )
    return (np.asarray(y, dtype=float) > THRESHOLD).astype(int)


def straddling_groups(y, groups) -> int:
    """Replicate groups whose rows fall on both sides of the threshold.

    Label noise from the same source the noise ceiling measures. Reported, not fixed.
    """
    lab = labels(y)
    ga = np.asarray(groups)
    return int(sum(len(set(lab[ga == g].tolist())) > 1 for g in np.unique(ga)))


def classifier():
    """Penalised logistic regression, C tuned in an inner stratified grouped search.

    The linear form matches the regression ladder's, so the two rungs differ in the
    question asked and not in the model class.
    """
    return ladder._Tuned(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
        {"logisticregression__C": [1.0 / a for a in config.RIDGE_ALPHA_GRID]},
        "logit",
        scoring="roc_auc",
        stratified=True,
    )


def validate_binary(X, y, groups, folds=None, threshold=None, estimator=None) -> BinaryResult:
    """Pooled out-of-fold AUC per repeat, on the ladder's own folds.

    `folds` defaults to the same `splits.grouped_folds(seed=config.SEED)` the ladder
    builds, so the default path is already comparable; test 17 passes them in
    explicitly and checks the signatures match.
    """
    if threshold is not None and threshold != THRESHOLD:
        raise ValueError(
            f"the threshold is declared, not passed: {THRESHOLD} d. See labels()."
        )
    Xa = np.asarray(X, dtype=float)
    ga = np.asarray(groups)
    lab = labels(y)
    folds = splits.grouped_folds(ga, seed=config.SEED) if folds is None else folds

    aucs, accs = [], []
    for r in sorted({f.repeat for f in folds}):
        prob = np.full(len(lab), np.nan)
        for f in (f for f in folds if f.repeat == r):
            est = classifier() if estimator is None else ladder._clone(estimator)
            est.fit(Xa[f.train], lab[f.train], groups=ga[f.train])
            prob[f.test] = est.predict_proba(Xa[f.test])[:, 1]
        if np.isnan(prob).any():
            raise ValueError(f"repeat {r} left rows untested; the AUC would be partial")
        aucs.append(float(roc_auc_score(lab, prob)))
        accs.append(float(accuracy_score(lab, (prob > 0.5).astype(int))))
    arr = np.asarray(aucs, dtype=float)

    return BinaryResult(
        auc=float(arr.mean()),
        auc_by_repeat=arr,
        spread=float(arr.std(ddof=1)),
        accuracy=float(np.mean(accs)),
        threshold_days=THRESHOLD,
        n_folds=len(folds),
        n_features=Xa.shape[1],
        fold_signature=ladder.fold_signature(folds),
        straddling_groups=straddling_groups(y, ga),
        detail=f"compare against AUC {ANCHOR_AUC} on n = {ANCHOR_N}; accuracy is not comparable",
    )
