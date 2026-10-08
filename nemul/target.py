"""The target transform, and the only sanctioned way back from it.

`stability_days` is a count of days with real zeros -- 0 d means the formulation
failed immediately, not that it is missing -- so the transform has to map 0 to a
finite value. `log1p` does; plain `log` sends it to -inf and Box-Cox needs a shift
chosen from the data, which is a fitted parameter nobody would report. Both are
refused by name here rather than merely left unused.

Coming back is the part that goes wrong quietly. `expm1` of a mean on the log scale
is **not** the mean in days; it is the geometric mean, which under-reports a
right-skewed shelf life. So `from_fit_scale` returns a labelled result and the mean
path raises. The deliverable quotes medians.
"""
from typing import NamedTuple

import numpy as np

FIT_SCALE = "log1p"

MEDIAN_LABEL = "median (geometric on the data scale), not the mean"


class BackTransformed(NamedTuple):
    """A number that carries what kind of average it is.

    Deliberately not a bare float: a value tagged `median` cannot be pasted into a
    sentence that says `mean` without someone reading this label first.
    """
    value: np.ndarray
    label: str
    scale: str = "days"


def to_fit_scale(y, how: str = FIT_SCALE) -> np.ndarray:
    """`log1p(y)`. The one transform; `how` exists only to refuse the others."""
    if how != FIT_SCALE:
        raise ValueError(
            f"{how!r} is not the project transform: log1p is, because a true-zero "
            "stability reading is real data. Plain log drops it to -inf and Box-Cox "
            "fits a shift parameter from the target, which is not reportable."
        )
    arr = np.asarray(y, dtype=float)
    if (arr < 0).any():
        raise ValueError("negative stability_days is a data error, not a transform case")
    return np.log1p(arr)


def from_fit_scale(z) -> BackTransformed:
    """`expm1`, labelled. The median on the data scale, never the mean."""
    return BackTransformed(np.expm1(np.asarray(z, dtype=float)), MEDIAN_LABEL)


def mean_on_data_scale(z):
    """Closed on purpose. `expm1` of a log-scale mean is the geometric mean."""
    raise NotImplementedError(
        "there is no mean-in-days path: expm1 of a log1p mean is the geometric mean "
        "and under-reports a right-skewed shelf life (Jensen). Report the median via "
        "from_fit_scale, or compute a mean in days directly from untransformed rows "
        "and label it as such."
    )
