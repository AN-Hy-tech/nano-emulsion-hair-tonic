"""Paths, the feature contract, the data-contract counts and the seed policy.

One place for everything the pipeline agrees on, so a change lands in one file.

The counts below are **owned by** `planning/project.md` and independently
drift-checked against the CSV by `scripts/stats_report.py`. They are restated here
because code needs them as constants, not as prose; two independent assertions of
the same documented number is the drift check working. Never edit one alone.
"""
from pathlib import Path
import os
import random

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV = REPO_ROOT / "data" / "processed" / "formulations_clean.csv"

# ---- seed policy -------------------------------------------------------------
# One seed for the whole project. Every splitter, shuffle and estimator that takes
# a random_state takes this one, so a rerun is bit-for-bit reproducible (test 14).
SEED = 20261008

def set_seeds(seed: int = SEED) -> int:
    """Seed every global RNG we can reach. Returns the seed, for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep; guard only so config stays importable
        pass
    return seed

# ---- the feature contract ----------------------------------------------------
# The 11 renormalised composition columns, and nothing else. p/n = 11/109 ~ 0.10.
# Correlated components are reported as groups; no selection, no single "driver".
FEATURES = [
    "oil_blend_pct",
    "span80_pct",
    "lecithin_pct",
    "tween80_oil_pct",
    "extract_pct",
    "tween80_water_pct",
    "glycerol_pct",
    "pg_pct",
    "xanthan_pct",
    "guar_pct",
    "water_pct",
]

TARGET = "stability_days"

# Columns that must never reach the feature matrix, each for a stated reason.
#   hlb_client          — no formula reproduces the stated column (project.md)
#   hlb_calc, surfactant_total_pct — exact functions of FEATURES; redundant, not new
#   ph                  — measured, swept, no signal on 109 rows
#   dls_measured        — encodes the outcome: survivors only were ever measured
#   z_average_nm, pdi, peak_size_nm, zeta_mv, mobility_cm2_vs — survivors-only,
#                         optimisation data, never inputs; zeta is characterisation
#   stability_quality, in_analysis_set, renorm_total_g, quarantine_reason — bookkeeping
NEVER_A_FEATURE = [
    "num",
    "hlb_client",
    "hlb_calc",
    "surfactant_total_pct",
    "ph",
    "dls_measured",
    "z_average_nm",
    "pdi",
    "peak_size_nm",
    "zeta_mv",
    "mobility_cm2_vs",
    "stability_days",
    "stability_quality",
    "in_analysis_set",
    "renorm_total_g",
    "quarantine_reason",
]

# ---- data-contract counts (see the module docstring) -------------------------
N_FILE_ROWS = 110            # all rows stay in the file; exclusion is a flag
N_ANALYSIS_ROWS = 109        # row 110 excluded, final
N_DISTINCT_COMPOSITIONS = 101
N_REPLICATE_GROUPS = 7
N_REPLICATE_ROWS = 15

CLOSURE_TOL_PP = 1e-3        # renormalisation closes to 100 wt% this tightly
GROUP_TOL_PP = 0.05          # two rows are the same composition within this, per component

# ---- cross-validation protocol ----------------------------------------------
# Repeated GroupKFold grouped on composition (see workcycle.md, Decisions). Not a
# derived figure: a protocol choice. 5 folds leaves ~20 compositions per test fold
# on 101 groups; 10 repeats make the across-repeat spread readable, which is what
# test 15 reports the mean-baseline margin against. A full run takes seconds.
N_SPLITS = 5
N_REPEATS = 10

# The ridge alpha the leakage comparison (test 7) scores with. The real alpha is
# tuned inside nested CV in PR4; this one only has to be the *same* on both sides
# of the comparison, so the delta measures the leak and not the tuning.
LEAKAGE_PROBE_ALPHA = 1.0

# ---- the model ladder (PR4) --------------------------------------------------
# Protocol choices, not derived figures. The ten-point alpha grid is the one named
# in workcycle.md; it is searched in the *inner* loop only, on training rows, so the
# outer grouped folds never see a tuned hyperparameter chosen with their own data.
RIDGE_ALPHA_GRID = tuple(10.0 ** (-3 + 6 * i / 9) for i in range(10))
L1_RATIO_GRID = (0.1, 0.5, 0.9)
N_INNER_SPLITS = 3           # inner grouped folds for nested tuning; 3 on ~80 train groups

# ---- the binarised rung (PR4) ------------------------------------------------
# 30 days, declared 2026-10-08 (Albert) before anything was scored, and never
# searched over. A month of shelf life is an external product criterion; it is not
# read off our own target distribution. See workcycle.md, Decisions.
BINARY_THRESHOLD_DAYS = 30

# A data-contract count, owned by planning/project.md and drift-checked by
# scripts/stats_report.py: rows with stability_days > BINARY_THRESHOLD_DAYS.
N_STABLE_PAST_THRESHOLD = 19

# ---- the confound triad (PR4) ------------------------------------------------
# Run order is split into this many equal blocks for the within-block view, and the
# time-split view trains on the first half of the run order and tests on the second.
N_RUN_BLOCKS = 3
