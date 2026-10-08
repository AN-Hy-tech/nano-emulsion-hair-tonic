"""The one way the pipeline reads the data.

Nothing downstream opens the CSV itself. `load_analysis_set` applies the
`in_analysis_set` flag; `feature_matrix` is the only place FEATURES is selected,
so the survivors-only guard (test 5) has a single door to watch.
"""
import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """All 110 rows exactly as the clean build wrote them. No filtering."""
    path = config.CLEAN_CSV if path is None else path
    df = pd.read_csv(path)
    df["quarantine_reason"] = df["quarantine_reason"].fillna("")
    return df


def load_analysis_set(path=None) -> pd.DataFrame:
    """The analysis set: the exclusion flag applied, never a deletion."""
    df = load_raw(path)
    keep = df.loc[df["in_analysis_set"] == 1].reset_index(drop=True)
    return keep


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The 11 composition features, in contract order. The only selection point."""
    return df[config.FEATURES].astype(float)


def target(df: pd.DataFrame) -> pd.Series:
    """Raw `stability_days`. The log1p transform is a separate, labelled step (PR3)."""
    return df[config.TARGET].astype(float)


def composition_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per row: rows agreeing to <0.05 wt% on all 11 components share one.

    Sequential leader clustering in row order, the same definition
    `scripts/stats_report.py` uses to report 101 distinct compositions. These ids are
    what cross-validation groups on — splitting rows instead lets a model score by
    recognising a replicate it already saw in training.
    """
    X = feature_matrix(df).to_numpy()
    leaders, ids = [], []
    for row in X:
        for gid, leader in enumerate(leaders):
            if max(abs(row - leader)) < config.GROUP_TOL_PP:
                ids.append(gid)
                break
        else:
            leaders.append(row)
            ids.append(len(leaders) - 1)
    return pd.Series(ids, index=df.index, name="composition_group")
