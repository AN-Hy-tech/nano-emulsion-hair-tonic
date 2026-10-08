"""Data-contract tests (PR1, tests 1-5 of planning/workcycle.md).

Each name is the contract. These guard the five ways the analysis set can go wrong
without anything crashing: a deleted row instead of a flag, a composition that no
longer closes, a hole in the feature matrix, a lost replicate group, and a leaked
survivors-only column. A red test here is diagnosed in the loader, never silenced.
"""
import numpy as np
import pandas as pd
import pytest

from nemul import config, loader


@pytest.fixture(scope="module")
def full():
    return loader.load_raw()


@pytest.fixture(scope="module")
def analysis():
    return loader.load_analysis_set()


def test_row_count_after_exclusion(full, analysis):
    """The analysis set is the ANALYSIS_EXCLUDE flag applied, not a deletion."""
    assert len(full) == config.N_FILE_ROWS
    assert len(analysis) == config.N_ANALYSIS_ROWS
    excluded = full.loc[full["in_analysis_set"] == 0, "num"].tolist()
    assert excluded == [110]
    # the excluded row is still in the file, and still carries its reason
    assert full.loc[full["num"] == 110, "quarantine_reason"].iloc[0].strip() != ""


def test_composition_closes(analysis):
    """Every row closes to 100 wt% within tolerance."""
    totals = analysis[config.FEATURES].sum(axis=1)
    off = analysis.loc[(totals - 100).abs() > config.CLOSURE_TOL_PP, "num"].tolist()
    assert off == [], f"rows not closing to 100 wt%: {off}"


def test_feature_matrix_has_no_nan(analysis):
    """The 11 modelling features are complete on every analysis row."""
    X = loader.feature_matrix(analysis)
    assert X.shape == (config.N_ANALYSIS_ROWS, len(config.FEATURES))
    assert list(X.columns) == config.FEATURES
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    y = loader.target(analysis)
    assert len(y) == config.N_ANALYSIS_ROWS
    assert not y.isna().any()


def test_group_count(analysis):
    """Distinct compositions, asserted against the count in planning/project.md."""
    groups = loader.composition_groups(analysis)
    assert len(groups) == config.N_ANALYSIS_ROWS
    assert groups.nunique() == config.N_DISTINCT_COMPOSITIONS
    sizes = groups.value_counts()
    replicates = sizes[sizes > 1]
    assert len(replicates) == config.N_REPLICATE_GROUPS
    assert replicates.sum() == config.N_REPLICATE_ROWS


def test_excluded_columns_absent(analysis):
    """Guards the survivors-only leak: no outcome-dependent column reaches the model."""
    X = loader.feature_matrix(analysis)
    for col in config.NEVER_A_FEATURE:
        assert col not in X.columns, f"{col} leaked into the feature matrix"
    # and nothing stated-but-unreproducible survived the clean build at all
    assert [c for c in analysis.columns if c.endswith("_stated")] == []
