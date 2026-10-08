"""Test 18: the repo audit. No client data, and no model file, in the git index.

The repo is private, but private is not the guard -- the guard is that client data
never enters git history at all, because history is the part you cannot take back.
`.gitignore` blocks `data/`, `*.csv`, `*.xlsx` and model files; this test checks the
*index*, which is what `.gitignore` only protects by default and not by force. A
`git add -f` bypasses the ignore file silently and this is the only thing that sees it.

It reads `git ls-files` rather than the working tree on purpose: an untracked CSV
sitting in `data/` is correct and expected, and a tracked one is the failure.
"""
import subprocess
from pathlib import Path

import pytest

from nemul import config

# Suffixes that are client data or a built model, never source. `.joblib` and `.pkl`
# are here because the PR5 artefact script writes a fitted model, and a model fitted
# on client rows carries those rows' information in its coefficients.
FORBIDDEN_SUFFIXES = {".csv", ".xlsx", ".xls", ".pkl", ".joblib", ".h5", ".parquet"}

# Directories whose whole contents are data or generated artefacts.
FORBIDDEN_DIRS = ("data/", "models/", "artifacts/")


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return [p for p in out.stdout.split("\0") if p]


def test_no_client_data_tracked():
    """Test 18. No CSV, spreadsheet or model file is tracked, and no data dir is."""
    tracked = _tracked_files()
    assert tracked, "git ls-files returned nothing; this test is not actually looking"

    by_suffix = [p for p in tracked if Path(p).suffix.lower() in FORBIDDEN_SUFFIXES]
    by_dir = [p for p in tracked if p.startswith(FORBIDDEN_DIRS)]

    assert not by_suffix, (
        "client data or a model file is in the git index: "
        f"{by_suffix}. Remove it with `git rm --cached`; .gitignore does not "
        "retroactively untrack, and history is the part that cannot be taken back"
    )
    assert not by_dir, f"a data or artefact directory is tracked: {by_dir}"


def test_the_audit_would_catch_a_planted_file(tmp_path):
    """The guard, checked by mutation: a planted path trips the same predicate.

    PR4's lesson -- a check that has never been red proves nothing. Nothing is added
    to the real index here; the classification logic is exercised on a fake listing.
    """
    planted = ["nemul/loader.py", "data/processed/formulations_clean.csv"]
    assert [p for p in planted if Path(p).suffix.lower() in FORBIDDEN_SUFFIXES]
    assert [p for p in planted if p.startswith(FORBIDDEN_DIRS)]


def test_gitignore_blocks_the_artifact_directory():
    """`artifacts/` is generated output and must stay out of git.

    The PR5 artefact script fills the delivered blocks with real figures and writes a
    fitted model beside them. Regenerating them is one command, so there is nothing
    to gain by tracking them and a client-data leak to lose.
    """
    ignore = (config.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/" in ignore, ".gitignore must block the artifacts directory"
