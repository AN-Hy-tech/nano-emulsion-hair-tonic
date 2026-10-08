"""PR6: the pre-commit hook that runs the drift check.

Why a hook and not a test. `test_stats_report_passes` was cut on the 2026-10-08 audit
(`workcycle.md`) because a pytest case that only shells out to a script proves nothing
you did not already know -- the script either runs or it does not. What was actually
wanted is the *timing*: the drift check has to run before a commit lands, not whenever
somebody remembers to run the suite. That is a git hook, and it was the last Done-gate
item left open because a hook is local config per checkout and no PR could carry it.

PR6 carries the two halves a PR can carry -- the hook script itself, tracked in
`.githooks/` so it travels with the repo, and the assertion that it is installed here --
and leaves one command to run per checkout:

    git config core.hooksPath .githooks

What is tested. The installation, and then the hook's three outcomes, each exercised by
running the real hook script against a planted stub rather than the real drift check:
drift blocks the commit, no drift lets it through, and a checkout with no client data in
it skips instead of blocking. The second of those three is the mutation evidence -- the
guard was red when the stub exited 1, which is the whole point of writing it that way.

`NEMUL_HOOK_PYTHON` is how the stub gets run: the hook takes the interpreter from that
variable if it is set and autodetects `.venv` otherwise, so a checkout whose interpreter
lives somewhere else can point at it without editing a tracked file.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from nemul import config

HOOK = config.REPO_ROOT / ".githooks" / "pre-commit"

# The hook is POSIX sh. On Windows that is the sh Git for Windows ships, which is also
# what git itself uses to run hooks -- so if it is missing, hooks do not run at all.
SH = shutil.which("sh") or shutil.which("bash")

# The file the hook looks for to decide whether this checkout has figures to check.
# Same path as `config.CLEAN_CSV`, relative, because that is what the hook can see.
DATA_REL = Path("data") / "processed" / "formulations_clean.csv"


def test_the_hook_script_is_tracked():
    """The hook lives in the repo, so a fresh checkout has it without being told."""
    assert HOOK.is_file(), f"{HOOK} is missing; the hook must be tracked, not local-only"

    tracked = subprocess.run(
        ["git", "ls-files", ".githooks"],
        cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    assert ".githooks/pre-commit" in tracked, (
        "the hook script is not in the git index, so it does not travel with the repo"
    )

    body = HOOK.read_text(encoding="utf-8")
    assert "stats_report.py" in body, "the hook must run the drift check"


def test_the_hook_has_unix_line_endings():
    """The hook stays LF, because a CRLF hook breaks on some shells and not others.

    Git's default on Windows converts line endings on checkout, so the tracked file can
    be fine while the checked-out file is not, and a hook that cannot run blocks nothing
    while looking installed. `.gitattributes` pins it to LF and this asserts both halves:
    the attribute git will apply, and the bytes on disk now.

    **Premise corrected on build (PR6).** This was written claiming `sh` fails on every
    line of a CRLF script. Planting CRLF in the real hook turned this test red as
    intended, but the hook itself still *ran*, exit 0, under the `sh` Git for Windows
    ships -- so that shell tolerates the CR. The breakage is shell-dependent and was not
    reproduced on this machine; it is not claimed here beyond that. The guard stays,
    because the hook is tracked and the next checkout may not be this one, and because
    pinning the line endings of a shell script costs nothing either way.
    """
    raw = HOOK.read_bytes()
    assert b"\r\n" not in raw, (
        "the hook has CRLF line endings; sh will fail on every line of it"
    )

    attrs = subprocess.run(
        ["git", "check-attr", "eol", "--", ".githooks/pre-commit"],
        cwd=config.REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert "eol: lf" in attrs, (
        "git is not pinning the hook to LF, so a checkout may convert it to CRLF and "
        f"break it: {attrs.strip()!r}. Needs `.githooks/** text eol=lf` in .gitattributes"
    )


def test_the_hook_is_installed_in_this_checkout():
    """The Done-gate item itself: `core.hooksPath` points at the tracked hook.

    This is local config, so it is red on a fresh clone **by design** -- and the error
    message is the install instruction. It is the only assertion in the suite about the
    machine rather than the code, and it is here because the alternative is a checklist
    item nobody re-reads.
    """
    out = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=config.REPO_ROOT, capture_output=True, text=True,
    )
    assert out.stdout.strip() == ".githooks", (
        "the pre-commit hook is not installed in this checkout. Run:\n"
        "    git config core.hooksPath .githooks\n"
        f"(core.hooksPath is currently {out.stdout.strip() or 'unset'!r})"
    )


def _sandbox(tmp_path: Path, *, drift_exit: int | None, with_data: bool) -> Path:
    """A throwaway git repo holding the real hook and a stub drift check.

    `drift_exit` is the exit status the stub reports; None writes no stub at all, which
    is how the no-data case is checked without one standing by to be run.
    """
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    hook = tmp_path / ".githooks" / "pre-commit"
    hook.parent.mkdir(parents=True)
    shutil.copy(HOOK, hook)

    if drift_exit is not None:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "stats_report.py").write_text(
            "import sys\n"
            "print('DRIFT: planted by the test', file=sys.stderr)\n"
            f"sys.exit({drift_exit})\n",
            encoding="utf-8",
        )

    if with_data:
        (tmp_path / DATA_REL).parent.mkdir(parents=True)
        (tmp_path / DATA_REL).write_text("num,stability_days\n1,10\n", encoding="utf-8")

    return hook


def _run(tmp_path: Path, hook: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, NEMUL_HOOK_PYTHON=sys.executable)
    return subprocess.run(
        [SH, str(hook)], cwd=tmp_path, env=env, capture_output=True, text=True,
    )


@pytest.mark.skipif(SH is None, reason="no POSIX sh on PATH, so git cannot run hooks either")
def test_the_hook_blocks_the_commit_when_a_figure_has_drifted(tmp_path):
    """The mutation. A drift check that exits 1 fails the commit, and says why."""
    hook = _sandbox(tmp_path, drift_exit=1, with_data=True)
    run = _run(tmp_path, hook)

    assert run.returncode != 0, (
        "the hook let a commit through while the drift check was failing:\n"
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    both = run.stdout + run.stderr
    assert "DRIFT: planted by the test" in both, (
        "the hook swallowed the drift check's own output, so the committer cannot see "
        "which figure moved"
    )


@pytest.mark.skipif(SH is None, reason="no POSIX sh on PATH, so git cannot run hooks either")
def test_the_hook_lets_the_commit_through_when_nothing_has_drifted(tmp_path):
    """The other side of the mutation: exit 0 is not blocked.

    Without this one, a hook that blocked every commit would pass the test above.
    """
    hook = _sandbox(tmp_path, drift_exit=0, with_data=True)
    run = _run(tmp_path, hook)

    assert run.returncode == 0, (
        "the hook blocked a clean commit:\n"
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )


@pytest.mark.skipif(SH is None, reason="no POSIX sh on PATH, so git cannot run hooks either")
def test_the_hook_skips_where_there_is_no_client_data(tmp_path):
    """A checkout without the data file skips, rather than blocking every commit.

    The data file is client data and is never in the repo, so a fresh clone has no
    figures to check. Blocking there would make the hook the first thing anybody
    removes, and a hook that gets removed checks nothing.
    """
    hook = _sandbox(tmp_path, drift_exit=None, with_data=False)
    run = _run(tmp_path, hook)

    assert run.returncode == 0, (
        "the hook blocked a commit in a checkout with no client data in it:\n"
        f"stdout={run.stdout!r} stderr={run.stderr!r}"
    )
    assert "skip" in (run.stdout + run.stderr).lower(), (
        "the hook skipped silently; a skipped check must say so or it reads as a pass"
    )
