# nano-emulsion-hair-tonic

Composition-only machine-learning analysis of **109 herbal nanoemulsion formulations**, modelling
days to visible degradation. Built for Dr. Manijeh Jokar (Shiraz University), under the Iran
National Science Foundation independent postdoctoral programme, tracking code 4042105.

**The headline result, so nobody has to dig for it:** with this dataset you can rank candidate
formulations, but you cannot predict how many days any one of them will last. The deliverable is a
screening aid, not a shelf-life forecast. Section 5 of the report says why, in detail.

## راهنمای فارسی

گزارش کامل این پروژه، به فارسی و بدون نیاز به پیش‌زمینهٔ یادگیری ماشین، همهٔ نتایج و محدودیت‌ها و
فهرست پیشنهادی برای ساخت بعدی را در بر دارد. **برای خواندن نتایج، به آن گزارش مراجعه کنید؛**
این مخزن فقط کد تحلیل است. نتیجهٔ کوتاه: با این داده‌ها می‌توان فرمولاسیون‌ها را **رتبه‌بندی** کرد،
اما نمی‌توان **تعداد روز** پایداری را پیش‌بینی کرد.

## What is and is not in this repository

**The client's data is not here, and never will be.** `.gitignore` blocks `data/`, every
spreadsheet and CSV extension, and model files; `tests/test_repo.py` asserts that the git *index*
holds none of them, because `.gitignore` is a default and `git add -f` walks straight past it.
History is the part you cannot take back.

Also absent by design:

| Not here | Where it lives |
|---|---|
| Planning docs, decisions, the PR log | A separate private `sdlc` repo |
| The generated client blocks and the fitted model | `artifacts/`, gitignored, rebuilt by one command |
| `CLAUDE.md`, `.claude/settings.local.json` | Local only |

What is here is the code, the tests, and the exact commands to reproduce every number.

## Running it

Python 3.13. Dependencies are pinned in `requirements.txt`; `joblib` is pinned too, because a
pickled model is not a stable format across versions.

```sh
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# place the client CSV at data/raw-data/formulations.csv first, then:
.venv/Scripts/python.exe scripts/build_clean_dataset.py   # renormalise, quarantine, build the clean CSV
.venv/Scripts/python.exe -m pytest                        # 46 tests
.venv/Scripts/python.exe scripts/stats_report.py          # drift-check every documented figure
.venv/Scripts/python.exe scripts/make_artifacts.py         # regenerate artifacts/

git config core.hooksPath .githooks                        # once per checkout
```

**Order matters for exactly one reason.** `stats_report.py` recomputes every figure quoted in the
planning docs from the CSV and **exits 1 on any mismatch**. Run it after anything that touches the
data and reconcile the docs before trusting a model result: a model fitted on data whose documented
figures have drifted is a model nobody can describe. The last line installs the pre-commit hook that
runs that check before every commit; it skips, out loud, in a checkout with no `data/` in it.

## Layout

```
nemul/          the library — one module per concern
  config.py     paths, the 11-feature contract, the never-a-feature list, the one seed
  loader.py     the only code that opens the CSV
  splits.py     repeated GroupKFold grouped on composition
  target.py     the log1p transform and its back-transform
  ceiling.py    the reproducibility ceiling from replicate groups
  intervals.py  jackknife+ prediction intervals
  ladder.py     every model scored by one validation function on identical folds
  binary.py     the 30-day classification rung
  confound.py   run-order covariate, within-block and time-split views
  shortlist.py  candidate generation, renormalisation, guardrails, ranking
scripts/        build_clean_dataset.py, stats_report.py, make_artifacts.py
tests/          46 tests, named as the planning docs contract them
.githooks/      the pre-commit drift check
```

## How to read the tests

They are the specification, not a safety net bolted on afterwards: each one was **written red
before the code existed**, and several guards were additionally mutation-checked — the guard was
shown to fail on a planted defect before it was believed.

Three are worth knowing about because they encode the project's actual findings rather than its
plumbing:

- `test_nonlinear_gain_does_not_cross_run_order` — the central result. Two non-linear models beat
  the mean baseline by about +0.33 R² under composition-grouped CV and lose the whole advantage
  when a block of the preparation sequence is held out. They were interpolating between
  formulations made close together in time, not learning composition chemistry. **No
  recommendation in the deliverable is derived from them.**
- `test_beats_mean_baseline` — records the margin over the mean baseline and the across-repeat
  spread; it deliberately does **not** fail when the margin is small or negative. That outcome was
  pre-registered before anything was scored, and a test that failed on it would be a test that
  fails on the project's honest result.
- `test_no_client_data_tracked` — the repo audit described above.

## A note on figures

No derived figure is written by hand anywhere in this project. `scripts/make_artifacts.py` renders
every delivered block from one `figures.json`, so a figure cannot appear twice with two values, and
an unfilled `{{token}}` fails the script rather than shipping. `stats_report.py` is the drift check
on the documentation side. If a number needs to change, change the data or the code and rerun both.

## License and use

Private, client-owned work. Not licensed for redistribution.
