"""Generate the report figures for the nano-emulsion progress report. One command.

What it writes, into `artifacts/figures/` (gitignored, one command away from being
rebuilt):

    fig01_target_distribution.png        raw data
    fig02_run_order_vs_stability.png     raw data
    fig03_data_completeness.png          raw data
    fig04_feature_distributions.png      processed data
    fig05_replicate_group_spread.png     processed data
    fig06_feature_correlation_heatmap.png   feature engineering
    fig07_lecithin_vs_stability.png      feature engineering
    fig08_lecithin_rank_sensitivity.png  feature engineering
    fig09_design_space_density.png       feature engineering
    fig10_model_ladder_r2.png            model analysis
    fig11_model_collapse.png             model analysis
    fig12_ceiling_vs_model_r2.png        model analysis
    fig13_classification_auc.png         model analysis

Every figure follows the same anatomy: a coloured section tag (which of the four
groups above it belongs to), a bold headline (the finding, not just the axes), a
one-line subtitle (what's plotted, with its one key number), the chart itself on a
bordered card, and a muted footer naming its source and sample size. See
`references/marks-and-anatomy.md` in the dataviz skill for the rules this follows.

Two rules this script follows, both already settled for `make_artifacts.py`:

- **Figures 10-13 read `artifacts/figures.json`, they never recompute the ladder.**
  That file is the single drift-checked source for every derived model number; a
  second computation of the same R2 is a second place for it to go stale. Run
  `scripts/make_artifacts.py` first if `figures.json` is missing or stale.
- **Figures 1-9 read the data directly** through `nemul.loader`, because they plot
  rows and distributions that `figures.json` does not carry (it carries summary
  statistics, not per-row values).

Run: `.venv/Scripts/python.exe scripts/make_figures.py`
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Patch, Rectangle
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemul import config, loader  # noqa: E402

OUT = config.REPO_ROOT / "artifacts" / "figures"

# ---------------------------------------------------------------- the palette
# references/palette.md (dataviz skill), light mode. Fixed categorical order;
# colour assigned by job (identity / magnitude / polarity), never by taste.
BLUE = "#2a78d6"     # categorical slot 1 -- identity default, sequential hue
ORANGE = "#eb6834"   # categorical slot 2 -- identity, second series
RED = "#e34948"      # categorical slot 8 -- diverging warm pole
GRAY_MID = "#f0efec"  # diverging neutral midpoint
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e8e7e0"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"     # chart / card surface
PAGE = "#f3f2ee"        # page plane behind the card

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]  # steps 100/200/350/450/600

# Section tags: a fixed colour per report section, distinct from the blue/orange used
# as in-chart series colours so the two encodings never collide. Each slot is used as
# an identity badge (a dot beside muted text), never as text colour -- marks spec.
SECTIONS = {
    "raw": ("RAW DATA", "#1baf7a"),                # categorical slot 3, aqua
    "processed": ("PROCESSED DATA", "#eda100"),    # categorical slot 4, yellow
    "feature": ("FEATURE ENGINEERING", "#6250d6"),  # categorical slot 7, violet
    "model": ("MODEL ANALYSIS", "#e87ba4"),         # categorical slot 5, magenta
}

HEADER_IN = 1.5   # inches reserved above the plot for the tag + headline + subtitle
FOOTER_IN = 0.34  # inches reserved below the plot for the source footer
MARGIN = 0.016    # card inset, figure fraction


def _style():
    plt.rcParams.update({
        "figure.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": PAGE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "grid.color": GRID,
        "font.size": 10.5,
        "font.family": "sans-serif",
        "legend.fontsize": 9.5,
    })


def _track(s: str) -> str:
    """Thin-space-joined caps, for a tracked-letterspacing look on short tags only."""
    return " ".join(s)


def _clean_axes(ax, grid_axis="y"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(0.9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _finish(fig, name, section, title, subtitle, caption, footer_in=FOOTER_IN):
    """The shared card: section tag, headline, subtitle, chart, footer. One recipe
    for all 13 figures, so the report reads as one system rather than 13 one-offs.
    """
    label, accent = SECTIONS[section]
    w_in, h_in = fig.get_size_inches()

    # -- the card: a bordered surface panel sitting on the page plane ------------
    fig.patches.append(Rectangle(
        (MARGIN, MARGIN), 1 - 2 * MARGIN, 1 - 2 * MARGIN, transform=fig.transFigure,
        facecolor=SURFACE, edgecolor="#e4e3db", linewidth=1.1, zorder=0,
    ))
    stripe_w = 0.009
    fig.patches.append(Rectangle(
        (MARGIN, MARGIN), stripe_w, 1 - 2 * MARGIN, transform=fig.transFigure,
        facecolor=accent, edgecolor="none", zorder=1,
    ))

    left = MARGIN + stripe_w + 0.028
    right = 1 - MARGIN - 0.022

    # -- header: tag, then headline, then subtitle, top-anchored -----------------
    tag_y = 1 - MARGIN - 0.052
    dot_r_pts = 3.1
    fig.text(left, tag_y, "●", fontsize=dot_r_pts * 2, color=accent,
              ha="left", va="center", zorder=3)
    fig.text(left + 0.024, tag_y, _track(label), fontsize=8.3, color=MUTED,
              fontweight="bold", ha="left", va="center", zorder=3)
    fig.text(left, tag_y - 0.058, title, fontsize=15.5, color=INK, fontweight="bold",
              ha="left", va="top", wrap=True, zorder=3)
    fig.text(left, tag_y - 0.058 - (0.405 / h_in), subtitle, fontsize=10.3,
              color=INK_SECONDARY, ha="left", va="top", wrap=True, zorder=3)

    # -- the plot area, constrained below the header and above the footer --------
    top_frac = 1 - (HEADER_IN / h_in)
    bottom_frac = footer_in / h_in
    fig.tight_layout(rect=(left, bottom_frac, right, top_frac))
    # Figure-level patches (the card, the stripe) render above child axes regardless
    # of the zorder passed to Rectangle -- explicitly raise every axes above them.
    for a in fig.axes:
        a.set_zorder(5)

    # -- footer: a hairline rule, then the muted source note ---------------------
    fig.add_artist(plt.Line2D([left, right], [bottom_frac - 0.028, bottom_frac - 0.028],
                               transform=fig.transFigure, color="#e8e7e0", linewidth=1,
                               zorder=2))
    fig.text(left, MARGIN + 0.028, caption, color=MUTED, fontsize=7.6, ha="left",
              va="bottom", wrap=True, zorder=3)

    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print(f"  wrote {name}")


# ----------------------------------------------------------------- figure 1
def fig01_target_distribution(df_raw, df_an):
    y = df_an["stability_days"].to_numpy()
    plateau_n = int((y == 10.0).sum())

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    ax.hist(y, bins=28, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.axvline(10.0, color=ORANGE, linewidth=1.6, zorder=3)
    ax.annotate(
        f"{plateau_n} formulations read exactly 10.0 d\n(one daily-inspection round)",
        xy=(10.0, ax.get_ylim()[1]), xytext=(16, ax.get_ylim()[1] * 0.93),
        color=INK_SECONDARY, fontsize=9,
    )
    ax.set_xlabel("stability (days)")
    ax.set_ylabel("formulations")
    _clean_axes(ax)
    _finish(fig, "fig01_target_distribution.png", "raw",
            "Stability is right-skewed, with a 10-day plateau",
            f"{len(y)} analysed formulations; {plateau_n} share the same reading, "
            "rounded by a daily inspection",
            "data/processed/formulations_clean.csv  ·  column: stability_days")


# ----------------------------------------------------------------- figure 2
def fig02_run_order_vs_stability(df_an):
    x = df_an["num"].to_numpy()
    y = df_an["stability_days"].to_numpy()
    is_ready = df_an["num"].isin(config.PRODUCT_READY_RUNS).to_numpy()
    lo, hi = config.EARLY_BLOCK_RUNS

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.set_ylim(top=y.max() * 1.14)  # headroom so the band label clears the top point
    ax.axvspan(lo, hi, color=MUTED, alpha=0.1, zorder=1)
    ax.scatter(x[~is_ready], y[~is_ready], s=28, color=BLUE, alpha=0.85,
               edgecolor=SURFACE, linewidth=0.3, label="other formulations", zorder=3)
    ax.scatter(x[is_ready], y[is_ready], s=52, color=ORANGE, edgecolor=INK,
               linewidth=0.5, label="client product-ready (9)", zorder=4)
    ax.annotate("early high-lecithin block", xy=(0.5 * (lo + hi), 1), xycoords=("data", "axes fraction"),
                xytext=(0, -4), textcoords="offset points", color=MUTED, fontsize=8.5, ha="center", va="top")
    ax.set_xlabel("run order (formulation's position in the preparation sequence)")
    ax.set_ylabel("stability (days)")
    ax.legend(frameon=False, loc="upper right")
    _clean_axes(ax)
    _finish(fig, "fig02_run_order_vs_stability.png", "raw",
            "Stability tracks run order, not just composition",
            f"All 9 client product-ready formulations fall inside one {hi - lo + 1}-run early block",
            f"shaded band = runs {lo}-{hi}  ·  n = {len(x)} analysed formulations")


# ----------------------------------------------------------------- figure 3
def fig03_data_completeness(df_raw):
    cols = ["stability_days", "z_average_nm", "pdi", "peak_size_nm", "zeta_mv",
            "mobility_cm2_vs"]
    labels = ["stability (days)", "z-average (nm)", "PDI", "peak size (nm)",
              "zeta potential (mV)", "mobility"]
    n = len(df_raw)
    counts = [int(df_raw[c].notna().sum()) for c in cols]
    order = np.argsort(counts)
    labels = [labels[i] for i in order]
    counts = [counts[i] for i in order]
    min_dls = min(counts[:-1]) if len(counts) > 1 else counts[0]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    y_pos = np.arange(len(labels))
    colors = [MUTED if c == n else BLUE for c in counts]
    ax.barh(y_pos, counts, color=colors, height=0.58, zorder=2)
    for yi, c in zip(y_pos, counts):
        ax.text(c + n * 0.015, yi, f"{c}/{n}", va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(y_pos, labels)
    ax.set_xlabel("formulations measured")
    ax.set_xlim(0, n * 1.14)
    _clean_axes(ax, grid_axis="x")
    _finish(fig, "fig03_data_completeness.png", "raw",
            "DLS data exists only for surviving formulations",
            f"Stability was read on all {n}; DLS metrics on as few as {min_dls}",
            f"data/raw-data/formulations.csv  ·  counts out of {n} rows")


# ----------------------------------------------------------------- figure 4
def fig04_feature_distributions(df_an):
    feats = config.FEATURES
    data = [df_an[f].to_numpy() for f in feats]
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    bp = ax.boxplot(data, tick_labels=[f.replace("_pct", "") for f in feats],
                     patch_artist=True, widths=0.55, zorder=2,
                     medianprops=dict(color=ORANGE, linewidth=1.9),
                     whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
                     flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                      markeredgecolor="none", alpha=0.6))
    for box in bp["boxes"]:
        box.set(facecolor=SEQ_BLUE[1], edgecolor=BLUE, linewidth=1.1)
    ax.set_ylabel("weight percent")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    _clean_axes(ax)
    _finish(fig, "fig04_feature_distributions.png", "processed",
            "The 11 composition features span different scales",
            "Botanical extract dominates the formula by mass; every active ingredient "
            "is a low single digit",
            f"n = {len(df_an)} analysed formulations  ·  orange = median, box = IQR, dots = outliers")


# ----------------------------------------------------------------- figure 5
def fig05_replicate_group_spread(df_an):
    groups = loader.composition_groups(df_an)
    sizes = groups.value_counts()
    rep_ids = sizes[sizes > 1].index

    rows = []
    for gid in rep_ids:
        sub = df_an.loc[groups == gid]
        rows.append({
            "runs": ",".join(str(int(n)) for n in sorted(sub["num"])),
            "lo": sub["stability_days"].min(),
            "hi": sub["stability_days"].max(),
        })
    rows.sort(key=lambda r: r["hi"] - r["lo"])
    spreads = [r["hi"] - r["lo"] for r in rows]
    widest = max(spreads)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y_pos = np.arange(len(rows))
    for yi, r, spread in zip(y_pos, rows, spreads):
        color = RED if spread == widest else BLUE
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=color, linewidth=2.4, zorder=2,
                solid_capstyle="round")
        ax.scatter([r["lo"], r["hi"]], [yi, yi], color=color, s=42, zorder=3,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(y_pos, [f"runs {r['runs']}" for r in rows])
    ax.set_xlabel("stability (days), min to max within the group")
    _clean_axes(ax)
    _finish(fig, "fig05_replicate_group_spread.png", "processed",
            f"Identical compositions still disagree by {widest:.0f} days",
            f"{len(rows)} replicate groups in the data; the widest (red) is runs "
            f"{rows[spreads.index(widest)]['runs']}",
            "min-to-max stability within each group  ·  direct evidence behind the reproducibility ceiling")


# ----------------------------------------------------------------- figure 6
def fig06_feature_correlation_heatmap(df_an):
    cols = config.FEATURES + [config.TARGET]
    corr = df_an[cols].corr(method="spearman").to_numpy()
    n = len(cols)
    cmap = LinearSegmentedColormap.from_list("div", [RED, GRAY_MID, BLUE], N=256)
    off_diag = corr[~np.eye(n, dtype=bool)]
    strongest_idx = np.unravel_index(np.argmax(np.abs(np.where(np.eye(n, dtype=bool), 0, corr))), corr.shape)
    strongest_val = corr[strongest_idx]
    a, b = [c.replace("_pct", "") for c in cols][strongest_idx[0]], [c.replace("_pct", "") for c in cols][strongest_idx[1]]

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    labels = [c.replace("_pct", "") for c in config.FEATURES] + ["stability"]
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            v = corr[i, j]
            txt_color = SURFACE if abs(v) > 0.55 else INK
            weight = "bold" if i == n - 1 or j == n - 1 else "normal"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2,
                     color=txt_color, fontweight=weight)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Spearman correlation")
    cbar.outline.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _finish(fig, "fig06_feature_correlation_heatmap.png", "feature",
            "Composition features correlate with each other",
            f"Strongest pair: {a} vs {b} at ρ = {strongest_val:.2f} — bigger than any single "
            "feature's link to stability",
            f"Spearman correlation, n = {len(df_an)}  ·  correlation alone can't separate composition "
            "from run order -- see fig08",
            footer_in=0.58)


# ----------------------------------------------------------------- figure 7
def fig07_lecithin_vs_stability(df_an):
    x = df_an["lecithin_pct"].to_numpy()
    y = np.log1p(df_an["stability_days"].to_numpy())
    order = df_an["num"].to_numpy()
    seq_cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE, N=256)

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    sc = ax.scatter(x, y, c=order, cmap=seq_cmap, s=36, edgecolor=INK, linewidth=0.3, zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("run order (dark = early)")
    cbar.outline.set_visible(False)
    ax.set_xlabel("lecithin (wt%)")
    ax.set_ylabel("log1p(stability, days)")
    _clean_axes(ax)
    _finish(fig, "fig07_lecithin_vs_stability.png", "feature",
            "Lecithin's effect is entangled with run order",
            "Dark points (early runs) sit both high in lecithin and high in stability",
            "colour = preparation run order  ·  the entanglement fig08 quantifies as a rank")


# ----------------------------------------------------------------- figure 8
def fig08_lecithin_rank_sensitivity(fig_json):
    tri = fig_json["triad"]
    order_labels = ["naive\n(run order ignored)", "with-covariate\n(run order added)",
                     "within-block\n(worst block)", "time-split\n(early fit)"]
    ranks = [v["rank"] for v in tri["views"]]
    n_feat = fig_json["n_features"]

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    x = np.arange(len(ranks))
    ax.axhspan(min(ranks), max(ranks), color=BLUE, alpha=0.09, zorder=1)
    ax.plot(x, ranks, color=BLUE, linewidth=2, zorder=2, solid_capstyle="round")
    ax.scatter(x, ranks, color=BLUE, s=78, zorder=3, edgecolor=SURFACE, linewidth=1.6)
    for xi, r in zip(x, ranks):
        ax.annotate(f"#{r}", (xi, r), textcoords="offset points", xytext=(0, 11),
                    ha="center", color=INK_SECONDARY, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x, order_labels)
    ax.set_ylim(n_feat + 0.6, -0.4)
    ax.set_ylabel("feature rank (1 = strongest)")
    _clean_axes(ax)
    _finish(fig, "fig08_lecithin_rank_sensitivity.png", "feature",
            f"Lecithin's rank moves from 1st to {max(ranks)}th of {n_feat}",
            "Direction never flips across any of the four views; only the magnitude is unsettled",
            "source: artifacts/figures.json, triad.sentence  ·  standardised-coefficient rank")


# ----------------------------------------------------------------- figure 9
def fig09_design_space_density(df_an):
    X = loader.feature_matrix(df_an).to_numpy()
    groups = loader.composition_groups(df_an).to_numpy()
    D = cdist(X, X)
    same_group = groups[:, None] == groups[None, :]
    D[same_group] = np.inf
    nearest = D.min(axis=1)
    median, p95 = np.median(nearest), np.percentile(nearest, config.EXTRAPOLATION_PERCENTILE)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.hist(nearest, bins=24, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=2)
    ax.axvline(median, color=INK_SECONDARY, linewidth=1.4, linestyle=":", zorder=3)
    ax.axvline(p95, color=ORANGE, linewidth=1.8, zorder=3)
    ax.annotate(f"median {median:.2f} wt%", (median, ax.get_ylim()[1] * 0.92),
                xytext=(7, 0), textcoords="offset points", color=INK_SECONDARY, fontsize=9)
    ax.annotate(f"95th pct {p95:.2f} wt%\n(extrapolation threshold)",
                (p95, ax.get_ylim()[1] * 0.74), xytext=(7, 0), textcoords="offset points",
                color=ORANGE, fontsize=9, fontweight="bold")
    ax.set_xlabel("distance to nearest different composition (wt%, Euclidean over 11 components)")
    ax.set_ylabel("formulations")
    _clean_axes(ax)
    _finish(fig, "fig09_design_space_density.png", "feature",
            "The design is dense locally, thin everywhere else",
            f"Half the formulations sit within {median:.2f} wt% of a neighbour; almost none "
            f"sits past {p95:.2f} wt%",
            f"n = {len(df_an)}  ·  same metric as the shortlist's extrapolation flag (nemul/shortlist.py)")


# ----------------------------------------------------------------- figure 10
def fig10_model_ladder_r2(fig_json):
    ladder = fig_json["ladder"]
    names = [r["rung"].replace("_", " ") for r in ladder]
    grouped = [r["r2_grouped"] for r in ladder]
    block_out = [r["r2_block_out"] for r in ladder]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(names))
    w = 0.36
    b1 = ax.bar(x - w / 2, grouped, width=w, color=BLUE, label="composition-grouped CV", zorder=2)
    b2 = ax.bar(x + w / 2, block_out, width=w, color=ORANGE, label="run-block held out", zorder=2)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            va = "bottom" if h >= 0 else "top"
            off = 0.012 if h >= 0 else -0.012
            ax.text(rect.get_x() + rect.get_width() / 2, h + off, f"{h:.2f}",
                    ha="center", va=va, fontsize=8, color=INK_SECONDARY)
    ax.axhline(0, color=BASELINE, linewidth=1.1, zorder=1)
    ax.set_xticks(x, names)
    ax.set_ylabel("R2 (log1p days, out-of-fold)")
    ax.legend(frameon=False, loc="upper left")
    _clean_axes(ax)
    _finish(fig, "fig10_model_ladder_r2.png", "model",
            "Non-linear models win only when folds can interpolate",
            "GP and GBM beat the mean under grouped CV, then drop below it once a "
            "run-order block is held out",
            f"source: artifacts/figures.json, commit {fig_json['git_rev']}  ·  "
            f"{fig_json['n_splits']} folds x {fig_json['n_repeats']} repeats")


# ----------------------------------------------------------------- figure 11
def fig11_model_collapse(fig_json):
    ladder = fig_json["ladder"]
    names = [r["rung"].replace("_", " ") for r in ladder]
    collapse = [r["collapse"] for r in ladder]
    nonlinear = {"gp", "gbm"}
    colors = [ORANGE if r["rung"] in nonlinear else BLUE for r in ladder]

    order = np.argsort(collapse)
    names = [names[i] for i in order]
    collapse = [collapse[i] for i in order]
    colors = [colors[i] for i in order]
    worst = max(collapse)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    y = np.arange(len(names))
    ax.barh(y, collapse, color=colors, height=0.6, zorder=2)
    for yi, c in zip(y, collapse):
        ax.text(c + worst * 0.015, yi, f"{c:.2f}", va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_yticks(y, names)
    ax.set_xlim(0, worst * 1.16)
    ax.set_xlabel("R2 grouped minus R2 block-out  (higher = more collapse)")
    ax.legend(handles=[Patch(color=BLUE, label="linear rung"),
                        Patch(color=ORANGE, label="non-linear rung")],
              frameon=False, loc="lower right")
    _clean_axes(ax, grid_axis="x")
    _finish(fig, "fig11_model_collapse.png", "model",
            "Non-linear rungs' skill is mostly run-order collapse",
            f"GP and GBM lose up to {worst:.2f} R2 under block-out; the linear rungs barely move",
            "collapse = R2 grouped minus R2 block-out, per rung  ·  source: artifacts/figures.json")


# ----------------------------------------------------------------- figure 12
def fig12_ceiling_vs_model_r2(fig_json):
    cf = fig_json["ceiling_fit"]
    ladder = fig_json["ladder"]
    names = [r["rung"].replace("_", " ") for r in ladder] + ["reproducibility ceiling"]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 5.1))
    ax.axvspan(cf["lo"], cf["hi"], color=MUTED, alpha=0.16, zorder=1)
    ax.axvline(cf["point"], color=INK_SECONDARY, linewidth=1.6, zorder=2)
    for yi, r in zip(y[:-1], ladder):
        ax.scatter(r["r2_grouped"], yi, color=BLUE, s=76, zorder=3, edgecolor=SURFACE, linewidth=1.4)
    ax.scatter(cf["point"], y[-1], color=INK_SECONDARY, marker="D", s=64, zorder=3,
               edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=BASELINE, linewidth=1.1, linestyle="--", zorder=1)
    ax.set_yticks(y, names)
    ax.set_xlabel("R2 on the log1p scale (the scale the model is fitted on)")
    _clean_axes(ax, grid_axis="x")
    _finish(fig, "fig12_ceiling_vs_model_r2.png", "model",
            "Models are scored against a ceiling near zero",
            f"Ceiling's 95% interval ({cf['lo']:.2f} to {cf['hi']:.2f}) spans almost the whole "
            "range of model scores",
            f"log1p fit scale, {cf['df']} replicate df  ·  source: artifacts/figures.json")


# ----------------------------------------------------------------- figure 13
def fig13_classification_auc(fig_json):
    b = fig_json["binary"]
    labels = ["our classifier\n(out-of-fold)", f"published benchmark\n(n = {b['anchor_n']})",
              "chance"]
    values = [b["auc"], b["anchor_auc"], 0.5]
    colors = [BLUE, ORANGE, MUTED]
    gap = b["anchor_auc"] - b["auc"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, zorder=2)
    for yi, v in zip(y, values):
        ax.text(v + 0.014, yi, f"{v:.3f}", va="center", color=INK_SECONDARY, fontsize=9.5,
                fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"area under the ROC curve -- stable past {b['threshold_days']} days")
    _clean_axes(ax, grid_axis="x")
    _finish(fig, "fig13_classification_auc.png", "model",
            "Classification trails the one published benchmark",
            f"{b['auc']:.3f} AUC here vs {b['anchor_auc']:.3f} on a {b['anchor_n']}-row published "
            f"dataset — a {gap:.2f} gap in the expected direction",
            f"anchor cited by decision, doi:{b['anchor_doi']}, not recomputed  ·  "
            "accuracy omitted: classes are lopsided")


# ------------------------------------------------------------------------- main
def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig_path = OUT.parent / "figures.json"
    if not fig_path.exists():
        raise SystemExit(
            "artifacts/figures.json is missing. Run "
            "`.venv/Scripts/python.exe scripts/make_artifacts.py` first -- figures 10-13 "
            "read the model numbers from there rather than recomputing them."
        )
    fig_json = json.loads(fig_path.read_text(encoding="utf-8"))

    df_raw = loader.load_raw()
    df_an = loader.load_analysis_set()
    print(f"raw: {len(df_raw)} rows, analysis set: {len(df_an)} rows")

    fig01_target_distribution(df_raw, df_an)
    fig02_run_order_vs_stability(df_an)
    fig03_data_completeness(df_raw)
    fig04_feature_distributions(df_an)
    fig05_replicate_group_spread(df_an)
    fig06_feature_correlation_heatmap(df_an)
    fig07_lecithin_vs_stability(df_an)
    fig08_lecithin_rank_sensitivity(fig_json)
    fig09_design_space_density(df_an)
    fig10_model_ladder_r2(fig_json)
    fig11_model_collapse(fig_json)
    fig12_ceiling_vs_model_r2(fig_json)
    fig13_classification_auc(fig_json)

    print(f"\nall 13 figures: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
