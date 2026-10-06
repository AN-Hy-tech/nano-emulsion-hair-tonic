"""Regenerate every figure quoted in sdlc/nano-emulsion-hair-tonic/planning/project.md.

Why this exists: those figures were originally computed by hand. When the client's
2026-10-06 answer grew the analysis set from 80 rows to 109, the headline tables were
recomputed but two sections were not, so the doc carried stale 80-row numbers for weeks
(hlb_client rho, every pH statistic). This script removes that failure mode: each expected
value below is the number currently written in project.md, and the script recomputes it
from the CSV and reports DRIFT on any mismatch.

Run it after anything that touches the data -- new worked sheets, a row-22 sign fix, a
rebuild via build_clean_dataset.py -- and update project.md wherever it reports DRIFT.
Exit status is 1 if anything drifted, so it also serves as a pre-commit check.

Stdlib only: there is no pandas/scipy in this environment, so ranks, Spearman, the partial
correlation and the Student-t survival function are implemented below.
"""
import csv, math, sys, statistics as st

RAW   = r"D:\Workspace\nano-emulsion-hair-tonic\data\raw-data\formulations.csv"
CLEAN = r"D:\Workspace\nano-emulsion-hair-tonic\data\processed\formulations_clean.csv"
QLOG  = r"D:\Workspace\nano-emulsion-hair-tonic\data\processed\quarantine-log.csv"

COMP = [("Oil_Blend_g", "oil_blend_pct"), ("Span80_g", "span80_pct"),
        ("Lecithin_g", "lecithin_pct"), ("Tween80_oil_phase_g", "tween80_oil_pct"),
        ("Extract_g", "extract_pct"), ("Tween80_water_phase_g", "tween80_water_pct"),
        ("Glycerol_g", "glycerol_pct"), ("PG_g", "pg_pct"),
        ("Xanthan_Gum_g", "xanthan_pct"), ("Guar_Gum_g", "guar_pct"),
        ("Water_added_g", "water_pct")]
LISTED = [k for k, _ in COMP if k != "Water_added_g"]   # Listed_mass_g excludes water
SURF = ["Span80_g", "Lecithin_g", "Tween80_oil_phase_g", "Tween80_water_phase_g"]
HLB_SPAN80, HLB_TWEEN80, HLB_LECITHIN = 4.3, 15.0, 4.0
# The client's worked F110 sheet, transcribed from their image. Its own numbers do not
# close, which is why two or three more sheets are still on the open list.
F110 = {"span80": 1.20, "lecithin": 0.35, "tween_oil": 0.35, "tween_water": 0.50,
        "stated_hlb": 10.5, "stated_surf_pct": 8.9}

# --------------------------------------------------------------- stats (stdlib only)
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def tp(t, df):
    """Two-tailed p-value for Student's t."""
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def ranks(v):
    """Ranks with ties averaged, as Spearman requires."""
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x)
    sy = sum((b - my) ** 2 for b in y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sx * sy)


def spearman(x, y):
    r = pearson(ranks(x), ranks(y))
    n = len(x)
    if n < 3 or r != r or abs(r) >= 1:
        return r, 0.0, n
    return r, tp(r * math.sqrt((n - 2) / (1 - r * r)), n - 2), n


def partial(x, y, z):
    """Spearman correlation of x,y with z partialled out. Here z is run order."""
    rx, ry, rz = ranks(x), ranks(y), ranks(z)
    ab, az, bz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    p = (ab - az * bz) / math.sqrt((1 - az * az) * (1 - bz * bz))
    n = len(x)
    return p, tp(p * math.sqrt((n - 3) / (1 - p * p)), n - 3), n


# --------------------------------------------------------------- data
def num(r, k):
    v = (r.get(k) or "").strip()
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


raw = list(csv.DictReader(open(RAW, encoding="utf-8-sig")))
clean = list(csv.DictReader(open(CLEAN, encoding="utf-8-sig")))
qlog = list(csv.DictReader(open(QLOG, encoding="utf-8-sig")))
A109 = [r for r in clean if int(r["num"]) != 110]   # row 110: composition quarantined
ROW = {int(r["num"]): r for r in clean}


def triple(rows, a, b, c=None):
    """Complete cases across two or three clean-CSV columns, order preserved."""
    xs, ys, zs = [], [], []
    for r in rows:
        u, v = num(r, a), num(r, b)
        w = num(r, c) if c else 0.0
        if u is None or v is None or w is None:
            continue
        xs.append(u)
        ys.append(v)
        zs.append(w)
    return (xs, ys, zs) if c else (xs, ys)


FAILED = []


def chk(label, expected, got, tol=0.005):
    """Compare one figure in project.md against the data. tol=0 for exact matches."""
    if isinstance(got, float) and isinstance(expected, (int, float)):
        ok = abs(expected - got) <= tol
    else:
        ok = expected == got
    if not ok:
        FAILED.append(label)
    shown = ("%.4g" % got) if isinstance(got, float) else got
    print("  %-5s %-52s doc=%-16s data=%s"
          % ("ok" if ok else "DRIFT", label, expected, shown))


# --------------------------------------------------------------- project.md : Data
print("\n== Data ==")
chk("raw rows", 110, len(raw), 0)
chk("raw cols", 27, len(raw[0]), 0)
chk("clean rows", 110, len(clean), 0)
chk("clean cols", 26, len(clean[0]), 0)
chk("quarantine notes", 19, len(qlog), 0)
chk("quarantine rows", 17, len(set(q["num"] for q in qlog)), 0)

chk("rows not closing to 100.000 wt%", 0,
    sum(1 for r in clean if abs(sum(num(r, c) or 0 for _, c in COMP) - 100.0) > 1e-3), 0)

tot = {int(r["Num"]): sum(num(r, k) or 0 for k, _ in COMP) for r in raw}
chk("rows within 0.5 g of 50 g (water counted)", 101,
    sum(1 for t in tot.values() if abs(t - 50.0) <= 0.5), 0)
nine = sorted(n for n, t in tot.items() if abs(t - 50.0) > 0.5)
chk("the nine over-batch rows", [10, 11, 12, 13, 14, 26, 27, 37, 54], nine, 0)
chk("their totals inside 50.5-52.3 g", True, all(50.5 <= tot[n] <= 52.3 for n in nine), 0)
chk("all nine record zero added water", True,
    all((num({int(r["Num"]): r for r in raw}[n], "Water_added_g") or 0) == 0 for n in nine), 0)
chk("renormalisation rescales them by <=4.5%", True,
    max(100 * (tot[n] - 50.0) / tot[n] for n in nine) <= 4.5, 0)

chk("Listed_mass_g reproduces from the masses", 110,
    sum(1 for r in raw if abs((num(r, "Listed_mass_g") or 0)
                              - sum(num(r, k) or 0 for k in LISTED)) < 5e-3), 0)
chk("Surfactant_pct reproduces from the masses", 110,
    sum(1 for r in raw if abs((num(r, "Surfactant_pct") or 0)
                              - sum(num(r, k) or 0 for k in SURF) / 50 * 100) < 5e-3), 0)
dev = [abs((num(r, "Surfactant_pct_stated") or 0)
           - sum(num(r, k) or 0 for k in SURF) / 50 * 100) for r in raw]
chk("Surfactant_pct_stated disagrees on", 56, sum(1 for d in dev if d > 5e-3), 0)
chk("  worst disagreement (pp)", 5.8, max(dev), 0.05)

# Quarantine is per cell, never per row: these four row sets are named in project.md.
chk("PDI>1 rows: whole DLS triplet blanked", True,
    all(num(ROW[n], c) is None for n in (54, 60, 61, 63)
        for c in ("pdi", "z_average_nm", "peak_size_nm")), 0)
chk("sub-nm peak rows: peak only, pdi kept", True,
    all(num(ROW[n], "peak_size_nm") is None and num(ROW[n], "pdi") is not None
        for n in (43, 44)), 0)
chk("row 22: zeta and mobility blanked", True,
    all(num(ROW[22], c) is None for c in ("zeta_mv", "mobility_cm2_vs")), 0)
chk("row 30: zeta kept, mobility blanked", True,
    num(ROW[30], "zeta_mv") is not None and num(ROW[30], "mobility_cm2_vs") is None, 0)

# --------------------------------------------------------------- project.md : Decisions
print("\n== Decisions ==")
chk("analysis set (all but row 110)", 109, len(A109), 0)
for col, n in [("stability_days", 110), ("z_average_nm", 21), ("pdi", 21),
               ("peak_size_nm", 19), ("zeta_mv", 11), ("mobility_cm2_vs", 10)]:
    chk("usable n: %s" % col, n, sum(1 for r in clean if num(r, col) is not None), 0)

fails = [r for r in clean if (num(r, "stability_days") or 0) <= 10]
chk("formulations failing inside 10 days", 48, len(fails), 0)
chk("  of those, DLS-measured", 0,
    sum(1 for r in fails if (r.get("dls_measured") or "").strip() == "1"), 0)
chk("PDI -> stability Pearson r (survivors only)", -0.56,
    pearson(*triple(clean, "pdi", "stability_days")), 0.01)


def hlb_mix(r, lecithin_hlb):
    w = [(num(r, "Span80_g") or 0, HLB_SPAN80),
         (num(r, "Tween80_oil_phase_g") or 0, HLB_TWEEN80),
         (num(r, "Tween80_water_phase_g") or 0, HLB_TWEEN80)]
    if lecithin_hlb is not None:
        w.append((num(r, "Lecithin_g") or 0, lecithin_hlb))
    den = sum(m for m, _ in w)
    return sum(m * h for m, h in w) / den if den else None


err = [abs(num(r, "HLB") - hlb_mix(r, HLB_LECITHIN))
       for r in raw if num(r, "HLB") is not None]
chk("client HLB column: mean error at lecithin=4.0", 1.34, sum(err) / len(err), 0.01)
chk("  rows matched within +-0.2", 21, sum(1 for e in err if e <= 0.2), 0)
chk("F110 sheet HLB, lecithin excluded", 8.74,
    (F110["span80"] * HLB_SPAN80
     + (F110["tween_oil"] + F110["tween_water"]) * HLB_TWEEN80)
    / (F110["span80"] + F110["tween_oil"] + F110["tween_water"]), 0.01)
chk("  against the HLB the sheet states", 10.5, F110["stated_hlb"], 0)
# The sheet is short on its own terms: Sur% 8.9 of a 50 g batch is 4.45 g of surfactant,
# but its own listed surfactant masses sum to 2.40 g.
chk("F110 sheet Sur% implies (g in 50 g batch)", 4.45, F110["stated_surf_pct"] / 100 * 50, 0.01)
chk("  its listed surfactant masses sum to (g)", 2.40,
    F110["span80"] + F110["lecithin"] + F110["tween_oil"] + F110["tween_water"], 0.01)

r, p, n = spearman(*triple(A109, "ph", "stability_days"))
chk("pH vs stability rho (109 rows)", 0.05, r, 0.01)
chk("  p", 0.58, p, 0.02)

# --------------------------------------------------------------- project.md : Headline
print("\n== Progress-report headline ==")
r, p, n = spearman(*triple(A109, "lecithin_pct", "stability_days"))
chk("lecithin vs stability, raw rho", 0.50, r, 0.01)
pr, pp, pn = partial(*triple(A109, "lecithin_pct", "stability_days", "num"))
chk("lecithin vs stability, PARTIAL on run order", 0.38, pr, 0.01)
chk("  n", 109, pn, 0)

# Dose-response bins are half-open, [lo, hi), with 0 wt% as its own bin and the top bin
# closed because the maximum is exactly 5.50 wt%.
zero = [num(r, "stability_days") for r in A109 if num(r, "lecithin_pct") == 0]
chk("dose bin 0 wt%: n", 11, len(zero), 0)
chk("  median stability (d)", 10.0, st.median(zero), 0.01)
for lo, hi, n_exp, med in [(0.0, 1.0, 26, 10.0), (1.0, 2.0, 27, 13.0), (2.0, 3.0, 14, 15.0),
                           (3.0, 4.0, 17, 21.0), (4.0, 5.5, 14, 34.5)]:
    v = [num(r, "stability_days") for r in A109
         if 0 < num(r, "lecithin_pct") and
         (lo <= num(r, "lecithin_pct") < hi or (hi == 5.5 and num(r, "lecithin_pct") == 5.5))]
    chk("dose bin %.1f-%.1f wt%%: n" % (lo, hi), n_exp, len(v), 0)
    chk("  median stability (d)", med, st.median(v), 0.01)

sub = [r for r in A109 if num(r, "stability_days") <= 120]
r, p, n = spearman(*triple(sub, "lecithin_pct", "stability_days"))
chk("robustness: drop the 6 long-lived rows", 0.44, r, 0.01)
sub = [r for r in A109 if num(r, "lecithin_pct") > 0]
r, p, n = spearman(*triple(sub, "lecithin_pct", "stability_days"))
chk("robustness: lecithin-bearing rows only", 0.54, r, 0.01)
chk("  n", 98, n, 0)

for col, rho in [("span80_pct", -0.47), ("xanthan_pct", 0.40), ("glycerol_pct", 0.38)]:
    r, p, n = spearman(*triple(A109, col, "stability_days"))
    chk("next lever: %s" % col, rho, r, 0.01)
for col, rho in [("span80_pct", -0.58), ("xanthan_pct", 0.47)]:
    r, p, n = spearman(*triple(A109, col, "lecithin_pct"))
    chk("  %s tracks lecithin at" % col, rho, r, 0.01)

print("\n-- the run-order confound --")
r, p, n = spearman(*triple(A109, "num", "stability_days"))
chk("run order vs stability", -0.39, r, 0.01)
r, p, n = spearman(*triple(A109, "num", "lecithin_pct"))
chk("run order vs lecithin", -0.55, r, 0.01)
for lo, hi, rho in [(1, 20, 0.13), (21, 40, -0.02), (41, 60, 0.11), (61, 80, 0.79)]:
    blk = [r for r in A109 if lo <= int(r["num"]) <= hi]
    r, p, n = spearman(*triple(blk, "lecithin_pct", "stability_days"))
    chk("block %d-%d rho" % (lo, hi), rho, r, 0.015)
tail = [r for r in A109 if int(r["num"]) >= 81]
chk("rows 81-109 stability values (barely vary)", [10.0, 11.0],
    sorted(set(num(r, "stability_days") for r in tail)), 0)

# project.md argues that blocks 21-40 and 41-60 are themselves range-restricted, which is
# why their null results weaken the lecithin effect rather than disprove it. Print the
# spans so that argument stays checkable against the data instead of being taken on trust.
print("\n  per-block spans (project.md: blocks 21-40 and 41-60 are range-restricted)")
for lo, hi in [(1, 20), (21, 40), (41, 60), (61, 80), (81, 109)]:
    blk = [r for r in A109 if lo <= int(r["num"]) <= hi]
    lec = [num(r, "lecithin_pct") for r in blk]
    stb = [num(r, "stability_days") for r in blk]
    print("    rows %3d-%3d   lecithin %.2f-%.2f wt%%   stability %3.0f-%3.0f d   n=%d"
          % (lo, hi, min(lec), max(lec), min(stb), max(stb), len(blk)))

print("\n" + "=" * 78)
if FAILED:
    print("DRIFT on %d figure(s) -- reconcile planning/project.md:" % len(FAILED))
    for f in FAILED:
        print("  - %s" % f)
    sys.exit(1)
print("every figure in planning/project.md matches the data")
