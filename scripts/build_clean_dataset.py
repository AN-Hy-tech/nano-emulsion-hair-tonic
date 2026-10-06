"""Rebuild + renormalise formulations.csv per the 2026-10-06 validation decisions.

Client answers that drive this script:
  - DLS was run only on formulations that already passed the stability screen  -> survivor-only
  - pH was measured; the 5.1-6.5 band is the range appropriate to the oil      -> kept
    phase (a 12-plant herbal blend, oleic-acid-rich; confirmed 2026-10-07 --
    it is NOT olive oil, an earlier note here said so and was wrong). The repeating 15-step ramp is a designed sweep, not a
    fabricated column. It carries no stability signal on the analysis set --
    scripts/stats_report.py owns that figure and drift-checks it; do not copy
    it here, because nothing checks this docstring.
  - Stability_days == 0 means the sample degraded immediately                  -> real data, kept
  - Rows 2-7 / 55-60 are different samples, not a duplication                  -> both kept
  - Quarantine confirmed: expensive tests inform optimisation, not prediction
"""
import csv, os

SRC = r"D:\Workspace\nano-emulsion-hair-tonic\data\raw-data\formulations.csv"
OUT = r"D:\Workspace\nano-emulsion-hair-tonic\data\processed\formulations_clean.csv"
QLOG = r"D:\Workspace\nano-emulsion-hair-tonic\data\processed\quarantine-log.csv"

# component -> clean mass-fraction column name
COMP = [
    ("Oil_Blend_g", "oil_blend_pct"), ("Span80_g", "span80_pct"),
    ("Lecithin_g", "lecithin_pct"), ("Tween80_oil_phase_g", "tween80_oil_pct"),
    ("Extract_g", "extract_pct"), ("Tween80_water_phase_g", "tween80_water_pct"),
    ("Glycerol_g", "glycerol_pct"), ("PG_g", "pg_pct"),
    ("Xanthan_Gum_g", "xanthan_pct"), ("Guar_Gum_g", "guar_pct"),
    ("Water_added_g", "water_pct"),
]
SURF = ["Span80_g", "Lecithin_g", "Tween80_oil_phase_g", "Tween80_water_phase_g"]
HLB_SPAN80, HLB_TWEEN80, HLB_LECITHIN = 4.3, 15.0, 4.0  # 4.0 = best-fit, see report

rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
def f(r, k):
    v = (r.get(k) or "").strip()
    return float(v) if v else None

# ---- quarantine rules: scoped to the offending cells, never a row deletion ----
PDI_INVALID   = {54, 60, 61, 63}        # PDI > 1 is physically impossible
PEAK_INVALID  = {43, 44, 54}            # peak size below molecular dimensions
ZETA_INVALID  = set()                   # row 22 resolved 2026-10-07, see CORRECTIONS
MOB_INVALID   = {30}                    # row 30: mobility 0.0 with zeta -79.4 mV

# ---- client-confirmed corrections to the raw CSV ----
# Applied when building the clean file; data/raw-data/ is never edited. Every entry
# carries who confirmed it and when, and lands in quarantine_reason so it stays visible.
CORRECTIONS = {
    (22, "zeta_mv"): (-78.9,
        "sign typo: the CSV reads +78.9 mV. Confirmed as -78.9 mV by the client "
        "2026-10-07. Consistent with row 22's own mobility of -0.000613, with all 11 "
        "other zeta readings being negative (-60.5 to -79.4), and with row 31 as a near "
        "twin at -78.8 mV / -0.000611."),
}

# ---- rows excluded from the analysis set -- marked, never deleted ----
# The file always keeps all 110 rows; `in_analysis_set` is what modelling filters on.
# Delete an entry here to put the row back; nothing else needs touching.
ANALYSIS_EXCLUDE = {
    110: "composition contradicted by the client's F110 sheet, confirmed 2026-10-06 "
         "as the lab's corrected figures: the sheet records Tween80_oil 0.35 g and "
         "Tween80_water 0.50 g that the CSV leaves blank, and PG 0.03 g against the "
         "CSV's 0.50 g. Not patched here because the sheet omits xanthan and guar, "
         "so its PG 0.03 g may be the CSV's xanthan; excluded until further sheets "
         "arrive.",
}

# ---- Stability_days semantics, confirmed by the client 2026-10-06 ----
# Days from formulation until visible degradation: a row reading 10 was stable for ten
# days and had degraded by the next check. The runs of identical 10s are therefore real
# readings, not an unfilled default, so every row counts as an observed event and nothing
# is right-censored. The earlier unverified_run_of_10 heuristic is retired.

HEADER = ["num"] + [c for _, c in COMP] + [
    "surfactant_total_pct", "hlb_client", "hlb_calc", "ph",
    "stability_days", "stability_quality",
    "z_average_nm", "pdi", "peak_size_nm", "zeta_mv", "mobility_cm2_vs",
    "dls_measured", "in_analysis_set", "renorm_total_g", "quarantine_reason",
]

out, qlog = [], []
for r in rows:
    n = int(r["Num"])
    total = sum(f(r, k) or 0 for k, _ in COMP)          # includes Water_added_g
    reasons = []

    rec = {"num": n, "renorm_total_g": round(total, 3)}
    # ---- renormalise every component to wt% of the actual total ----
    for k, name in COMP:
        rec[name] = round(100 * (f(r, k) or 0) / total, 4)
    # ---- rebuild the derived columns from the masses ----
    surf_g = sum(f(r, k) or 0 for k in SURF)
    rec["surfactant_total_pct"] = round(100 * surf_g / total, 4)
    rec["hlb_client"] = f(r, "HLB")
    hlb_w = {"Span80_g": HLB_SPAN80, "Lecithin_g": HLB_LECITHIN,
             "Tween80_oil_phase_g": HLB_TWEEN80, "Tween80_water_phase_g": HLB_TWEEN80}
    rec["hlb_calc"] = round(sum((f(r, k) or 0) * w for k, w in hlb_w.items()) / surf_g, 3) if surf_g else ""
    # pH: a measured, deliberately swept design factor. Kept as a column; it does not
    # yet earn a place in the feature set: no signal against stability_days. The figure
    # lives in planning/project.md and is checked by scripts/stats_report.py.
    rec["ph"] = f(r, "pH")

    # ---- responses, with quarantined cells blanked ----
    z, pdi, peak = f(r, "Z_Average_nm"), f(r, "PDI"), f(r, "Particle_Size_nm")
    zeta, mob = f(r, "Zeta_mV"), f(r, "Electrophoretic_Mobility_cm2_Vs")
    rec["dls_measured"] = 1 if z is not None else 0

    if n in PDI_INVALID:
        reasons.append("PDI=%.3f > 1: cumulants fit invalid, whole DLS triplet dropped" % pdi)
        z = pdi = peak = None
    if n in PEAK_INVALID and peak is not None:
        reasons.append("peak size %.1f nm below molecular dimensions" % peak)
        peak = None
    if (n, "zeta_mv") in CORRECTIONS:
        zeta, why = CORRECTIONS[(n, "zeta_mv")]
        reasons.append("zeta corrected to %.1f mV -- %s" % (zeta, why))
    if n in ZETA_INVALID:
        reasons.append("zeta +%.1f mV contradicts negative mobility; sign convention unresolved" % zeta)
        zeta = None
    if n in MOB_INVALID and mob is not None:
        reasons.append("mobility %.6f inconsistent with zeta %.1f mV" % (mob, f(r, "Zeta_mV")))
        mob = None

    rec["z_average_nm"] = "" if z is None else z
    rec["pdi"] = "" if pdi is None else pdi
    rec["peak_size_nm"] = "" if peak is None else peak
    rec["zeta_mv"] = "" if zeta is None else zeta
    rec["mobility_cm2_vs"] = "" if mob is None else mob

    rec["stability_days"] = f(r, "Stability_days")
    rec["stability_quality"] = "measured"
    if abs(total - 50) > 0.5:
        reasons.append("pre-renormalisation total %.2f g vs 50 g batch" % total)
    rec["in_analysis_set"] = 0 if n in ANALYSIS_EXCLUDE else 1
    if n in ANALYSIS_EXCLUDE:
        reasons.append("excluded from the analysis set: " + ANALYSIS_EXCLUDE[n])

    rec["quarantine_reason"] = "; ".join(reasons)
    out.append(rec)
    for why in reasons:
        qlog.append({"num": n, "reason": why})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HEADER); w.writeheader(); w.writerows(out)
with open(QLOG, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["num", "reason"]); w.writeheader(); w.writerows(qlog)

# ---------------- verification ----------------
print("wrote %s  (%d rows x %d cols)" % (OUT, len(out), len(HEADER)))
print("wrote %s  (%d entries)" % (QLOG, len(qlog)))
bad = [r["num"] for r in out if abs(sum(r[c] for _, c in COMP) - 100) > 1e-3]
print("\nrenormalisation closes to 100 wt%%: %d/110 (failures: %s)" % (110 - len(bad), bad or "none"))
print("surfactant_total_pct range: %.2f - %.2f wt%%" % (
    min(r["surfactant_total_pct"] for r in out), max(r["surfactant_total_pct"] for r in out)))
print("\nusable n per response (after quarantine):")
for c in ["stability_days", "z_average_nm", "pdi", "peak_size_nm", "zeta_mv", "mobility_cm2_vs"]:
    print("  %-18s %d" % (c, sum(1 for r in out if r[c] != "")))
print("  all stability readings are observed events (client-confirmed, none censored)")
keep = [r for r in out if r["in_analysis_set"] == 1]
print("\nanalysis set: %d of %d rows (excluded: %s)" % (
    len(keep), len(out), ", ".join(str(n) for n in sorted(ANALYSIS_EXCLUDE))))
print("  rows with any quarantine note   %d" % sum(1 for r in out if r["quarantine_reason"]))
print("\nhlb_client vs hlb_calc: mean abs diff %.2f, within 0.2 on %d/110" % (
    sum(abs(r["hlb_client"] - r["hlb_calc"]) for r in out if r["hlb_calc"] != "") / 110,
    sum(1 for r in out if r["hlb_calc"] != "" and abs(r["hlb_client"] - r["hlb_calc"]) <= 0.2)))
