#!/usr/bin/env python3
"""
Bootstrap effect sizes + 95% CIs
=================================

Replaces the bare Wilcoxon p-values in numerical_analysis.py with the
quantities reviewers actually want:

  * Cliff's δ  (rank-based, non-parametric, range [-1, +1])
  * Hedges' g  (small-sample-corrected standardized mean difference)
  * Bootstrap 95% confidence intervals around both

Sign convention: positive value = the *memory* (active mandate-refresh)
agent wins on this metric.  We auto-flip the sign per metric so the reader
never has to remember whether higher or lower is better.

Inputs
------
  - results/master_summary_*.csv          (from run_experiments.py)
  - analysis/outputs/persona_classifier/drift_scores.csv  (optional, from
    persona_classifier.py score)

Outputs
-------
  - analysis/outputs/effect_sizes/effect_size_table.csv  (paper table)
  - analysis/outputs/effect_sizes/forest_plot.png         (paper figure)

Usage
-----
    python analysis/effect_sizes.py
    python analysis/effect_sizes.py --n-boot 5000   # faster for iteration
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────

# Mapping: which metrics to evaluate + whether higher is better.
# We sign-flip the difference so positive = memory wins for all of them.
METRICS = {
    "MAS_Deviation":      {"higher_better": False, "short": "MAS (drift)"},
    "Max_Drawdown_Pct":   {"higher_better": False, "short": "Max DD"},
    "Rationality_Score":  {"higher_better": True,  "short": "Rationality"},
    "Return_Pct":         {"higher_better": True,  "short": "Return"},
    "Trade_Count":        {"higher_better": False, "short": "Trade churn"},
}

PAIR_KEYS = ["Model", "Persona", "Scenario", "Seed", "Crash_Discount"]


# ──────────────────────────────────────────────────────────────────────────
#  STATISTICS
# ──────────────────────────────────────────────────────────────────────────

def paired_cliffs_delta(diffs: np.ndarray) -> float:
    """Cliff's δ of paired differences against zero. Range [-1, 1].
    +1 = every memory-static difference favors memory; -1 = the reverse."""
    diffs = np.asarray(diffs, dtype=float)
    n_pos = float(np.sum(diffs > 0))
    n_neg = float(np.sum(diffs < 0))
    if len(diffs) == 0:
        return float("nan")
    return (n_pos - n_neg) / len(diffs)


def hedges_g_paired(diffs: np.ndarray) -> float:
    """Hedges' g for paired differences (small-sample-corrected d_z)."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return float("nan")
    m = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float("nan")
    d_z = m / sd
    # Hedges (1981) small-sample correction
    denom = 4 * (n - 1) - 1
    J = 1.0 - 3.0 / denom if denom > 0 else 1.0
    return J * d_z


def bootstrap_ci(diffs: np.ndarray, statistic, n_boot: int = 10_000,
                  ci: float = 0.95, seed: int = 42):
    """Percentile bootstrap CI for `statistic(diffs_resampled)`."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        samples[b] = statistic(diffs[idx])
    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(samples, 100 * alpha))
    hi = float(np.nanpercentile(samples, 100 * (1 - alpha)))
    return (lo, hi)


# ──────────────────────────────────────────────────────────────────────────
#  PAIRING + COMPUTE
# ──────────────────────────────────────────────────────────────────────────

def _pair_static_memory(df: pd.DataFrame, metric: str, higher_better: bool):
    """Pair static vs memory rows by (Model, Persona, Scenario, Seed, Crash_Discount).
    Returns DataFrame with a `diff` column = signed so positive means memory wins."""
    keys = [k for k in PAIR_KEYS if k in df.columns]
    cols = keys + [metric]
    s = df[df["Agent_Type"] == "static"][cols].copy()
    m = df[df["Agent_Type"] == "memory"][cols].copy()
    merged = s.merge(m, on=keys, suffixes=("_s", "_m")).dropna()
    if merged.empty:
        return merged
    # Sign so positive = memory wins
    if higher_better:
        merged["diff"] = merged[f"{metric}_m"] - merged[f"{metric}_s"]
    else:
        merged["diff"] = merged[f"{metric}_s"] - merged[f"{metric}_m"]
    return merged


def compute_effects_for_metric(scen_df, scenario, metric, info,
                                n_boot, ci, seed):
    """Per (scenario, metric): pair, compute Cliff's δ, Hedges' g, CIs."""
    paired = _pair_static_memory(scen_df, metric, info["higher_better"])
    if len(paired) < 3:
        return None
    diffs = paired["diff"].to_numpy()

    cliffs        = paired_cliffs_delta(diffs)
    cliffs_lo, cliffs_hi = bootstrap_ci(diffs, paired_cliffs_delta, n_boot, ci, seed)

    hedges        = hedges_g_paired(diffs)
    hedges_lo, hedges_hi = bootstrap_ci(diffs, hedges_g_paired, n_boot, ci, seed)

    return {
        "Scenario":      scenario,
        "Metric":        metric,
        "Short":         info["short"],
        "Higher_Better": info["higher_better"],
        "N_pairs":       int(len(paired)),
        "Mean_Static":   round(float(paired[f"{metric}_s"].mean()), 4),
        "Mean_Memory":   round(float(paired[f"{metric}_m"].mean()), 4),
        "Mean_Diff":     round(float(np.mean(diffs)), 4),
        "Cliffs_delta":  round(cliffs, 4),
        "Cliffs_CI_lo":  round(cliffs_lo, 4),
        "Cliffs_CI_hi":  round(cliffs_hi, 4),
        "Hedges_g":      round(hedges, 4),
        "Hedges_CI_lo":  round(hedges_lo, 4),
        "Hedges_CI_hi":  round(hedges_hi, 4),
        # Significant if the Cliff's δ CI excludes 0
        "Significant_95": bool(not (cliffs_lo <= 0 <= cliffs_hi)),
    }


def compute_behavioral_effects(master_df, n_boot, ci, seed):
    """For every (scenario, metric) cell, compute effect sizes + CIs."""
    # Pass-only, default crash discount only (don't inflate with sensitivity sweep)
    df = master_df[master_df["Status"].astype(str).str.startswith("PASS")].copy()
    if "Crash_Discount" in df.columns:
        df = df[(df["Scenario"] != "crash") | (df["Crash_Discount"] == 0.92)]

    rows = []
    for scenario in sorted(df["Scenario"].unique()):
        scen_df = df[df["Scenario"] == scenario]
        for metric, info in METRICS.items():
            if metric not in scen_df.columns:
                continue
            r = compute_effects_for_metric(scen_df, scenario, metric, info,
                                            n_boot, ci, seed)
            if r is not None:
                rows.append(r)
    return pd.DataFrame(rows)


def compute_drift_effects(drift_df, n_boot, ci, seed):
    """Per (scenario), pair the per-run mean p_intended of static vs memory."""
    # persona_classifier.py writes the persona column as "MBTI"; effect_sizes
    # uses "Persona" elsewhere (matching master_summary). Normalize.
    if "Persona" not in drift_df.columns and "MBTI" in drift_df.columns:
        drift_df = drift_df.rename(columns={"MBTI": "Persona"})

    # Aggregate per-run (mean over days)
    agg = (drift_df
           .groupby(["Model", "Persona", "Scenario", "Seed", "Agent_Type"])
           ["p_intended"].mean()
           .reset_index())

    rows = []
    info = {"higher_better": True, "short": "Persona adherence"}
    for scenario in sorted(agg["Scenario"].unique()):
        scen_df = agg[agg["Scenario"] == scenario]
        # Treat p_intended as the "metric" for our pairing function
        keys = ["Model", "Persona", "Scenario", "Seed"]
        s = scen_df[scen_df["Agent_Type"] == "static"][keys + ["p_intended"]]
        m = scen_df[scen_df["Agent_Type"] == "memory"][keys + ["p_intended"]]
        merged = s.merge(m, on=keys, suffixes=("_s", "_m")).dropna()
        if len(merged) < 3:
            continue
        diffs = (merged["p_intended_m"] - merged["p_intended_s"]).to_numpy()

        cliffs = paired_cliffs_delta(diffs)
        cliffs_lo, cliffs_hi = bootstrap_ci(diffs, paired_cliffs_delta, n_boot, ci, seed)
        hedges = hedges_g_paired(diffs)
        hedges_lo, hedges_hi = bootstrap_ci(diffs, hedges_g_paired, n_boot, ci, seed)

        rows.append({
            "Scenario":      scenario,
            "Metric":        "p_intended_persona",
            "Short":         info["short"],
            "Higher_Better": True,
            "N_pairs":       int(len(merged)),
            "Mean_Static":   round(float(merged["p_intended_s"].mean()), 4),
            "Mean_Memory":   round(float(merged["p_intended_m"].mean()), 4),
            "Mean_Diff":     round(float(np.mean(diffs)), 4),
            "Cliffs_delta":  round(cliffs, 4),
            "Cliffs_CI_lo":  round(cliffs_lo, 4),
            "Cliffs_CI_hi":  round(cliffs_hi, 4),
            "Hedges_g":      round(hedges, 4),
            "Hedges_CI_lo":  round(hedges_lo, 4),
            "Hedges_CI_hi":  round(hedges_hi, 4),
            "Significant_95": bool(not (cliffs_lo <= 0 <= cliffs_hi)),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
#  PLOT
# ──────────────────────────────────────────────────────────────────────────

def forest_plot(effects, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if effects.empty:
        print("No effects to plot.")
        return

    # Order rows: scenario-group, then metric
    scen_order = {"flat": 0, "bull_trap": 1, "crash": 2}
    effects = effects.copy()
    effects["_scen_idx"] = effects["Scenario"].map(scen_order).fillna(99)
    effects = effects.sort_values(["_scen_idx", "Metric"]).reset_index(drop=True)

    y = np.arange(len(effects))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * len(effects) + 1)))

    colors = ["#1B998B" if s else "#94A3B8" for s in effects["Significant_95"]]

    err_lo = effects["Cliffs_delta"] - effects["Cliffs_CI_lo"]
    err_hi = effects["Cliffs_CI_hi"] - effects["Cliffs_delta"]

    ax.errorbar(effects["Cliffs_delta"], y, xerr=[err_lo, err_hi],
                fmt="none", ecolor="#475569", elinewidth=1.4, capsize=4, zorder=2)
    ax.scatter(effects["Cliffs_delta"], y, s=90, c=colors,
               edgecolors="#0F172A", linewidth=0.8, zorder=3)

    # Faint scenario-group separators
    for i in range(1, len(effects)):
        if effects.loc[i, "Scenario"] != effects.loc[i - 1, "Scenario"]:
            ax.axhline(i - 0.5, color="#E2E8F0", linewidth=0.8, zorder=0)

    ax.axvline(0, color="#0F172A", linestyle=":", linewidth=1.2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.Scenario}  ·  {r.Short}" for r in effects.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("Cliff's δ  (positive = memory agent wins)", fontsize=11)
    ax.set_title(
        "Paired effect sizes with 95% bootstrap CIs\n"
        "Green = CI excludes 0  ·  Grey = not significant",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", default="results",
                   help="Where master_summary_*.csv lives (default: results)")
    p.add_argument("--drift-scores",
                   default="analysis/outputs/persona_classifier/drift_scores.csv",
                   help="Per-rationale classifier scores from persona_classifier.py")
    p.add_argument("--output-dir", default="analysis/outputs/effect_sizes")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="Bootstrap iterations (default 10000; 5000 ~ 2x faster, slightly noisier)")
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all master summary CSVs
    results_dir = Path(args.results_dir)
    master_files = sorted(results_dir.glob("master_summary_*.csv"))
    if not master_files:
        print(f"ERROR: no master_summary_*.csv found under {results_dir}")
        return 1

    dfs = []
    for f in master_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  WARN: could not read {f}: {e}")
    master_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    n_pass = (master_df["Status"].astype(str).str.startswith("PASS")).sum()
    print(f"[Data] Master summary: {len(master_df):,} rows ({n_pass:,} PASS) "
          f"from {len(master_files)} file(s)")

    # Behavioral metrics
    print(f"[Compute] Behavioral effect sizes  (bootstrap n={args.n_boot})...")
    behav = compute_behavioral_effects(master_df, args.n_boot, args.ci, args.seed)
    print(f"          {len(behav)} (scenario × metric) cells computed")

    all_parts = [behav]

    # Linguistic drift metric (if persona_classifier.py score has been run)
    drift_path = Path(args.drift_scores)
    if drift_path.exists():
        drift_df = pd.read_csv(drift_path)
        print(f"[Data] Drift scores: {len(drift_df):,} rationale-level rows")
        print("[Compute] Linguistic drift effect sizes...")
        drift_eff = compute_drift_effects(drift_df, args.n_boot, args.ci, args.seed)
        print(f"          {len(drift_eff)} scenarios computed")
        all_parts.append(drift_eff)
    else:
        print(f"[Skip] Linguistic drift effect sizes — {drift_path} not found.")
        print("       Run `python analysis/persona_classifier.py score` first to include it.")

    effects = pd.concat(all_parts, ignore_index=True)

    # Save table
    csv_path = output_dir / "effect_size_table.csv"
    effects.to_csv(csv_path, index=False)

    # Print readable summary
    print(f"\n{'=' * 78}")
    print("Effect-size summary  (positive = memory agent wins, * = 95% CI excludes 0)")
    print('=' * 78)
    print(
        f"{'Scenario':<11} {'Metric':<20} {'N':>3} "
        f"{'Mean_S':>9} {'Mean_M':>9}  "
        f"{'Cliff δ [95% CI]':<22} "
        f"{'Hedges g':>9} "
        f"{'sig':>4}"
    )
    print("-" * 78)
    for _, r in effects.iterrows():
        sig = "  *" if r["Significant_95"] else ""
        cliffs_str = f"{r['Cliffs_delta']:+.2f} [{r['Cliffs_CI_lo']:+.2f},{r['Cliffs_CI_hi']:+.2f}]"
        print(
            f"{r['Scenario']:<11} {r['Short']:<20} {r['N_pairs']:>3} "
            f"{r['Mean_Static']:>9.3f} {r['Mean_Memory']:>9.3f}  "
            f"{cliffs_str:<22} "
            f"{r['Hedges_g']:>+9.2f} "
            f"{sig:>4}"
        )

    # Forest plot
    plot_path = output_dir / "forest_plot.png"
    forest_plot(effects, plot_path)
    print(f"\nSaved:")
    print(f"  table → {csv_path}")
    print(f"  plot  → {plot_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
