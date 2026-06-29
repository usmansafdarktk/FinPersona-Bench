#!/usr/bin/env python3
"""
Per-Slice Effect-Size Breakdown
================================

Splits the headline effect-size table into three slicings:

  * per_model.csv          — does memory help smaller models more?  (scaling)
  * per_persona.csv        — which persona benefits most from memory?
  * per_model_persona.csv  — interaction effects

Each row reports Cliff's δ + bootstrap 95% CI + Hedges' g + significance,
restricted to the slice's paired observations.

Figures
-------
  * per_model_scaling.png       — bars per model, sorted by approximate size
  * per_persona_bars.png        — bars per persona, color-coded by scenario
  * heatmap_model_persona.png   — (model × persona) heatmap of linguistic effect size

Usage
-----
    python analysis/drift_breakdown.py
    python analysis/drift_breakdown.py --n-boot 5000   # faster
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
#  CONFIG (mirrors effect_sizes.py so the two scripts agree)
# ──────────────────────────────────────────────────────────────────────────

METRICS = {
    "MAS_Deviation":      {"higher_better": False, "short": "MAS"},
    "Max_Drawdown_Pct":   {"higher_better": False, "short": "MaxDD"},
    "Rationality_Score":  {"higher_better": True,  "short": "Rationality"},
    "Return_Pct":         {"higher_better": True,  "short": "Return"},
    "Trade_Count":        {"higher_better": False, "short": "TradeChurn"},
}

PAIR_KEYS = ["Model", "Persona", "Scenario", "Seed", "Crash_Discount"]

# Rough parameter counts (Billions) for the scaling-axis figure.
# API models get tier-based ordinal proxies.
MODEL_PARAMS_B = {
    # vLLM (open weight) — actual sizes
    "google/gemma-3-4b-it":               4.0,
    "Qwen/Qwen2.5-7B-Instruct":           7.0,
    "meta-llama/Llama-3.1-8B-Instruct":   8.0,
    "google/gemma-2-9b-it":               9.0,
    "Qwen/Qwen2.5-14B-Instruct":         14.0,
    "google/gemma-3-27b-it":             27.0,
    # API (sizes are guesses; used only for sort order on the x-axis)
    "gpt-4o-mini":                        8.0,    # ≈
    "gemini-2.5-flash":                  35.0,    # placeholder
    "gemini-3-flash-preview":            40.0,    # placeholder
    "claude-sonnet-4-6":                100.0,    # placeholder
    "claude-opus-4-7":                  175.0,    # placeholder
}


# ──────────────────────────────────────────────────────────────────────────
#  STATS
# ──────────────────────────────────────────────────────────────────────────

def paired_cliffs_delta(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return float("nan")
    return (np.sum(diffs > 0) - np.sum(diffs < 0)) / n


def hedges_g_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        return float("nan")
    d_z = np.mean(diffs) / sd
    denom = 4 * (n - 1) - 1
    J = 1.0 - 3.0 / denom if denom > 0 else 1.0
    return J * d_z


def bootstrap_ci(diffs, statistic, n_boot=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = statistic(diffs[idx])
    alpha = (1 - ci) / 2
    return (float(np.nanpercentile(out, 100 * alpha)),
            float(np.nanpercentile(out, 100 * (1 - alpha))))


# ──────────────────────────────────────────────────────────────────────────
#  PAIRING + COMPUTE
# ──────────────────────────────────────────────────────────────────────────

def _signed_diff(merged: pd.DataFrame, metric: str, higher_better: bool) -> np.ndarray:
    if higher_better:
        return (merged[f"{metric}_m"] - merged[f"{metric}_s"]).to_numpy()
    return (merged[f"{metric}_s"] - merged[f"{metric}_m"]).to_numpy()


def _pair(df: pd.DataFrame, on_keys: list, value_col: str) -> pd.DataFrame:
    """Pair static vs memory rows on `on_keys`. Each row must have an Agent_Type column."""
    s = df[df["Agent_Type"] == "static"][on_keys + [value_col]]
    m = df[df["Agent_Type"] == "memory"][on_keys + [value_col]]
    return s.merge(m, on=on_keys, suffixes=("_s", "_m")).dropna()


def _effect_stats(diffs, n_boot, ci, seed):
    cliffs = paired_cliffs_delta(diffs)
    cliffs_lo, cliffs_hi = bootstrap_ci(diffs, paired_cliffs_delta, n_boot, ci, seed)
    hedges = hedges_g_paired(diffs)
    hedges_lo, hedges_hi = bootstrap_ci(diffs, hedges_g_paired, n_boot, ci, seed)
    return {
        "Cliffs_delta":   round(cliffs, 4),
        "Cliffs_CI_lo":   round(cliffs_lo, 4),
        "Cliffs_CI_hi":   round(cliffs_hi, 4),
        "Hedges_g":       round(hedges, 4),
        "Hedges_CI_lo":   round(hedges_lo, 4),
        "Hedges_CI_hi":   round(hedges_hi, 4),
        "Significant_95": bool(not (cliffs_lo <= 0 <= cliffs_hi)),
    }


def slice_behavioral(master_df, slice_cols, n_boot, ci, seed):
    """For each (slice × scenario × metric), pair static vs memory and compute effects."""
    df = master_df[master_df["Status"].astype(str).str.startswith("PASS")].copy()
    if "Crash_Discount" in df.columns:
        df = df[(df["Scenario"] != "crash") | (df["Crash_Discount"] == 0.92)]

    # Determine pairing keys: everything in PAIR_KEYS except the slice
    keys_used = [k for k in PAIR_KEYS if k in df.columns and k not in slice_cols]

    rows = []
    for slice_vals, slice_df in df.groupby(slice_cols, dropna=False):
        if not isinstance(slice_vals, tuple):
            slice_vals = (slice_vals,)
        for scenario in sorted(slice_df["Scenario"].unique()):
            scen_df = slice_df[slice_df["Scenario"] == scenario]
            for metric, info in METRICS.items():
                if metric not in scen_df.columns:
                    continue
                merged = _pair(scen_df, on_keys=[k for k in keys_used if k != "Scenario"]
                               + ["Scenario"], value_col=metric)
                if len(merged) < 3:
                    continue
                diffs = _signed_diff(merged, metric, info["higher_better"])
                row = dict(zip(slice_cols, slice_vals))
                row.update({
                    "Scenario":     scenario,
                    "Metric":       metric,
                    "Short":        info["short"],
                    "Higher_Better": info["higher_better"],
                    "N_pairs":      int(len(merged)),
                    "Mean_Static":  round(float(merged[f"{metric}_s"].mean()), 4),
                    "Mean_Memory":  round(float(merged[f"{metric}_m"].mean()), 4),
                    "Mean_Diff":    round(float(np.mean(diffs)), 4),
                    **_effect_stats(diffs, n_boot, ci, seed),
                })
                rows.append(row)
    return pd.DataFrame(rows)


def slice_drift(drift_df, slice_cols, n_boot, ci, seed):
    """Same as slice_behavioral but on the linguistic p_intended metric."""
    if "Persona" not in drift_df.columns and "MBTI" in drift_df.columns:
        drift_df = drift_df.rename(columns={"MBTI": "Persona"})

    agg = (drift_df
           .groupby(["Model", "Persona", "Scenario", "Seed", "Agent_Type"])
           ["p_intended"].mean()
           .reset_index())

    keys_used = [k for k in ["Model", "Persona", "Scenario", "Seed"]
                 if k not in slice_cols]

    rows = []
    for slice_vals, slice_df in agg.groupby(slice_cols, dropna=False):
        if not isinstance(slice_vals, tuple):
            slice_vals = (slice_vals,)
        for scenario in sorted(slice_df["Scenario"].unique()):
            scen_df = slice_df[slice_df["Scenario"] == scenario]
            merged = _pair(scen_df,
                            on_keys=[k for k in keys_used if k != "Scenario"] + ["Scenario"],
                            value_col="p_intended")
            if len(merged) < 3:
                continue
            diffs = (merged["p_intended_m"] - merged["p_intended_s"]).to_numpy()
            row = dict(zip(slice_cols, slice_vals))
            row.update({
                "Scenario":      scenario,
                "Metric":        "p_intended_persona",
                "Short":         "PersonaAdherence",
                "Higher_Better": True,
                "N_pairs":       int(len(merged)),
                "Mean_Static":   round(float(merged["p_intended_s"].mean()), 4),
                "Mean_Memory":   round(float(merged["p_intended_m"].mean()), 4),
                "Mean_Diff":     round(float(np.mean(diffs)), 4),
                **_effect_stats(diffs, n_boot, ci, seed),
            })
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
#  PLOTS
# ──────────────────────────────────────────────────────────────────────────

def _safe_import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_per_model_scaling(df: pd.DataFrame, metric_focus: str, out_path: Path, title: str):
    """Bar/line of Cliff's δ vs model size, restricted to one metric."""
    plt = _safe_import_plt()
    sub = df[df["Metric"] == metric_focus].copy()
    if sub.empty:
        print(f"[Plot] No data for metric={metric_focus}, skipping {out_path}")
        return

    sub["params"] = sub["Model"].map(MODEL_PARAMS_B).fillna(1.0)
    sub = sub.sort_values("params")

    scenarios = sorted(sub["Scenario"].unique())
    colors = {"flat": "#1B998B", "bull_trap": "#E84855", "crash": "#F4A261"}

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for sc in scenarios:
        scen = sub[sub["Scenario"] == sc].sort_values("params")
        if scen.empty:
            continue
        x = np.arange(len(scen))
        y = scen["Cliffs_delta"].values
        err_lo = (scen["Cliffs_delta"] - scen["Cliffs_CI_lo"]).values
        err_hi = (scen["Cliffs_CI_hi"] - scen["Cliffs_delta"]).values
        ax.errorbar(x, y, yerr=[err_lo, err_hi], fmt="o-",
                    color=colors.get(sc, "gray"),
                    linewidth=2, markersize=8, capsize=4, label=sc)

    # Use one scenario for the x-axis labels (any one — they're sorted the same)
    label_scen = sub[sub["Scenario"] == scenarios[0]].sort_values("params")
    ax.set_xticks(np.arange(len(label_scen)))
    ax.set_xticklabels([m.split("/")[-1][:22] for m in label_scen["Model"]],
                        rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Cliff's δ  (memory − static)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(title="scenario", fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def plot_per_persona_bars(df: pd.DataFrame, metric_focus: str, out_path: Path, title: str):
    """Grouped bars: x=persona, hue=scenario."""
    plt = _safe_import_plt()
    sub = df[df["Metric"] == metric_focus].copy()
    if sub.empty:
        print(f"[Plot] No data for metric={metric_focus}, skipping {out_path}")
        return

    personas = sorted(sub["Persona"].unique())
    scenarios = sorted(sub["Scenario"].unique())
    colors = {"flat": "#1B998B", "bull_trap": "#E84855", "crash": "#F4A261"}

    x = np.arange(len(personas))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, sc in enumerate(scenarios):
        ys, los, his = [], [], []
        for p in personas:
            row = sub[(sub["Persona"] == p) & (sub["Scenario"] == sc)]
            if row.empty:
                ys.append(np.nan); los.append(0); his.append(0)
            else:
                r = row.iloc[0]
                ys.append(r["Cliffs_delta"])
                los.append(r["Cliffs_delta"] - r["Cliffs_CI_lo"])
                his.append(r["Cliffs_CI_hi"] - r["Cliffs_delta"])
        ax.bar(x + (i - 1) * width, ys, width, yerr=[los, his],
               color=colors.get(sc, "gray"), label=sc,
               edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(personas, fontsize=11)
    ax.set_ylabel("Cliff's δ  (memory − static)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.legend(title="scenario", fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def plot_model_persona_heatmap(df: pd.DataFrame, metric_focus: str, out_path: Path,
                                 title: str):
    """Heatmap: rows=Model (sorted by size), cols=Persona, color=Cliff's δ."""
    plt = _safe_import_plt()
    sub = df[df["Metric"] == metric_focus].copy()
    if sub.empty:
        print(f"[Plot] No data for metric={metric_focus}, skipping {out_path}")
        return

    # Aggregate over scenarios per (model, persona)
    grid = (sub.groupby(["Model", "Persona"])["Cliffs_delta"]
            .mean().reset_index()
            .pivot(index="Model", columns="Persona", values="Cliffs_delta"))
    # Sort rows by approximate params
    grid["__sort"] = grid.index.map(lambda m: MODEL_PARAMS_B.get(m, 9999))
    grid = grid.sort_values("__sort").drop(columns="__sort")

    fig, ax = plt.subplots(figsize=(7, max(4, 0.45 * len(grid) + 2)))
    vmax = max(0.05, np.nanmax(np.abs(grid.values)))
    im = ax.imshow(grid.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(grid.columns)))
    ax.set_xticklabels(grid.columns, fontsize=11)
    ax.set_yticks(np.arange(len(grid.index)))
    ax.set_yticklabels([m.split("/")[-1][:24] for m in grid.index], fontsize=9)

    # Annotate cells
    for i in range(len(grid.index)):
        for j in range(len(grid.columns)):
            val = grid.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=10, color="black" if abs(val) < 0.5 else "white")

    cbar = fig.colorbar(im, ax=ax, label="Cliff's δ  (memory − static)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-dir", default="results")
    p.add_argument("--drift-scores",
                   default="analysis/outputs/persona_classifier/drift_scores.csv")
    p.add_argument("--output-dir", default="analysis/outputs/breakdown")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load master summaries
    master_files = sorted(Path(args.results_dir).glob("master_summary_*.csv"))
    if not master_files:
        print(f"ERROR: no master_summary_*.csv under {args.results_dir}")
        return 1
    master_df = (pd.concat([pd.read_csv(f) for f in master_files], ignore_index=True)
                   .drop_duplicates())
    n_pass = (master_df["Status"].astype(str).str.startswith("PASS")).sum()
    print(f"[Data] Master summary: {len(master_df):,} rows ({n_pass:,} PASS) "
          f"from {len(master_files)} file(s)")

    # Drift scores (optional)
    drift_df = None
    drift_path = Path(args.drift_scores)
    if drift_path.exists():
        drift_df = pd.read_csv(drift_path)
        print(f"[Data] Drift scores: {len(drift_df):,} rationale-level rows")

    # ── Three slicings ─────────────────────────────────────────────────
    print(f"\n[Compute] Per-model breakdown  (bootstrap n={args.n_boot})...")
    per_model = slice_behavioral(master_df, ["Model"], args.n_boot, args.ci, args.seed)
    if drift_df is not None:
        per_model_drift = slice_drift(drift_df, ["Model"], args.n_boot, args.ci, args.seed)
        per_model = pd.concat([per_model, per_model_drift], ignore_index=True)
    per_model.to_csv(out_dir / "per_model.csv", index=False)
    print(f"          {len(per_model)} rows → per_model.csv")

    print(f"\n[Compute] Per-persona breakdown...")
    per_persona = slice_behavioral(master_df, ["Persona"], args.n_boot, args.ci, args.seed)
    if drift_df is not None:
        per_persona_drift = slice_drift(drift_df, ["Persona"], args.n_boot, args.ci, args.seed)
        per_persona = pd.concat([per_persona, per_persona_drift], ignore_index=True)
    per_persona.to_csv(out_dir / "per_persona.csv", index=False)
    print(f"          {len(per_persona)} rows → per_persona.csv")

    print(f"\n[Compute] Per-(model × persona) breakdown...")
    per_model_persona = slice_behavioral(master_df, ["Model", "Persona"],
                                          args.n_boot, args.ci, args.seed)
    if drift_df is not None:
        per_mp_drift = slice_drift(drift_df, ["Model", "Persona"],
                                    args.n_boot, args.ci, args.seed)
        per_model_persona = pd.concat([per_model_persona, per_mp_drift], ignore_index=True)
    per_model_persona.to_csv(out_dir / "per_model_persona.csv", index=False)
    print(f"          {len(per_model_persona)} rows → per_model_persona.csv")

    # ── Print summary tables to stdout ─────────────────────────────────
    def _print_summary(df, slice_cols, name):
        if df.empty:
            return
        print(f"\n{'=' * 78}")
        print(f"  {name}  (positive = memory wins; * = 95% CI excludes 0)")
        print('=' * 78)
        focus = df[df["Metric"] == "p_intended_persona"]
        if focus.empty:
            focus = df[df["Metric"] == "MAS_Deviation"]
        if focus.empty:
            return
        cols = slice_cols + ["Scenario", "N_pairs", "Cliffs_delta",
                              "Cliffs_CI_lo", "Cliffs_CI_hi", "Hedges_g", "Significant_95"]
        print(focus[cols].round(3).to_string(index=False))

    _print_summary(per_model, ["Model"],
                   "Per-Model (linguistic adherence)")
    _print_summary(per_persona, ["Persona"],
                   "Per-Persona (linguistic adherence)")

    # ── Plots — focus on the linguistic-drift metric ───────────────────
    print("\n[Plot] Rendering figures...")
    if drift_df is not None:
        plot_per_model_scaling(
            per_model, "p_intended_persona",
            out_dir / "per_model_scaling.png",
            title="Memory mitigation effect by model — linguistic persona adherence",
        )
        plot_per_persona_bars(
            per_persona, "p_intended_persona",
            out_dir / "per_persona_bars.png",
            title="Memory mitigation effect by persona — linguistic persona adherence",
        )
        plot_model_persona_heatmap(
            per_model_persona, "p_intended_persona",
            out_dir / "heatmap_model_persona.png",
            title="Memory effect: Cliff's δ across (model × persona), averaged over scenarios",
        )

    # Also produce behavioral-side scaling plots for the paper's appendix
    plot_per_model_scaling(
        per_model, "MAS_Deviation",
        out_dir / "per_model_scaling_MAS.png",
        title="Memory mitigation effect by model — MAS_Deviation (behavioral)",
    )

    print("\nAll outputs in:", out_dir)


if __name__ == "__main__":
    sys.exit(main() or 0)
