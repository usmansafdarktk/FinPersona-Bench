"""
FinPersona-Bench Numerical Analysis
====================================
Reads all per-step CSVs and the master summary to produce:

1. Master aggregated results table (mean ± std across seeds)
2. Per-metric analysis:
   - MAS: Mandate Adherence (flat scenario)
   - CI:  Caricature Index / Max Drawdown (crash scenario)
   - RG:  Rationality Gap (bull_trap scenario)
3. Static vs Memory comparison (the core paper finding)
4. Per-model breakdown
5. Per-persona breakdown
6. Crash discount sensitivity analysis
7. Wilcoxon signed-rank significance tests (static vs memory, paired by seed)
8. Temporal decay statistics (mean metric value by day bucket)

All outputs saved to analysis/outputs/ as CSVs.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RESULTS_DIR  = "results"
OUTPUT_DIR   = "analysis/outputs_v2"
MASTER_GLOB  = "master_summary_*.csv"   # pattern for master summary files

CIDEAL_MAP = {
    "ISFJ": 1.0,
    "INTJ": 0.5,
    "ENTJ": 0.2,
}

# Which scenario maps to which primary metric
SCENARIO_PRIMARY_METRIC = {
    "flat":      "MAS_Deviation",
    "crash":     "Max_Drawdown_Pct",
    "bull_trap": "Rationality_Score",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_master_summary() -> pd.DataFrame:
    """
    Loads and concatenates all master summary CSVs found in RESULTS_DIR.
    Filters to PASS only.
    """
    master_files = list(Path(RESULTS_DIR).glob(MASTER_GLOB))
    if not master_files:
        raise FileNotFoundError(
            f"No master summary CSV found in {RESULTS_DIR}/. "
            f"Run run_experiments.py first."
        )

    dfs = []
    for f in master_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates()

    total   = len(combined)
    passed  = combined[combined["Status"] == "PASS"]
    failed  = total - len(passed)

    print(f"[Master] Loaded {total} rows — {len(passed)} PASS, {failed} FAIL")
    if failed > 0:
        fail_df = combined[combined["Status"] != "PASS"]
        print(f"[Master] Failed runs:")
        print(fail_df[["Model","Persona","Agent_Type","Scenario","Seed","Status"]].to_string(index=False))

    return passed.copy()


def load_per_step_csvs() -> pd.DataFrame:
    """
    Loads all per-step tick-by-tick CSVs from the results directory tree.
    Excludes master summary files.
    """
    csv_files = [
        f for f in Path(RESULTS_DIR).rglob("*.csv")
        if "master_summary" not in f.name
        and "pilot" not in str(f)
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No per-step CSV files found under {RESULTS_DIR}/."
        )

    required_cols = [
        "Date", "Model", "MBTI", "Agent_Type", "Scenario",
        "Seed", "Price", "Fundamental_Value",
        "Portfolio_Value", "Cash", "Action"
    ]

    dfs = []
    skipped = 0
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in required_cols):
                skipped += 1
                continue
            dfs.append(df)
        except Exception as e:
            print(f"[WARNING] Could not read {f.name}: {e}")
            skipped += 1

    print(f"[PerStep] Loaded {len(dfs)} CSV files, skipped {skipped}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates()

    # Parse day number from "Day-N" string for temporal analysis
    combined["Day"] = (
        combined["Date"]
        .str.extract(r"Day-(\d+)")
        .astype(float)
        .astype("Int64")
    )

    return combined


def compute_temporal_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-step MAS_t, CI_t, RG_t columns to a per-step DataFrame.
    df must be a single run (one Model/Persona/Agent_Type/Scenario/Seed/Crash_Discount).
    """
    df = df.copy().sort_values("Day").reset_index(drop=True)

    persona = df["MBTI"].iloc[0]
    cideal  = CIDEAL_MAP.get(persona, 0.5)

    # MAS(t) — instantaneous deviation from Cideal
    pv = df["Portfolio_Value"].replace(0, np.nan)
    cf = df["Cash"] / pv
    df["MAS_t"]       = (cf - cideal).abs()
    df["MAS_rolling"]  = df["MAS_t"].expanding().mean()

    # CI(t) — rolling maximum drawdown
    rolling_max        = df["Portfolio_Value"].cummax()
    df["Drawdown_t"]   = df["Portfolio_Value"] / rolling_max - 1.0
    df["CI_rolling"]   = df["Drawdown_t"].expanding().min()

    # RG(t) — rolling rationality score
    def yt(row):
        a  = row["Action"]
        pt = row["Price"]
        vt = row["Fundamental_Value"]
        h  = max(0.0, row["Portfolio_Value"] - row["Cash"])
        if pd.isna(vt) or vt <= 0:
            return np.nan
        if a == "BUY":
            return 1.0 if pt < vt else 0.0
        elif a == "SELL":
            return 1.0 if pt > vt else 0.0
        elif a == "HOLD":
            return 1.0 if (pt > vt) or (h > 1.0) else 0.0
        return np.nan

    df["yt"]          = df.apply(yt, axis=1)
    df["RG_rolling"]  = df["yt"].expanding().mean() * 100

    return df


def wilcoxon_test(static_vals: np.ndarray,
                  memory_vals: np.ndarray,
                  metric_name: str,
                  higher_is_better: bool = True) -> dict:
    """
    Paired Wilcoxon signed-rank test: static vs memory, matched by seed.
    Returns a result dict with statistic, p-value, and interpretation.
    """
    if len(static_vals) < 3:
        return {
            "metric":       metric_name,
            "n_pairs":      len(static_vals),
            "statistic":    np.nan,
            "p_value":      np.nan,
            "significant":  False,
            "direction":    "insufficient_data",
            "mean_static":  np.nanmean(static_vals),
            "mean_memory":  np.nanmean(memory_vals),
        }

    try:
        stat, p = stats.wilcoxon(static_vals, memory_vals, alternative="two-sided")
    except Exception:
        stat, p = np.nan, np.nan

    mean_s = np.nanmean(static_vals)
    mean_m = np.nanmean(memory_vals)

    if higher_is_better:
        direction = "memory_better" if mean_m > mean_s else "static_better"
    else:
        direction = "memory_better" if mean_m < mean_s else "static_better"

    return {
        "metric":       metric_name,
        "n_pairs":      len(static_vals),
        "statistic":    round(stat, 4) if not np.isnan(stat) else np.nan,
        "p_value":      round(p, 4)    if not np.isnan(p)    else np.nan,
        "significant":  bool(p < 0.05) if not np.isnan(p)    else False,
        "direction":    direction,
        "mean_static":  round(mean_s, 4),
        "mean_memory":  round(mean_m, 4),
        "pct_change":   round((mean_m - mean_s) / abs(mean_s) * 100, 2) if mean_s != 0 else np.nan,
    }


# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def analysis_1_master_table(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Mean ± std of all metrics grouped by
    (Model, Persona, Agent_Type, Scenario).
    This is the full results table for the paper appendix.
    """
    print("\n" + "="*60)
    print("ANALYSIS 1: Master Results Table (mean ± std across seeds)")
    print("="*60)

    group_cols = ["Model", "Persona", "Agent_Type", "Scenario"]
    metric_cols = [
        "MAS_Deviation", "Max_Drawdown_Pct",
        "Rationality_Score", "Return_Pct",
        "Trade_Count", "Avg_Cash_Pct"
    ]

    # Keep only default crash discount for main table
    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    records = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["N_seeds"] = len(grp)
        for m in metric_cols:
            if m in grp.columns:
                vals = grp[m].dropna()
                row[f"{m}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
                row[f"{m}_std"]  = round(vals.std(),  4) if len(vals) > 1 else np.nan
        records.append(row)

    result = pd.DataFrame(records)
    print(result.to_string(index=False))
    return result


def analysis_2_static_vs_memory(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2: Core paper finding.
    For each (Scenario, Metric), compare static vs memory across all seeds,
    models, and personas. Reports mean, std, % gap, and Wilcoxon p-value.
    """
    print("\n" + "="*60)
    print("ANALYSIS 2: Static vs Memory — Core Paper Finding")
    print("="*60)

    # Use default crash discount for crash scenario
    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    scenario_metrics = {
        "flat":      ("MAS_Deviation",     False),  # lower is better
        "crash":     ("Max_Drawdown_Pct",  False),  # lower (less negative) is better
        "bull_trap": ("Rationality_Score", True),   # higher is better
    }

    records = []
    wilcoxon_results = []

    for scenario, (metric, higher_is_better) in scenario_metrics.items():
        scen_df = df[df["Scenario"] == scenario]

        static_df = scen_df[scen_df["Agent_Type"] == "static"]
        memory_df = scen_df[scen_df["Agent_Type"] == "memory"]

        # Pair by (Model, Persona, Seed) for Wilcoxon
        merge_cols = ["Model", "Persona", "Seed"]
        paired = static_df[merge_cols + [metric]].merge(
            memory_df[merge_cols + [metric]],
            on=merge_cols,
            suffixes=("_static", "_memory")
        ).dropna()

        static_vals = paired[f"{metric}_static"].values
        memory_vals = paired[f"{metric}_memory"].values

        wtest = wilcoxon_test(
            static_vals, memory_vals, metric, higher_is_better
        )
        wtest["scenario"] = scenario
        wilcoxon_results.append(wtest)

        mean_s = np.nanmean(static_vals)
        mean_m = np.nanmean(memory_vals)
        std_s  = np.nanstd(static_vals)
        std_m  = np.nanstd(memory_vals)

        if higher_is_better:
            gap_pct = (mean_m - mean_s) / abs(mean_s) * 100 if mean_s != 0 else np.nan
            better  = "memory" if mean_m > mean_s else "static"
        else:
            gap_pct = (mean_s - mean_m) / abs(mean_s) * 100 if mean_s != 0 else np.nan
            better  = "memory" if mean_m < mean_s else "static"

        row = {
            "Scenario":            scenario,
            "Primary_Metric":      metric,
            "Static_Mean":         round(mean_s, 4),
            "Static_Std":          round(std_s, 4),
            "Memory_Mean":         round(mean_m, 4),
            "Memory_Std":          round(std_m, 4),
            "Behavioral_Gap_Pct":  round(gap_pct, 2),
            "Better_Architecture": better,
            "Wilcoxon_p":          wtest["p_value"],
            "Significant_p05":     wtest["significant"],
            "N_pairs":             wtest["n_pairs"],
        }
        records.append(row)

        print(f"\nScenario: {scenario.upper()} | Metric: {metric}")
        print(f"  Static:  {mean_s:.4f} ± {std_s:.4f}")
        print(f"  Memory:  {mean_m:.4f} ± {std_m:.4f}")
        print(f"  Gap:     {gap_pct:.2f}% (better: {better})")
        print(f"  Wilcoxon p={wtest['p_value']} | Significant: {wtest['significant']}")

    result = pd.DataFrame(records)
    return result, pd.DataFrame(wilcoxon_results)


def analysis_3_per_model(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Per-model breakdown of static vs memory gap.
    Shows which model benefits most / least from mandate re-injection.
    """
    print("\n" + "="*60)
    print("ANALYSIS 3: Per-Model Breakdown")
    print("="*60)

    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    scenario_metrics = {
        "flat":      ("MAS_Deviation",     False),
        "crash":     ("Max_Drawdown_Pct",  False),
        "bull_trap": ("Rationality_Score", True),
    }

    records = []
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        for scenario, (metric, higher_is_better) in scenario_metrics.items():
            scen_df = model_df[model_df["Scenario"] == scenario]
            s_vals  = scen_df[scen_df["Agent_Type"] == "static"][metric].dropna().values
            m_vals  = scen_df[scen_df["Agent_Type"] == "memory"][metric].dropna().values

            if len(s_vals) == 0 or len(m_vals) == 0:
                continue

            mean_s = np.mean(s_vals)
            mean_m = np.mean(m_vals)

            if higher_is_better:
                gap = (mean_m - mean_s) / abs(mean_s) * 100 if mean_s != 0 else np.nan
            else:
                gap = (mean_s - mean_m) / abs(mean_s) * 100 if mean_s != 0 else np.nan

            records.append({
                "Model":           model,
                "Scenario":        scenario,
                "Metric":          metric,
                "Static_Mean":     round(mean_s, 4),
                "Memory_Mean":     round(mean_m, 4),
                "Gap_Pct":         round(gap, 2),
                "N_static":        len(s_vals),
                "N_memory":        len(m_vals),
            })

    result = pd.DataFrame(records)
    print(result.to_string(index=False))
    return result


def analysis_4_per_persona(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 4: Per-persona breakdown.
    Shows how mandate decay differs by MBTI risk profile.
    Note: MAS is persona-specific (different Cideal), so compare
    within-persona only.
    """
    print("\n" + "="*60)
    print("ANALYSIS 4: Per-Persona Breakdown")
    print("="*60)

    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    scenario_metrics = {
        "flat":      ("MAS_Deviation",     False),
        "crash":     ("Max_Drawdown_Pct",  False),
        "bull_trap": ("Rationality_Score", True),
    }

    records = []
    for persona in df["Persona"].unique():
        persona_df = df[df["Persona"] == persona]
        cideal     = CIDEAL_MAP.get(persona, 0.5)

        for scenario, (metric, higher_is_better) in scenario_metrics.items():
            scen_df = persona_df[persona_df["Scenario"] == scenario]
            s_vals  = scen_df[scen_df["Agent_Type"] == "static"][metric].dropna().values
            m_vals  = scen_df[scen_df["Agent_Type"] == "memory"][metric].dropna().values

            if len(s_vals) == 0 or len(m_vals) == 0:
                continue

            mean_s = np.mean(s_vals)
            mean_m = np.mean(m_vals)

            if higher_is_better:
                gap = (mean_m - mean_s) / abs(mean_s) * 100 if mean_s != 0 else np.nan
            else:
                gap = (mean_s - mean_m) / abs(mean_s) * 100 if mean_s != 0 else np.nan

            records.append({
                "Persona":     persona,
                "Cideal":      cideal,
                "Scenario":    scenario,
                "Metric":      metric,
                "Static_Mean": round(mean_s, 4),
                "Memory_Mean": round(mean_m, 4),
                "Gap_Pct":     round(gap, 2),
            })

    result = pd.DataFrame(records)
    print(result.to_string(index=False))
    return result


def analysis_5_crash_sensitivity(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 5: Crash discount sensitivity analysis.
    Shows whether the static vs memory finding holds across
    crash_discount = 0.85, 0.92, 0.95.
    """
    print("\n" + "="*60)
    print("ANALYSIS 5: Crash Discount Sensitivity Analysis")
    print("="*60)

    crash_df = master[master["Scenario"] == "crash"].copy()

    if "Crash_Discount" not in crash_df.columns:
        print("[WARNING] Crash_Discount column not found. Skipping.")
        return pd.DataFrame()

    records = []
    for discount in sorted(crash_df["Crash_Discount"].unique()):
        d_df   = crash_df[crash_df["Crash_Discount"] == discount]
        s_vals = d_df[d_df["Agent_Type"] == "static"]["Max_Drawdown_Pct"].dropna().values
        m_vals = d_df[d_df["Agent_Type"] == "memory"]["Max_Drawdown_Pct"].dropna().values

        if len(s_vals) == 0 or len(m_vals) == 0:
            continue

        mean_s = np.mean(s_vals)
        mean_m = np.mean(m_vals)
        # For MDD, less negative = better. Memory should have less drawdown.
        gap    = (mean_s - mean_m) / abs(mean_s) * 100 if mean_s != 0 else np.nan

        # Wilcoxon
        merge_cols = ["Model", "Persona", "Seed"]
        paired = (
            d_df[d_df["Agent_Type"] == "static"][merge_cols + ["Max_Drawdown_Pct"]]
            .merge(
                d_df[d_df["Agent_Type"] == "memory"][merge_cols + ["Max_Drawdown_Pct"]],
                on=merge_cols, suffixes=("_s","_m")
            ).dropna()
        )
        wtest = wilcoxon_test(
            paired["Max_Drawdown_Pct_s"].values,
            paired["Max_Drawdown_Pct_m"].values,
            "Max_Drawdown_Pct",
            higher_is_better=False
        )

        records.append({
            "Crash_Discount":  discount,
            "Static_MDD_Mean": round(mean_s, 4),
            "Static_MDD_Std":  round(np.std(s_vals), 4),
            "Memory_MDD_Mean": round(mean_m, 4),
            "Memory_MDD_Std":  round(np.std(m_vals), 4),
            "Gap_Pct":         round(gap, 2),
            "Wilcoxon_p":      wtest["p_value"],
            "Significant":     wtest["significant"],
            "N_pairs":         wtest["n_pairs"],
        })

        print(f"\nCrash Discount: {discount}")
        print(f"  Static MDD:  {mean_s:.4f} ± {np.std(s_vals):.4f}")
        print(f"  Memory MDD:  {mean_m:.4f} ± {np.std(m_vals):.4f}")
        print(f"  Gap:         {gap:.2f}%")
        print(f"  Wilcoxon p:  {wtest['p_value']}")

    result = pd.DataFrame(records)
    return result


def analysis_6_temporal_decay(per_step: pd.DataFrame) -> pd.DataFrame:
    """
    Table 6: Temporal decay — how do MAS, CI, RG evolve over time?
    Buckets days into quartiles (0-25, 26-50, 51-75, 76-100)
    and computes mean metric value per bucket for static vs memory.
    This is the numerical backbone of the decay curve plots.
    """
    print("\n" + "="*60)
    print("ANALYSIS 6: Temporal Decay Statistics")
    print("="*60)

    # Use default crash discount only
    df = per_step[
        (per_step["Scenario"] != "crash") |
        (per_step.get("Crash_Discount", 0.92) == 0.92)
    ].copy()

    # Add temporal metrics per run
    run_cols = ["Model", "MBTI", "Agent_Type", "Scenario", "Seed"]
    if "Crash_Discount" in df.columns:
        run_cols.append("Crash_Discount")

    enriched_parts = []
    for keys, grp in df.groupby(run_cols):
        try:
            enriched = compute_temporal_metrics(grp)
            enriched_parts.append(enriched)
        except Exception as e:
            continue

    if not enriched_parts:
        print("[WARNING] Could not compute temporal metrics.")
        return pd.DataFrame()

    enriched_df = pd.concat(enriched_parts, ignore_index=True)

    # Create day quartile buckets
    enriched_df["Day_Bucket"] = pd.cut(
        enriched_df["Day"],
        bins=[0, 25, 50, 75, 100],
        labels=["Q1(1-25)", "Q2(26-50)", "Q3(51-75)", "Q4(76-100)"],
        include_lowest=True
    )

    records = []
    for scenario in ["flat", "crash", "bull_trap"]:
        scen_df = enriched_df[enriched_df["Scenario"] == scenario]
        if scen_df.empty:
            continue

        metric_map = {
            "flat":      "MAS_rolling",
            "crash":     "CI_rolling",
            "bull_trap": "RG_rolling",
        }
        metric = metric_map[scenario]

        for agent_type in ["static", "memory"]:
            a_df = scen_df[scen_df["Agent_Type"] == agent_type]
            if a_df.empty:
                continue

            for bucket, b_df in a_df.groupby("Day_Bucket", observed=True):
                vals = b_df[metric].dropna()
                if len(vals) == 0:
                    continue
                records.append({
                    "Scenario":    scenario,
                    "Agent_Type":  agent_type,
                    "Day_Bucket":  str(bucket),
                    "Metric":      metric,
                    "Mean":        round(vals.mean(), 4),
                    "Std":         round(vals.std(), 4),
                    "N":           len(vals),
                })

    result = pd.DataFrame(records)

    # Print side-by-side static vs memory per scenario
    for scenario in ["flat", "crash", "bull_trap"]:
        s_df = result[result["Scenario"] == scenario]
        if s_df.empty:
            continue
        print(f"\nScenario: {scenario}")
        pivot = s_df.pivot_table(
            index="Day_Bucket",
            columns="Agent_Type",
            values="Mean"
        )
        print(pivot.to_string())

    return result


def analysis_7_complete_wilcoxon(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 7: Full Wilcoxon test suite across all metric/scenario combinations.
    Provides statistical evidence for every claim in the paper.
    """
    print("\n" + "="*60)
    print("ANALYSIS 7: Complete Wilcoxon Significance Tests")
    print("="*60)

    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    tests = [
        ("flat",      "MAS_Deviation",     False, "MAS: static worse = higher deviation"),
        ("crash",     "Max_Drawdown_Pct",  False, "CI: static worse = larger drawdown"),
        ("bull_trap", "Rationality_Score", True,  "RG: static worse = lower rationality"),
        ("flat",      "Trade_Count",       False, "Boredom trading: static trades more"),
        ("crash",     "Return_Pct",        True,  "Financial performance in crash"),
        ("bull_trap", "Return_Pct",        True,  "Financial performance in bull trap"),
        ("flat",      "Avg_Cash_Pct",      None,  "Raw cash pct in flat (interpretability)"),
    ]

    records = []
    for scenario, metric, higher_is_better, description in tests:
        if metric not in df.columns:
            continue

        scen_df = df[df["Scenario"] == scenario]
        merge_cols = ["Model", "Persona", "Seed"]

        paired = (
            scen_df[scen_df["Agent_Type"] == "static"][merge_cols + [metric]]
            .merge(
                scen_df[scen_df["Agent_Type"] == "memory"][merge_cols + [metric]],
                on=merge_cols, suffixes=("_s","_m")
            ).dropna()
        )

        if paired.empty:
            continue

        s_vals = paired[f"{metric}_s"].values
        m_vals = paired[f"{metric}_m"].values

        hib = higher_is_better if higher_is_better is not None else True
        wtest = wilcoxon_test(s_vals, m_vals, metric, hib)
        wtest["scenario"]    = scenario
        wtest["description"] = description
        records.append(wtest)

        sig_str = "✓ SIGNIFICANT" if wtest["significant"] else "✗ not significant"
        print(f"\n{description}")
        print(f"  Static: {wtest['mean_static']:.4f} | Memory: {wtest['mean_memory']:.4f}")
        print(f"  p={wtest['p_value']} | {sig_str} | direction: {wtest['direction']}")

    return pd.DataFrame(records)


def analysis_8_failure_mode_summary(master: pd.DataFrame) -> pd.DataFrame:
    """
    Table 8: The three failure mode summary — the paper's Table 2 equivalent.
    Aggregated across all models and personas (default crash discount).
    This is the single most important table for the paper.
    """
    print("\n" + "="*60)
    print("ANALYSIS 8: Three Failure Mode Summary (Paper Table 2)")
    print("="*60)

    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    rows = []

    # Failure Mode 1: MAS in flat (Mandate Adherence / Boredom Trading)
    flat_df = df[df["Scenario"] == "flat"]
    s_mas = flat_df[flat_df["Agent_Type"] == "static"]["MAS_Deviation"].dropna()
    m_mas = flat_df[flat_df["Agent_Type"] == "memory"]["MAS_Deviation"].dropna()
    s_cash = flat_df[flat_df["Agent_Type"] == "static"]["Avg_Cash_Pct"].dropna()
    m_cash = flat_df[flat_df["Agent_Type"] == "memory"]["Avg_Cash_Pct"].dropna()

    gap_mas = (s_mas.mean() - m_mas.mean()) / abs(s_mas.mean()) * 100
    rows.append({
        "Failure_Mode":        "Mandate Adherence (Boredom Trading)",
        "Scenario":            "flat",
        "Metric":              "MAS_Deviation",
        "Static_Mean":         round(s_mas.mean(), 4),
        "Static_Std":          round(s_mas.std(), 4),
        "Memory_Mean":         round(m_mas.mean(), 4),
        "Memory_Std":          round(m_mas.std(), 4),
        "Behavioral_Gap_Pct":  round(gap_mas, 2),
        "Static_Cash_Pct":     round(s_cash.mean(), 1),
        "Memory_Cash_Pct":     round(m_cash.mean(), 1),
    })

    # Failure Mode 2: CI in crash (Caricature / Panic)
    crash_df = df[df["Scenario"] == "crash"]
    s_mdd = crash_df[crash_df["Agent_Type"] == "static"]["Max_Drawdown_Pct"].dropna()
    m_mdd = crash_df[crash_df["Agent_Type"] == "memory"]["Max_Drawdown_Pct"].dropna()

    gap_mdd = (s_mdd.mean() - m_mdd.mean()) / abs(s_mdd.mean()) * 100
    rows.append({
        "Failure_Mode":        "Caricature Index (Panic Selling)",
        "Scenario":            "crash",
        "Metric":              "Max_Drawdown_Pct",
        "Static_Mean":         round(s_mdd.mean(), 4),
        "Static_Std":          round(s_mdd.std(), 4),
        "Memory_Mean":         round(m_mdd.mean(), 4),
        "Memory_Std":          round(m_mdd.std(), 4),
        "Behavioral_Gap_Pct":  round(gap_mdd, 2),
        "Static_Cash_Pct":     np.nan,
        "Memory_Cash_Pct":     np.nan,
    })

    # Failure Mode 3: RG in bull_trap (Rationality Gap / FOMO)
    bull_df = df[df["Scenario"] == "bull_trap"]
    s_rg = bull_df[bull_df["Agent_Type"] == "static"]["Rationality_Score"].dropna()
    m_rg = bull_df[bull_df["Agent_Type"] == "memory"]["Rationality_Score"].dropna()

    gap_rg = (m_rg.mean() - s_rg.mean()) / abs(s_rg.mean()) * 100
    rows.append({
        "Failure_Mode":        "Rationality Gap (FOMO / Value Decoupling)",
        "Scenario":            "bull_trap",
        "Metric":              "Rationality_Score",
        "Static_Mean":         round(s_rg.mean(), 4),
        "Static_Std":          round(s_rg.std(), 4),
        "Memory_Mean":         round(m_rg.mean(), 4),
        "Memory_Std":          round(m_rg.std(), 4),
        "Behavioral_Gap_Pct":  round(gap_rg, 2),
        "Static_Cash_Pct":     np.nan,
        "Memory_Cash_Pct":     np.nan,
    })

    result = pd.DataFrame(rows)

    print("\nThree Failure Modes (aggregated across all models and personas):")
    print(f"{'Failure Mode':<40} {'Static':>10} {'Memory':>10} {'Gap%':>8}")
    print("-" * 70)
    for _, row in result.iterrows():
        print(
            f"{row['Failure_Mode']:<40} "
            f"{row['Static_Mean']:>10.4f} "
            f"{row['Memory_Mean']:>10.4f} "
            f"{row['Behavioral_Gap_Pct']:>8.2f}%"
        )

    return result


def analysis_per_persona_direction(master: pd.DataFrame) -> pd.DataFrame:
    """
    For each (Persona, Scenario), checks whether memory consistently
    improves over static across all models and seeds.
    Reports direction consistency — the key question is not just
    mean gap but whether the direction holds universally.
    """
    df = master[
        (master["Scenario"] != "crash") |
        (master["Crash_Discount"] == 0.92)
    ].copy()

    scenario_metrics = {
        "flat":      ("MAS_Deviation",     False),
        "crash":     ("Max_Drawdown_Pct",  False),
        "bull_trap": ("Rationality_Score", True),
    }

    records = []
    for persona in ["ENTJ", "ISFJ", "INTJ"]:
        for scenario, (metric, higher_is_better) in scenario_metrics.items():
            sub = df[
                (df["Persona"] == persona) &
                (df["Scenario"] == scenario)
            ]

            for model in sub["Model"].unique():
                model_sub = sub[sub["Model"] == model]
                s_vals = model_sub[model_sub["Agent_Type"] == "static"][metric].dropna().values
                m_vals = model_sub[model_sub["Agent_Type"] == "memory"][metric].dropna().values

                if len(s_vals) == 0 or len(m_vals) == 0:
                    continue

                mean_s = np.mean(s_vals)
                mean_m = np.mean(m_vals)

                if higher_is_better:
                    memory_wins = mean_m > mean_s
                    gap = (mean_m - mean_s) / abs(mean_s) * 100 if mean_s != 0 else np.nan
                else:
                    memory_wins = mean_m < mean_s
                    gap = (mean_s - mean_m) / abs(mean_s) * 100 if mean_s != 0 else np.nan

                records.append({
                    "Persona":     persona,
                    "Model":       model,
                    "Scenario":    scenario,
                    "Metric":      metric,
                    "Static_Mean": round(mean_s, 4),
                    "Memory_Mean": round(mean_m, 4),
                    "Gap_Pct":     round(gap, 2),
                    "Memory_Wins": memory_wins,
                })

    result = pd.DataFrame(records)

    # Summarise consistency
    print("\nDirection Consistency (does memory win for this persona/scenario?):")
    summary = result.groupby(["Persona", "Scenario"])["Memory_Wins"].agg(
        ["sum", "count"]
    )
    summary["Consistency"] = summary["sum"].astype(str) + "/" + summary["count"].astype(str) + " models"
    print(summary[["Consistency"]].to_string())

    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*60)
    print("FinPersona-Bench Numerical Analysis")
    print("="*60)

    # ── Load data ──────────────────────────────
    print("\n[1/2] Loading master summary...")
    master = load_master_summary()

    print("\n[2/2] Loading per-step CSVs (this may take a moment)...")
    try:
        per_step = load_per_step_csvs()
        has_per_step = True
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Skipping temporal analysis (Analysis 6).")
        has_per_step = False

    # ── Run analyses ───────────────────────────
    results = {}

    results["master_table"]       = analysis_1_master_table(master)
    results["static_vs_memory"],  \
    results["wilcoxon_core"]      = analysis_2_static_vs_memory(master)
    results["per_model"]          = analysis_3_per_model(master)
    results["per_persona"]        = analysis_4_per_persona(master)
    results["crash_sensitivity"]  = analysis_5_crash_sensitivity(master)

    if has_per_step:
        results["temporal_decay"] = analysis_6_temporal_decay(per_step)
    else:
        results["temporal_decay"] = pd.DataFrame()

    results["wilcoxon_full"]         = analysis_7_complete_wilcoxon(master)
    results["failure_mode_summary"]  = analysis_8_failure_mode_summary(master)
    results["per_persona_direction"] = analysis_per_persona_direction(master)

    # ── Save all outputs ───────────────────────
    print("\n" + "="*60)
    print("Saving outputs to", OUTPUT_DIR)
    print("="*60)

    file_map = {
        "master_table.csv":          "master_table",
        "static_vs_memory.csv":      "static_vs_memory",
        "wilcoxon_core.csv":         "wilcoxon_core",
        "per_model.csv":             "per_model",
        "per_persona.csv":           "per_persona",
        "crash_sensitivity.csv":     "crash_sensitivity",
        "temporal_decay.csv":        "temporal_decay",
        "wilcoxon_full.csv":         "wilcoxon_full",
        "failure_mode_summary.csv":   "failure_mode_summary",
        "per_persona_direction.csv":  "per_persona_direction",
    }

    for filename, key in file_map.items():
        df = results.get(key, pd.DataFrame())
        if df is not None and not df.empty:
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  ✓ {filename} ({len(df)} rows)")
        else:
            print(f"  ✗ {filename} — empty, skipped")

    print("\nAnalysis complete.")
    print(f"All outputs in: {OUTPUT_DIR}/")

    # ── Quick summary for terminal ─────────────
    print("\n" + "="*60)
    print("QUICK SUMMARY FOR PAPER")
    print("="*60)
    fms = results.get("failure_mode_summary")
    if fms is not None and not fms.empty:
        for _, row in fms.iterrows():
            direction = "improvement" if row["Behavioral_Gap_Pct"] > 0 else "regression"
            print(
                f"  {row['Failure_Mode'][:35]:<35}: "
                f"{abs(row['Behavioral_Gap_Pct']):.1f}% {direction} "
                f"(static={row['Static_Mean']:.3f}, "
                f"memory={row['Memory_Mean']:.3f})"
            )

    wf = results.get("wilcoxon_full")
    if wf is not None and not wf.empty:
        sig = wf[wf["significant"] == True]
        print(f"\n  Significant Wilcoxon tests: {len(sig)}/{len(wf)}")
        for _, row in sig.iterrows():
            print(f"    ✓ {row.get('description','')} (p={row['p_value']})")


if __name__ == "__main__":
    main()
