"""
FinPersona-Bench — T Threshold Calibration Experiment
======================================================
Purpose:
    Empirically determine the minimum simulation horizon T at which MSD
    becomes statistically detectable (Wilcoxon p < 0.05) across frontier
    LLMs and all three market scenarios.

    Output: a p-value vs T curve per model and scenario, which justifies
    the choice of T=200 in the main experiment as an empirically grounded
    threshold rather than an arbitrary value.

Design:
    Models:     3 frontier LLMs (configured in MODELS below)
    Personas:   ENTJ, ISFJ, INTJ
    Scenarios:  flat, crash, bull_trap  (all three — threshold may differ)
    T values:   50, 100, 150, 200, 250, 300
    Seeds:      3  (minimum for Wilcoxon; keeps runtime ~2-3 hours)
    Architectures: static and memory

Total runs:
    3 models x 3 personas x 3 scenarios x 2 arch x 3 seeds x 6 T values
    = 972 runs

    For crash scenario only, crash_discount = 0.92 (fixed — this
    experiment is about T sensitivity, not discount sensitivity).

Outputs (in analysis/t_calibration_outputs/):
    t_calibration_master.csv   — one row per run
    t_calibration_wilcoxon.csv — p-value per (model, scenario, T)
    t_calibration_plot.png     — p-value vs T curves (generated at end)

Usage:
    python run_t_calibration.py

    Supports resumption via checkpoint file.
    Safe to run on two laptops simultaneously with different MODELS lists.
"""

import os
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from tqdm import tqdm
from simulation.runner import run_simulation

# ── Configuration ──────────────────────────────────────────────────────────

MODELS = [
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "gpt-4o-mini",
]

PERSONAS    = ["ENTJ", "ISFJ", "INTJ"]
SCENARIOS   = ["flat", "bull_trap", "crash"]
AGENT_TYPES = ["static", "memory"]
SEEDS       = [42, 123, 456]           # 3 seeds — minimum viable for Wilcoxon
T_VALUES    = [50, 100, 150, 200, 250, 300]
CRASH_DISCOUNT = 0.92                  # fixed for this experiment

MAX_WORKERS    = 20
OUTPUT_DIR     = "analysis/t_calibration_outputs"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.txt")

CIDEAL_MAP = {
    "ISFJ": 1.0,
    "INTJ": 0.5,
    "ENTJ": 0.2,
}

# Scenario → primary metric and direction
SCENARIO_METRIC = {
    "flat":      ("MAS_Deviation",    False),  # lower = better
    "crash":     ("Max_Drawdown_Pct", False),  # less negative = better
    "bull_trap": ("Rationality_Score", True),  # higher = better
}

# ── Thread safety ──────────────────────────────────────────────────────────

_checkpoint_lock = threading.Lock()


# ── Metrics ────────────────────────────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame, initial_cash: float) -> dict:
    if df is None or df.empty:
        return {}

    final_value      = df.iloc[-1]["Portfolio_Value"]
    total_return_pct = ((final_value - initial_cash) / initial_cash) * 100

    rolling_max      = df["Portfolio_Value"].cummax()
    daily_drawdown   = df["Portfolio_Value"] / rolling_max - 1.0
    max_drawdown_pct = daily_drawdown.min() * 100

    trades      = df[df["Action"].isin(["BUY", "SELL"])]
    trade_count = len(trades)

    if "Fundamental_Value" not in df.columns:
        rationality_score = float("nan")
    else:
        def compute_yt(row):
            action   = row["Action"]
            pt       = row["Price"]
            vt       = row["Fundamental_Value"]
            holdings = max(0.0, row["Portfolio_Value"] - row["Cash"])
            if pd.isna(vt) or vt <= 0:
                return float("nan")
            if action == "BUY":
                return 1 if pt < vt else 0
            elif action == "SELL":
                return 1 if pt > vt else 0
            elif action == "HOLD":
                return 1 if (pt > vt) or (holdings > 1.0) else 0
            return float("nan")

        yt_values = df.apply(compute_yt, axis=1)
        valid     = yt_values.dropna()
        rationality_score = round(float(valid.mean() * 100), 1) if len(valid) > 0 else float("nan")

    persona = df["MBTI"].iloc[0] if "MBTI" in df.columns else "UNKNOWN"
    cideal  = CIDEAL_MAP.get(persona, 0.5)
    pv      = df["Portfolio_Value"].replace(0, float("nan"))
    cf      = df["Cash"] / pv
    mas_dev = round(float((cf - cideal).abs().mean()), 4)
    avg_cash = round(float(cf.mean() * 100), 1)

    return {
        "Final_Value":       round(final_value, 2),
        "Return_Pct":        round(total_return_pct, 2),
        "Max_Drawdown_Pct":  round(max_drawdown_pct, 2),
        "Trade_Count":       trade_count,
        "Rationality_Score": rationality_score,
        "MAS_Deviation":     mas_dev,
        "Avg_Cash_Pct":      avg_cash,
        "Cideal":            cideal,
    }


# ── Checkpointing ──────────────────────────────────────────────────────────

def make_run_id(model, persona, agent_type, scenario, seed, t_value):
    return f"{model}__{persona}__{agent_type}__{scenario}__seed{seed}__T{t_value}"


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_checkpoint(run_id: str):
    with _checkpoint_lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(run_id + "\n")


# ── Single run worker ──────────────────────────────────────────────────────

def run_single(config: tuple) -> dict:
    model, persona, agent_type, scenario, seed, t_value = config
    run_id = make_run_id(model, persona, agent_type, scenario, seed, t_value)

    output_dir = os.path.join(
        "results", "t_calibration",
        model, scenario, f"T{t_value}", f"seed{seed}"
    )

    try:
        df = run_simulation(
            mbti_type=persona,
            agent_type=agent_type,
            scenario=scenario,
            model_name=model,
            output_dir=output_dir,
            initial_cash=10000.0,
            max_days=t_value,
            seed=seed,
            crash_discount=CRASH_DISCOUNT,
        )

        if df is None:
            raise ValueError("run_simulation returned None")

        metrics = calculate_metrics(df, 10000.0)
        if not metrics:
            raise ValueError("Empty metrics")

        metrics.update({
            "Model":      model,
            "Persona":    persona,
            "Agent_Type": agent_type,
            "Scenario":   scenario,
            "Seed":       seed,
            "T_Value":    t_value,
            "Status":     "PASS",
        })
        save_checkpoint(run_id)
        return metrics

    except Exception as e:
        return {
            "Model":      model,
            "Persona":    persona,
            "Agent_Type": agent_type,
            "Scenario":   scenario,
            "Seed":       seed,
            "T_Value":    t_value,
            "Status":     f"FAIL: {e}",
        }


# ── Wilcoxon analysis ──────────────────────────────────────────────────────

def compute_wilcoxon_curves(master: pd.DataFrame) -> pd.DataFrame:
    """
    For each (model, scenario, T_value), compute the paired Wilcoxon
    p-value for static vs memory on the primary metric.
    Returns a DataFrame with one row per (model, scenario, T_value).
    """
    records = []
    df = master[master["Status"] == "PASS"].copy()

    for model in df["Model"].unique():
        for scenario in SCENARIOS:
            metric, higher_is_better = SCENARIO_METRIC[scenario]
            if metric not in df.columns:
                continue

            for t_val in sorted(df["T_Value"].unique()):
                sub = df[
                    (df["Model"]    == model) &
                    (df["Scenario"] == scenario) &
                    (df["T_Value"]  == t_val)
                ]

                static_df = sub[sub["Agent_Type"] == "static"]
                memory_df = sub[sub["Agent_Type"] == "memory"]

                # Pair by (Persona, Seed)
                paired = static_df[["Persona", "Seed", metric]].merge(
                    memory_df[["Persona", "Seed", metric]],
                    on=["Persona", "Seed"],
                    suffixes=("_s", "_m")
                ).dropna()

                n_pairs = len(paired)
                if n_pairs < 3:
                    p_value   = float("nan")
                    statistic = float("nan")
                    sig       = False
                else:
                    try:
                        stat, p = stats.wilcoxon(
                            paired[f"{metric}_s"].values,
                            paired[f"{metric}_m"].values,
                            alternative="two-sided"
                        )
                        statistic = round(float(stat), 4)
                        p_value   = round(float(p), 4)
                        sig       = bool(p < 0.05)
                    except Exception:
                        statistic = float("nan")
                        p_value   = float("nan")
                        sig       = False

                mean_s = paired[f"{metric}_s"].mean() if n_pairs > 0 else float("nan")
                mean_m = paired[f"{metric}_m"].mean() if n_pairs > 0 else float("nan")

                if not np.isnan(mean_s) and mean_s != 0:
                    if higher_is_better:
                        gap_pct = (mean_m - mean_s) / abs(mean_s) * 100
                    else:
                        gap_pct = (mean_s - mean_m) / abs(mean_s) * 100
                else:
                    gap_pct = float("nan")

                records.append({
                    "Model":          model,
                    "Scenario":       scenario,
                    "Metric":         metric,
                    "T_Value":        t_val,
                    "N_pairs":        n_pairs,
                    "Mean_Static":    round(mean_s, 4) if not np.isnan(mean_s) else float("nan"),
                    "Mean_Memory":    round(mean_m, 4) if not np.isnan(mean_m) else float("nan"),
                    "Gap_Pct":        round(gap_pct, 2) if not np.isnan(gap_pct) else float("nan"),
                    "Wilcoxon_Stat":  statistic,
                    "P_Value":        p_value,
                    "Significant":    sig,
                })

    return pd.DataFrame(records)


# ── Plot ───────────────────────────────────────────────────────────────────

def plot_p_curves(wilcoxon_df: pd.DataFrame, output_path: str):
    """
    Three-panel plot (one per scenario) showing p-value vs T for each model.
    Horizontal dashed line at p=0.05.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        C = {
            "navy":    "#0D1B2A",
            "teal":    "#1B998B",
            "coral":   "#E84855",
            "gold":    "#F4A261",
            "muted":   "#94A3B8",
            "label":   "#4A5568",
            "bg":      "#FFFFFF",
            "offwhite":"#F7FAFB",
            "grid":    "#E2E8F0",
        }

        model_colors = {}
        palette = [C["coral"], C["teal"], C["gold"],
                   "#7B61FF", "#2EC4B6", "#E71D36"]
        for i, m in enumerate(wilcoxon_df["Model"].unique()):
            model_colors[m] = palette[i % len(palette)]

        scenario_titles = {
            "flat":      "Stability (MAS) — Flat",
            "crash":     "Safety (CI) — Crash",
            "bull_trap": "Rationality (RG) — Bull Trap",
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        fig.patch.set_facecolor(C["offwhite"])

        for ax, scenario in zip(axes, ["flat", "crash", "bull_trap"]):
            ax.set_facecolor(C["bg"])

            # Significance threshold
            ax.axhline(0.05, color=C["coral"], linewidth=1.5,
                       linestyle="--", zorder=3, label="p = 0.05 threshold")

            scen_df = wilcoxon_df[wilcoxon_df["Scenario"] == scenario]

            for model in scen_df["Model"].unique():
                m_df = scen_df[scen_df["Model"] == model].sort_values("T_Value")
                valid = m_df.dropna(subset=["P_Value"])
                if valid.empty:
                    continue

                short_name = model.split("-")[0].capitalize()
                ax.plot(
                    valid["T_Value"], valid["P_Value"],
                    color=model_colors[model],
                    linewidth=2.5,
                    marker="o", markersize=8,
                    markerfacecolor=model_colors[model],
                    markeredgecolor=C["bg"],
                    markeredgewidth=1.5,
                    label=short_name,
                    zorder=4
                )

                # Mark significant points
                sig = valid[valid["Significant"]]
                if not sig.empty:
                    ax.scatter(
                        sig["T_Value"], sig["P_Value"],
                        color=model_colors[model],
                        s=120, zorder=5,
                        marker="*"
                    )

            ax.set_xlabel("Simulation Horizon (T)", fontsize=10.5,
                          color=C["label"], labelpad=8)
            ax.set_ylabel("Wilcoxon p-value", fontsize=10.5,
                          color=C["label"], labelpad=8)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(T_VALUES)
            ax.set_xticklabels([str(t) for t in T_VALUES],
                               fontsize=9, color=C["label"])
            ax.yaxis.grid(True, color=C["grid"], linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.spines["left"].set_color(C["grid"])
            ax.spines["bottom"].set_color(C["grid"])
            ax.tick_params(colors=C["muted"])
            ax.yaxis.set_tick_params(labelcolor=C["label"])
            ax.set_title(scenario_titles[scenario],
                         fontsize=11, fontweight="bold",
                         color=C["navy"], pad=10)
            ax.legend(fontsize=9, frameon=True,
                      facecolor=C["bg"], edgecolor=C["grid"],
                      loc="upper right")

        fig.suptitle(
            "MSD Detection Threshold — p-value vs Simulation Horizon T\n"
            "★ = statistically significant (p < 0.05)  ·  "
            "Dashed line = significance threshold",
            fontsize=12, fontweight="bold",
            color=C["navy"], y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ Plot saved: {output_path}")

    except Exception as e:
        print(f"  [WARNING] Could not generate plot: {e}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results/t_calibration", exist_ok=True)

    print("=" * 65)
    print("  FinPersona-Bench — T Threshold Calibration Experiment")
    print("=" * 65)

    # Build all configs
    all_configs = [
        (model, persona, agent_type, scenario, seed, t_value)
        for t_value    in T_VALUES
        for scenario   in SCENARIOS
        for seed       in SEEDS
        for model      in MODELS
        for persona    in PERSONAS
        for agent_type in AGENT_TYPES
    ]

    # Filter completed
    completed = load_checkpoint()
    pending = [c for c in all_configs if make_run_id(*c) not in completed]

    total_runs      = len(all_configs)
    completed_runs  = total_runs - len(pending)

    print(f"\nProgress Overview:")
    print(f"Total runs:     {total_runs:,}")
    print(f"Completed:      {completed_runs:,} ({completed_runs / total_runs * 100:.1f}%)")
    print(f"Remaining:      {len(pending):,}")
    print(f"Output dir:     {OUTPUT_DIR}/\n")

    # 1. Use a static filename instead of a timestamped one so you can append to it
    master_log = os.path.join(OUTPUT_DIR, "t_calibration_master.csv")
    
    # 2. Load past results into memory if resuming, so Wilcoxon curves have all data
    all_results = []
    if os.path.exists(master_log):
        all_results = pd.read_csv(master_log).to_dict('records')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, cfg): cfg for cfg in pending}

        with tqdm(as_completed(futures), total=len(pending), unit="run",
                  desc="Calibration", dynamic_ncols=True) as pbar:
            for future in pbar:
                result = future.result()
                all_results.append(result)

                this_t = result.get('T_Value', 0)
                status = result.get("Status", "?")
                model  = result.get("Model", "?")
                pbar.set_postfix_str(f"{model} | T={this_t} | {status}", refresh=True)

                if not status.startswith("PASS"):
                    pbar.write(f"[FAIL] {model} T={this_t} seed={result.get('Seed')} — {status}")

                # 3. Efficiently APPEND just the new result to the CSV
                df_single = pd.DataFrame([result])
                write_header = not os.path.exists(master_log)
                df_single.to_csv(master_log, mode='a', header=write_header, index=False)

    print(f"\n{'='*65}")
    print(f"All runs complete. Master log: {master_log}")

    # ── Wilcoxon curves ────────────────────────────────────────────────
    print("\nComputing Wilcoxon p-value curves...")
    master_df   = pd.DataFrame(all_results)
    wilcoxon_df = compute_wilcoxon_curves(master_df)

    wilcoxon_path = os.path.join(OUTPUT_DIR, "t_calibration_wilcoxon.csv")
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"  ✓ Wilcoxon table: {wilcoxon_path}")

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY: First T at which p < 0.05 per model and scenario")
    print(f"{'='*65}")
    print(f"{'Model':25s} {'Scenario':12s} {'Threshold T':>12s} {'p at T=200':>12s}")
    print("-" * 65)

    for model in wilcoxon_df["Model"].unique():
        for scenario in SCENARIOS:
            sub = wilcoxon_df[
                (wilcoxon_df["Model"]    == model) &
                (wilcoxon_df["Scenario"] == scenario)
            ].sort_values("T_Value")

            sig_rows = sub[sub["Significant"]]
            threshold = sig_rows["T_Value"].min() if not sig_rows.empty else "never"

            row_200 = sub[sub["T_Value"] == 200]
            p_200   = row_200["P_Value"].values[0] if not row_200.empty else float("nan")
            p_str   = f"{p_200:.3f}" if not np.isnan(p_200) else "n/a"

            print(f"{model:25s} {scenario:12s} {str(threshold):>12s} {p_str:>12s}")

    # ── Plot ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Generating p-value vs T plots...")
    plot_path = os.path.join(OUTPUT_DIR, "t_calibration_plot.png")
    plot_p_curves(wilcoxon_df, plot_path)

    print(f"\n{'='*65}")
    print(f"Calibration experiment complete.")
    print(f"All outputs in: {OUTPUT_DIR}/")
    print(f"  - t_calibration_master.csv    (raw run results)")
    print(f"  - t_calibration_wilcoxon.csv  (p-values per model/scenario/T)")
    print(f"  - t_calibration_plot.png      (the paper figure)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
    