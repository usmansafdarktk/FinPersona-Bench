"""
FinPersona-OCEAN Experiment Orchestrator
==========================================
Runs the Big Five / OCEAN framework-independence robustness experiment.
This is the rebuttal addition for Section 4.5 of the paper.

Grid (spec Section 3.1):
  Core (O1 + O2):     2 personas × 3 scenarios × 5 seeds × 2 agent_types × 3 models = 180 sims
  Ablation (O3 flat): 2 personas × 1 scenario  × 5 seeds × 2 agent_types × 3 models =  60 sims
  ─────────────────────────────────────────────────────────────────────────────────────────────
  Grand total: 240 simulations

Parameters held constant (spec Section 3.2):
  T = 200 trading days
  δ = 0.92 crash discount (default only — no sensitivity sweep for this robustness check)
  Temperature = 0.2, no top_p (identical to main FinPersona-Bench panel)
  W₀ = $10,000 initial capital
  Same market seeds as the FinPersona-Bench main panel

Outputs:
  results_ocean/<model>/<scenario>/seed<N>/<run_id>.csv  — per-step tick logs
  results_ocean/master_summary_<timestamp>.csv            — one row per run (all metrics)
  results_ocean/checkpoint.txt                            — completed run IDs for resume
"""

import os
import time
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from simulation.ocean_runner import run_ocean_simulation
from agent.ocean_prompts import OCEAN_CIDEAL_MAP

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODELS = [
    "claude-sonnet-4-6",
    "gpt-4o-mini",
    "gemini-2.5-flash",
]

# Core personas — all three scenarios
CORE_PERSONAS   = ["O1_conservative", "O2_aggressive"]
CORE_SCENARIOS  = ["flat", "bull_trap", "crash"]

# Ablation personas — flat scenario only (spec Section 2.3)
ABLATION_PERSONAS  = ["O3_conservative", "O3_aggressive"]
ABLATION_SCENARIOS = ["flat"]

AGENT_TYPES = ["static", "memory"]
SEEDS       = [42, 123, 456, 789, 999]

# Single crash discount value — robustness check, not sensitivity sweep
CRASH_DISCOUNT = 0.92

T           = 200  # Trading days (spec Section 3.2)
MAX_WORKERS = 15   # Parallel API threads

# Retry policy for transient network / server errors at the run level
RUN_MAX_RETRIES   = 4       # attempts before marking a run as permanently failed
RUN_BACKOFF_BASE  = 5       # seconds — doubles each retry (5, 10, 20, 40)
RETRYABLE_ERRORS  = (        # exception substrings that warrant a retry
    "connection", "timeout", "rate_limit", "rate limit",
    "503", "502", "529", "overloaded",
)

OUTPUT_ROOT     = "results_ocean"
CHECKPOINT_FILE = f"{OUTPUT_ROOT}/checkpoint.txt"
_checkpoint_lock = threading.Lock()


# ---------------------------------------------------------------------------
# METRICS CALCULATOR
# Identical logic to run_experiments.py for clean cross-framework comparison
# ---------------------------------------------------------------------------

def calculate_metrics(df: pd.DataFrame, initial_cash: float) -> dict:
    if df is None or df.empty:
        return {}

    # Financial performance
    final_value = df.iloc[-1]["Portfolio_Value"]
    total_return_pct = ((final_value - initial_cash) / initial_cash) * 100

    # Max Drawdown (Caricature Index proxy)
    rolling_max = df["Portfolio_Value"].cummax()
    daily_drawdown = df["Portfolio_Value"] / rolling_max - 1.0
    max_drawdown_pct = daily_drawdown.min() * 100

    # Trade activity
    trades = df[df["Action"].isin(["BUY", "SELL"])]
    trade_count = len(trades)

    # Rationality Score (Value Decoupling metric)
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
                if pt > vt:
                    return 1
                else:
                    return 1 if holdings > 1.0 else 0
            return float("nan")

        yt_values = df.apply(compute_yt, axis=1)
        valid = yt_values.dropna()
        rationality_score = (
            round(float(valid.mean() * 100), 1) if len(valid) > 0 else float("nan")
        )

    # Bubble participation (avg P/E at buy decisions)
    buy_decisions = df[df["Action"] == "BUY"]
    if not buy_decisions.empty and "Reported_PE" in buy_decisions.columns:
        pe_vals = pd.to_numeric(buy_decisions["Reported_PE"], errors="coerce").fillna(0)
        avg_buy_pe = pe_vals.mean()
    else:
        avg_buy_pe = 0.0

    # Mandate Adherence Score (MAS) — Boredom Trading metric
    persona = df["Persona"].iloc[0] if "Persona" in df.columns else "UNKNOWN"
    cideal = OCEAN_CIDEAL_MAP.get(persona, 0.5)

    portfolio_values = df["Portfolio_Value"].replace(0, float("nan"))
    cash_fraction    = df["Cash"] / portfolio_values
    mas_deviation    = round(float((cash_fraction - cideal).abs().mean()), 4)
    avg_cash_pct     = round(float(cash_fraction.mean() * 100), 1)

    return {
        "Final_Value":       round(final_value, 2),
        "Return_Pct":        round(total_return_pct, 2),
        "Max_Drawdown_Pct":  round(max_drawdown_pct, 2),
        "Trade_Count":       trade_count,
        "Rationality_Score": rationality_score,
        "Avg_Buy_PE":        round(avg_buy_pe, 1),
        "MAS_Deviation":     mas_deviation,
        "Avg_Cash_Pct":      avg_cash_pct,
        "Cideal":            cideal,
    }


# ---------------------------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------------------------

def make_run_id(
    model: str, persona: str, agent_type: str, scenario: str,
    seed: int, crash_discount: float,
) -> str:
    if scenario == "crash":
        return (
            f"{model}__{persona}__{agent_type}__{scenario}"
            f"__seed{seed}__discount{crash_discount}"
        )
    return f"{model}__{persona}__{agent_type}__{scenario}__seed{seed}"


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f)


def save_checkpoint(run_id: str):
    with _checkpoint_lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(run_id + "\n")


# ---------------------------------------------------------------------------
# SINGLE RUN WORKER  (with exponential-backoff retry for transient errors)
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in RETRYABLE_ERRORS)


def run_single(config: tuple) -> dict:
    model, persona, agent_type, scenario, seed, crash_discount = config
    run_id = make_run_id(model, persona, agent_type, scenario, seed, crash_discount)

    if scenario == "crash":
        output_dir = os.path.join(
            OUTPUT_ROOT, model, scenario, f"discount{crash_discount}", f"seed{seed}"
        )
    else:
        output_dir = os.path.join(OUTPUT_ROOT, model, scenario, f"seed{seed}")

    last_error = None
    for attempt in range(1, RUN_MAX_RETRIES + 1):
        try:
            df = run_ocean_simulation(
                persona_type=persona,
                agent_type=agent_type,
                scenario=scenario,
                model_name=model,
                output_dir=output_dir,
                initial_cash=10_000.0,
                max_days=T,
                seed=seed,
                crash_discount=crash_discount,
            )

            if df is None:
                print(f"[FAILED] {run_id}: run_ocean_simulation returned None")
                return {
                    "Model": model, "Persona": persona,
                    "Agent_Type": agent_type, "Scenario": scenario,
                    "Seed": seed, "Crash_Discount": crash_discount,
                    "Status": "FAIL: silent_failure",
                }

            metrics = calculate_metrics(df, 10_000.0)
            metrics.update({
                "Model": model, "Persona": persona,
                "Agent_Type": agent_type, "Scenario": scenario,
                "Seed": seed, "Crash_Discount": crash_discount,
                "Status": "PASS",
            })
            save_checkpoint(run_id)
            return metrics

        except Exception as e:
            last_error = e
            if attempt < RUN_MAX_RETRIES and _is_retryable(e):
                wait = RUN_BACKOFF_BASE * (2 ** (attempt - 1))  # 5, 10, 20, 40 s
                print(
                    f"[RETRY {attempt}/{RUN_MAX_RETRIES}] {run_id} — "
                    f"{type(e).__name__}: {e}. Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                break

    print(f"[FAILED] {run_id}: {last_error}")
    return {
        "Model": model, "Persona": persona,
        "Agent_Type": agent_type, "Scenario": scenario,
        "Seed": seed, "Crash_Discount": crash_discount,
        "Status": f"FAIL: {last_error}",
    }


# ---------------------------------------------------------------------------
# CONFIG BUILDER
# ---------------------------------------------------------------------------

def build_configs() -> list:
    configs = []

    # Core grid: O1 + O2, all three scenarios
    for model in MODELS:
        for persona in CORE_PERSONAS:
            for agent_type in AGENT_TYPES:
                for scenario in CORE_SCENARIOS:
                    for seed in SEEDS:
                        configs.append(
                            (model, persona, agent_type, scenario, seed, CRASH_DISCOUNT)
                        )

    # Ablation grid: O3 conservative + aggressive, flat only
    for model in MODELS:
        for persona in ABLATION_PERSONAS:
            for agent_type in AGENT_TYPES:
                for scenario in ABLATION_SCENARIOS:
                    for seed in SEEDS:
                        configs.append(
                            (model, persona, agent_type, scenario, seed, CRASH_DISCOUNT)
                        )

    return configs


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  FINPERSONA-OCEAN  —  Big Five / OCEAN Robustness Experiment")
    print("=" * 62)
    print(f"  Models:    {MODELS}")
    print(f"  Core:      {CORE_PERSONAS}")
    print(f"             × scenarios {CORE_SCENARIOS}")
    print(f"  Ablation:  {ABLATION_PERSONAS}")
    print(f"             × scenarios {ABLATION_SCENARIOS}  (flat only)")
    print(f"  Seeds:     {SEEDS}   |   T = {T} days   |   Workers = {MAX_WORKERS}")
    print("=" * 62)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_configs = build_configs()
    completed   = load_checkpoint()
    pending     = [c for c in all_configs if make_run_id(*c) not in completed]

    print(f"\nTotal configs:    {len(all_configs)}")
    print(f"Already complete: {len(completed)}")
    print(f"Remaining:        {len(pending)}\n")

    if not pending:
        print("All runs already complete. Nothing to do.")
        return

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = f"{OUTPUT_ROOT}/master_summary_{timestamp}.csv"
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, cfg): cfg for cfg in pending}

        pbar = tqdm(
            as_completed(futures),
            total=len(pending),
            desc="OCEAN runs",
            unit="sim",
            dynamic_ncols=True,
        )
        for future in pbar:
            result = future.result()
            all_results.append(result)

            scenario      = result.get("Scenario", "")
            status        = result.get("Status", "")
            discount_info = f" δ={result.get('Crash_Discount')}" if scenario == "crash" else ""
            passed        = sum(1 for r in all_results if r.get("Status") == "PASS")
            failed        = len(all_results) - passed

            pbar.set_postfix(
                model=result.get("Model", "")[-12:],
                persona=result.get("Persona", ""),
                ok=passed,
                fail=failed,
            )
            tqdm.write(
                f"  {result.get('Model')} | {result.get('Persona')} | "
                f"{result.get('Agent_Type')} | {scenario}{discount_info} | "
                f"seed{result.get('Seed')} → {status}"
            )

            # Incremental save — survives interruption
            pd.DataFrame(all_results).to_csv(master_log, index=False)

    print(f"\nAll done. Master summary: {master_log}")
    print(f"Per-step CSVs: {OUTPUT_ROOT}/<model>/<scenario>/seed<N>/")
    print("=" * 62)

    df_results = pd.DataFrame(all_results)
    if "Status" in df_results.columns:
        tally = df_results["Status"].value_counts()
        print("\nStatus tally:")
        for status, count in tally.items():
            print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
