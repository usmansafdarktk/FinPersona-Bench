"""
FinPersona Long-Horizon Experiment Runner (T=800)
=================================================

Runs the representative longitudinal extension:
  Claude Sonnet 4.6 × 3 personas × 3 scenarios × 5 seeds × 2 architectures

For the crash scenario the full crash-discount sensitivity grid
[0.85, 0.92, 0.95] is preserved for consistency with the main panel.

Total simulations:
  flat      : 3p × 5s × 2a              =  30
  bull_trap : 3p × 5s × 2a              =  30
  crash     : 3p × 5s × 2a × 3 discounts=  90
  ─────────────────────────────────────────────
  TOTAL                                  = 150

Outputs mirror the main runner layout under results_long_horizon/:
  results_long_horizon/<model>/<scenario>/seed<N>/<run>.csv
  results_long_horizon/<model>/<scenario>/discount<d>/seed<N>/<run>.csv
  results_long_horizon/checkpoint.txt          ← resume state
  results_long_horizon/master_summary_<ts>.csv ← incremental aggregate
"""

import os
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from simulation.runner import run_simulation

# ── CONFIGURATION ────────────────────────────────────────────────────────────

MODEL          = "claude-sonnet-4-6"           # canonical name — used for IDs, paths, logs
INFERENCE_MODEL = "openrouter/anthropic/claude-sonnet-4-6"  # actual API call target
PERSONAS    = ["ENTJ", "ISFJ", "INTJ"]
SCENARIOS   = ["flat", "bull_trap", "crash"]
AGENT_TYPES = ["static", "memory"]
SEEDS       = [42, 123, 456, 789, 999]
CRASH_DISCOUNTS = [0.85, 0.92, 0.95]

T           = 800   # Trading days — the whole point of this runner
MAX_WORKERS = 15    # Concurrent API threads

OUTPUT_ROOT     = "results_long_horizon"
CHECKPOINT_FILE = os.path.join(OUTPUT_ROOT, "checkpoint.txt")

CIDEAL_MAP = {
    "ISFJ": 1.0,
    "INTJ": 0.5,
    "ENTJ": 0.2,
}

_checkpoint_lock = threading.Lock()
_results_lock    = threading.Lock()

# ── METRICS ──────────────────────────────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame, initial_cash: float, persona: str,
                      crash_discount: float) -> dict:
    if df is None or df.empty:
        return {}

    final_value       = df.iloc[-1]["Portfolio_Value"]
    total_return_pct  = ((final_value - initial_cash) / initial_cash) * 100

    rolling_max       = df["Portfolio_Value"].cummax()
    daily_drawdown    = df["Portfolio_Value"] / rolling_max - 1.0
    max_drawdown_pct  = daily_drawdown.min() * 100

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
                if pt > vt:
                    return 1
                return 1 if holdings > 1.0 else 0
            return float("nan")

        yt_values         = df.apply(compute_yt, axis=1)
        valid             = yt_values.dropna()
        rationality_score = round(float(valid.mean() * 100), 1) if len(valid) > 0 else float("nan")

    buy_decisions = df[df["Action"] == "BUY"]
    if not buy_decisions.empty and "Reported_PE" in buy_decisions.columns:
        pe_vals   = pd.to_numeric(buy_decisions["Reported_PE"], errors="coerce").fillna(0)
        avg_buy_pe = pe_vals.mean()
    else:
        avg_buy_pe = 0.0

    cideal           = CIDEAL_MAP.get(persona, 0.5)
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

# ── CHECKPOINT HELPERS ────────────────────────────────────────────────────────

def make_run_id(persona: str, agent_type: str, scenario: str,
                seed: int, crash_discount: float) -> str:
    if scenario == "crash":
        return f"{MODEL}__{persona}__{agent_type}__{scenario}__seed{seed}__discount{crash_discount}__T{T}"
    return f"{MODEL}__{persona}__{agent_type}__{scenario}__seed{seed}__T{T}"


def load_checkpoint() -> set:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(run_id: str):
    with _checkpoint_lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(run_id + "\n")

# ── SINGLE RUN WORKER ────────────────────────────────────────────────────────

def run_single(config: tuple) -> dict:
    persona, agent_type, scenario, seed, crash_discount = config
    run_id = make_run_id(persona, agent_type, scenario, seed, crash_discount)

    safe_model = MODEL.replace("/", "__")
    if scenario == "crash":
        output_dir = os.path.join(OUTPUT_ROOT, safe_model, scenario,
                                  f"discount{crash_discount}", f"seed{seed}")
    else:
        output_dir = os.path.join(OUTPUT_ROOT, safe_model, scenario, f"seed{seed}")

    base_result = {
        "Model":          MODEL,
        "Persona":        persona,
        "Agent_Type":     agent_type,
        "Scenario":       scenario,
        "Seed":           seed,
        "Crash_Discount": crash_discount,
        "T":              T,
    }

    try:
        df = run_simulation(
            mbti_type=persona,
            agent_type=agent_type,
            scenario=scenario,
            model_name=INFERENCE_MODEL,
            output_dir=output_dir,
            initial_cash=10_000.0,
            max_days=T,
            seed=seed,
            crash_discount=crash_discount,
        )

        if df is None:
            return {**base_result, "Status": "FAIL: silent_failure"}

        metrics = calculate_metrics(df, 10_000.0, persona, crash_discount)
        save_checkpoint(run_id)
        return {**base_result, **metrics, "Status": "PASS"}

    except Exception as e:
        return {**base_result, "Status": f"FAIL: {e}"}

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_configs = [
        (persona, agent_type, scenario, seed, crash_discount)
        for scenario       in SCENARIOS
        for seed           in SEEDS
        for persona        in PERSONAS
        for agent_type     in AGENT_TYPES
        for crash_discount in (CRASH_DISCOUNTS if scenario == "crash" else [0.92])
    ]

    completed = load_checkpoint()
    pending = [
        c for c in all_configs
        if make_run_id(*c) not in completed
    ]

    total   = len(all_configs)
    n_done  = total - len(pending)
    n_left  = len(pending)

    print("=" * 60)
    print("  FINPERSONA LONG-HORIZON RUNNER  (T=800)")
    print(f"  Model      : {MODEL}")
    print(f"  Total runs : {total}")
    print(f"  Completed  : {n_done}  (checkpoint)")
    print(f"  Remaining  : {n_left}")
    print(f"  Workers    : {MAX_WORKERS}")
    print("=" * 60)

    if not pending:
        print("Nothing to run — all configurations already completed.")
        return

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = os.path.join(OUTPUT_ROOT, f"master_summary_{timestamp}.csv")
    all_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, cfg): cfg for cfg in pending}

        with tqdm(total=n_left, desc="Simulations", unit="run",
                  dynamic_ncols=True, colour="green") as pbar:
            for future in as_completed(futures):
                result = future.result()

                with _results_lock:
                    all_results.append(result)
                    pd.DataFrame(all_results).to_csv(master_log, index=False)

                status = result.get("Status", "?")
                label  = (
                    f"{result.get('Persona')} | {result.get('Agent_Type')} | "
                    f"{result.get('Scenario')} | seed{result.get('Seed')} → {status}"
                )
                pbar.set_postfix_str(label, refresh=True)
                pbar.update(1)

    pass_count = sum(1 for r in all_results if r.get("Status") == "PASS")
    fail_count = len(all_results) - pass_count

    print("\n" + "=" * 60)
    print(f"  DONE  — {pass_count} PASS  |  {fail_count} FAIL")
    print(f"  Master log : {master_log}")
    print(f"  Traces     : {OUTPUT_ROOT}/{MODEL.replace('/', '__')}/<scenario>/seed<N>/")
    print("=" * 60)


if __name__ == "__main__":
    main()
