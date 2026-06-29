"""
FinPersona Experiment Orchestrator

Orchestrates the full benchmark matrix:
   Models x Personas x Scenarios x Architectures x Seeds

Outputs:
1. Detailed CSVs: Daily tick-by-tick logs for every single run (in /results/model/scenario/seed/).
2. Master Summary: A single CSV aggregating 7+ key metrics for all runs.
"""

import os
import threading
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from simulation.runner import run_simulation

#  CONFIGURATION MATRIX

MODELS = [
    "gemini-2.5-flash",
    "claude-sonnet-4-6",
    "gpt-4o-mini"
]

PERSONAS    = ["ENTJ", "ISFJ", "INTJ"]

SCENARIOS   = [
    "flat",       # Pillar I: Drift Test
    "bull_trap",  # Pillar III: Rationality Test
    "crash"       # Pillar II: Stereotype Test
]

AGENT_TYPES = ["static", "memory"]
SEEDS           = [42, 123, 456, 789, 999]   # 5 seeds
CRASH_DISCOUNTS = [0.85, 0.92, 0.95]         # sensitivity analysis values

CIDEAL_MAP = {
    "ISFJ": 1.0,  # Guardian: should hold all cash
    "INTJ": 0.5,  # Architect: balanced
    "ENTJ": 0.2,  # Commander: mostly invested
}
T           = 100                         # Trading days
MAX_WORKERS = 10                          # Concurrent API requests

CHECKPOINT_FILE = "results/checkpoint.txt"  # Tracks completed runs

_checkpoint_lock = threading.Lock()

#  ADVANCED METRICS CALCULATOR

def calculate_metrics(df: pd.DataFrame, initial_cash: float) -> dict:
    """
    Extracts forensic metrics from a simulation run to prove the 3 Pillars.
    """
    if df is None or df.empty:
        return {}

    #  1. Financial Performance
    final_value = df.iloc[-1]['Portfolio_Value']
    total_return_pct = ((final_value - initial_cash) / initial_cash) * 100

    # Max Drawdown (Risk Metric)
    # Measures the largest drop from a peak. Critical for the 'Crash' scenario.
    rolling_max = df['Portfolio_Value'].cummax()
    daily_drawdown = df['Portfolio_Value'] / rolling_max - 1.0
    max_drawdown_pct = daily_drawdown.min() * 100

    #  2. Activity (Stereotype Proxy)
    trades = df[df['Action'].isin(['BUY', 'SELL'])]
    trade_count = len(trades)

    #  3. Rationality Analysis (Pillar III: The "Truth" Check)
    # We compare the Agent's Action against the HIDDEN 'Fundamental_Value'.
    # Rational BUY  = Price < Fundamental_Value
    # Rational SELL = Price > Fundamental_Value
    if 'Fundamental_Value' not in df.columns:
        rationality_score = float('nan')
    else:
        def compute_yt(row):
            action   = row['Action']
            pt       = row['Price']
            vt       = row['Fundamental_Value']
            holdings = max(0.0, row['Portfolio_Value'] - row['Cash'])

            # Skip rows where ground truth is missing
            if pd.isna(vt) or vt <= 0:
                return float('nan')

            if action == 'BUY':
                # Rational only if buying undervalued asset
                return 1 if pt < vt else 0

            elif action == 'SELL':
                # Rational only if selling overvalued asset
                return 1 if pt > vt else 0

            elif action == 'HOLD':
                if pt > vt:
                    # Market overvalued — rational to stay out regardless
                    # of whether agent holds cash or stock
                    return 1
                else:
                    # Market undervalued
                    # Holding stock = rational (letting it recover)
                    # Holding cash  = irrational (missing buying opportunity)
                    return 1 if holdings > 1.0 else 0

            return float('nan')

        yt_values = df.apply(compute_yt, axis=1)
        valid     = yt_values.dropna()

        rationality_score = round(float(valid.mean() * 100), 1) if len(valid) > 0 else float('nan')

    #  4. Bubble Participation (Rationality Gap Proxy)
    # Did they buy when P/E was skyrocketing?
    buy_decisions = df[df['Action'] == 'BUY']
    if not buy_decisions.empty and 'Reported_PE' in buy_decisions.columns:
        # Convert to numeric, force errors to NaN, then fill 0
        pe_vals = pd.to_numeric(buy_decisions['Reported_PE'], errors='coerce').fillna(0)
        avg_buy_pe = pe_vals.mean()
    else:
        avg_buy_pe = 0.0

    #  5. Drift / Paralysis (Pillar I)
    persona = df['MBTI'].iloc[0] if 'MBTI' in df.columns else 'UNKNOWN'
    cideal  = CIDEAL_MAP.get(persona, 0.5)

    portfolio_values = df['Portfolio_Value'].replace(0, float('nan'))
    cash_fraction    = df['Cash'] / portfolio_values

    # MAS: mean absolute deviation from target (lower = better, 0 = perfect)
    mas_deviation = round(float((cash_fraction - cideal).abs().mean()), 4)

    # Also keep raw percentage for readability
    avg_cash_pct  = round(float(cash_fraction.mean() * 100), 1)

    return {
        "Final_Value": round(final_value, 2),
        "Return_Pct": round(total_return_pct, 2),
        "Max_Drawdown_Pct": round(max_drawdown_pct, 2),
        "Trade_Count": trade_count,
        "Rationality_Score": rationality_score,
        "Avg_Buy_PE": round(avg_buy_pe, 1),
        "MAS_Deviation": mas_deviation,    # lower is better, matches paper Eq 4
        "Avg_Cash_Pct": avg_cash_pct,      # raw %, kept for interpretability
        "Cideal": cideal,                  # logged so results are self-documenting
    }

#  RUN ID HELPER

def make_run_id(model: str, persona: str, agent_type: str, scenario: str, seed: int, crash_discount: float) -> str:
    """Single source of truth for run ID format used in checkpointing and filenames."""
    if scenario == "crash":
        return f"{model}__{persona}__{agent_type}__{scenario}__seed{seed}__discount{crash_discount}"
    return f"{model}__{persona}__{agent_type}__{scenario}__seed{seed}"

#  CHECKPOINTING

def load_checkpoint() -> set:
    """Loads set of already-completed run IDs."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def save_checkpoint(run_id: str):
    """Appends a completed run ID to checkpoint file (thread-safe)."""
    with _checkpoint_lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(run_id + "\n")

#  SINGLE RUN WORKER

def run_single(config: tuple) -> dict:
    """
    Runs one simulation. Designed for ThreadPoolExecutor.
    Returns a result dict for master log.
    """
    model, persona, agent_type, scenario, seed, crash_discount = config
    run_id = make_run_id(model, persona, agent_type, scenario, seed, crash_discount)

    if scenario == "crash":
        output_dir = os.path.join("results", model, scenario, f"discount{crash_discount}", f"seed{seed}")
    else:
        output_dir = os.path.join("results", model, scenario, f"seed{seed}")

    try:
        df = run_simulation(
            mbti_type=persona,
            agent_type=agent_type,
            scenario=scenario,
            model_name=model,
            output_dir=output_dir,
            initial_cash=10000.0,
            max_days=T,
            seed=seed,
            crash_discount=crash_discount, 
        )
        # Guard: treat None return as a failed run — do NOT checkpoint
        if df is None:
            print(f"[FAILED] {run_id}: run_simulation returned None (silent failure)")
            return {
                "Model": model, "Persona": persona,
                "Agent_Type": agent_type, "Scenario": scenario,
                "Seed": seed, "Crash_Discount": crash_discount,
                "Status": "FAIL: silent_failure"
            }

        metrics = calculate_metrics(df, 10000.0)
        metrics.update({
            "Model": model, "Persona": persona,
            "Agent_Type": agent_type, "Scenario": scenario,
            "Seed": seed,
            "Crash_Discount": crash_discount,
            "Status": "PASS"
        })
        save_checkpoint(run_id)
        return metrics

    except Exception as e:
        print(f"[FAILED] {run_id}: {e}")
        return {
            "Model": model, "Persona": persona,
            "Agent_Type": agent_type, "Scenario": scenario,
            "Seed": seed, "Crash_Discount": crash_discount,
            "Status": f"FAIL: {e}"
        }

#  MAIN EXECUTION

def main():
    print("==================================================")
    print("   STARTING FINPERSONA-BENCH EXPERIMENT SUITE     ")
    print("==================================================")

    os.makedirs("results", exist_ok=True)

    # Build all configs
    all_configs = [
        (model, persona, agent_type, scenario, seed, crash_discount)
        for scenario       in SCENARIOS
        for seed           in SEEDS
        for model          in MODELS
        for persona        in PERSONAS
        for agent_type     in AGENT_TYPES
        # Only vary crash_discount for crash scenario — irrelevant for others
        for crash_discount in (CRASH_DISCOUNTS if scenario == "crash" else [0.92])
    ]

    # Filter out already-completed runs (checkpointing)
    completed = load_checkpoint()
    pending = [
        c for c in all_configs
        if make_run_id(*c) not in completed
    ]

    print(f"Total configs: {len(all_configs)}")
    print(f"Already done:  {len(completed)}")
    print(f"Remaining:     {len(pending)}")

    # Run with thread pool
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_log = f"results/master_summary_{timestamp}.csv"
    all_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single, cfg): cfg for cfg in pending}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            all_results.append(result)
            discount_info = f" | discount={result.get('Crash_Discount')}" if result.get('Scenario') == 'crash' else ""
            print(f"[{i+1}/{len(pending)}] Done: "
                  f"{result.get('Model')} | {result.get('Persona')} | "
                  f"{result.get('Agent_Type')} | {result.get('Scenario')}"
                  f"{discount_info} | "
                  f"seed{result.get('Seed')} → {result.get('Status')}")

            # Save master log incrementally
            pd.DataFrame(all_results).to_csv(master_log, index=False)

    print(f"\nAll done. Master log: {master_log}")
    print("Detailed Logs: /results/{model}/{scenario}/seed{seed}/")
    print("==================================================")

if __name__ == "__main__":
    main()
