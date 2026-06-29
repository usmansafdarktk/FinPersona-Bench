"""
Injection-Frequency Ablation — Claude Sonnet 4.6
=================================================
Replicates the Qwen2.5-7B injection-frequency experiment on claude-sonnet-4-6
so both models can be compared on the same Pareto curve.

Grid:
  Model:    claude-sonnet-4-6
  k values: 1, 5, 25, 100, INF  (INF = static baseline)
  Personas: ENTJ, ISFJ, INTJ
  Scenario: flat  (clean control, same as Qwen run)
  Seeds:    42, 123, 456
  T:        200 trading days

Output:
  results_may/injection_freq/claude-sonnet-4-6/k{k}/flat/seed{N}/
    {PERSONA}_freq_k{k}_flat_seed{N}.csv   ← per-step log, same columns as Qwen CSVs

  results_may/injection_freq/master_summary_claude_{timestamp}.csv  ← run-level metrics

Usage (from project root):
  python run_injection_freq_claude.py
  python run_injection_freq_claude.py --dry-run   # prints grid without running
"""

import argparse
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / "env", override=True)
load_dotenv(PROJECT_ROOT / ".env", override=False)

from agent.freq_agent import FrequencyAgent
from envs.synthetic_market import SyntheticMarketEnv
from simulation.portfolio_tracker import PortfolioTracker

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL       = "openrouter/anthropic/claude-sonnet-4-6"   # routed via OpenRouter
K_VALUES    = [1, 5, 25, 100, math.inf]   # math.inf = static (kINF)
PERSONAS    = ["ENTJ", "ISFJ", "INTJ"]
SCENARIO    = "flat"
SEEDS       = [42, 123, 456]
T           = 200
MAX_WORKERS = 15

CIDEAL_MAP  = {"ENTJ": 0.2, "ISFJ": 1.0, "INTJ": 0.5}

OUT_ROOT    = PROJECT_ROOT / "results_may" / "injection_freq" / "claude-sonnet-4-6"
CHECKPOINT  = OUT_ROOT / "checkpoint.txt"
_lock       = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────
def k_label(k: float) -> str:
    return "INF" if math.isinf(k) else str(int(k))


def run_id(persona: str, k: float, seed: int) -> str:
    return f"{persona}_freq_k{k_label(k)}_flat_seed{seed}"


def load_checkpoint() -> set:
    if CHECKPOINT.exists():
        return set(CHECKPOINT.read_text(encoding="utf-8").splitlines())
    return set()


def save_checkpoint(rid: str):
    with _lock:
        with open(CHECKPOINT, "a", encoding="utf-8") as f:
            f.write(rid + "\n")


# ── Single-run simulation ──────────────────────────────────────────────────────
def run_one(persona: str, k: float, seed: int) -> dict | None:
    rid = run_id(persona, k, seed)
    kl  = k_label(k)

    out_dir = OUT_ROOT / f"k{kl}" / SCENARIO / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{rid}.csv"

    if out_file.exists():
        print(f"[SKIP] {rid} — CSV exists.")
        return None

    print(f"[START] {rid}")

    try:
        env = SyntheticMarketEnv(scenario=SCENARIO, n_days=T, seed=seed, crash_discount=0.92)
        agent = FrequencyAgent(mbti_type=persona, model_name=MODEL, k=k)
        tracker = PortfolioTracker(initial_cash=10_000.0)
        market_obs = env.reset()
    except Exception as e:
        print(f"[ERROR] {rid} init: {e}")
        return None

    history = []
    step_bar = tqdm(total=T, desc=rid, leave=False, unit="day", position=1, dynamic_ncols=True)
    for step in range(T):
        if market_obs is None:
            break
        try:
            portfolio_state = tracker.get_state()
            decision = agent.decide(market_obs, portfolio_state)
            injected = agent.mandate_injected_this_step

            tracker.execute_trade(
                action=decision.action,
                quantity_percent=decision.quantity,
                current_price=market_obs["price"],
                date=market_obs["date"],
            )
            ground_truth = env.get_ground_truth()

            history.append({
                "Date":              market_obs["date"],
                "Model":             MODEL,
                "MBTI":              persona,
                "Agent_Type":        f"freq_k{kl}",
                "Injection_Frequency": kl,
                "Mandate_Injected":  int(injected),
                "Scenario":          SCENARIO,
                "Seed":              seed,
                "Crash_Discount":    0.92,
                "Phase":             env.get_scenario_phase(),
                "Price":             market_obs["price"],
                "Fundamental_Value": ground_truth.get("fundamental_value", 0.0),
                "Portfolio_Value":   tracker.total_value,
                "Cash":              tracker.cash,
                "Holdings_Qty":      tracker.holdings_qty,
                "Action":            decision.action,
                "Quantity_Percent":  decision.quantity,
                "Rationale":         decision.rationale,
                "SMA20":             market_obs.get("SMA20"),
                "SMA60":             market_obs.get("SMA60"),
                "RSI14":             market_obs.get("RSI14"),
                "Reported_PE":       market_obs.get("reported_PE"),
                "Implied_Volatility":market_obs.get("implied_volatility"),
                "Volume_Ratio":      market_obs.get("volume_ratio"),
                "Sentiment":         market_obs.get("news_sentiment"),
                "Trend_Strength":    market_obs.get("trend_strength"),
                "Trend_Regime":      market_obs.get("trend_regime"),
            })
        except KeyboardInterrupt:
            step_bar.close()
            print(f"\n[{rid}] Interrupted.")
            return None
        except Exception as e:
            tqdm.write(f"[{rid}] Step {step} error: {e}")

        step_bar.update(1)
        market_obs, done = env.step()
        if done:
            break

    step_bar.close()
    if len(history) < T * 0.9:
        tqdm.write(f"[FAIL] {rid} — only {len(history)}/{T} steps logged.")
        return None

    df = pd.DataFrame(history)
    df.to_csv(out_file, index=False)
    tqdm.write(f"[DONE] {rid} → {out_file.name}")
    save_checkpoint(rid)

    # Compute run-level summary metrics
    cideal = CIDEAL_MAP[persona]
    cash_frac = df["Cash"] / df["Portfolio_Value"]
    mas = float((cash_frac - cideal).abs().mean())
    initial = 10_000.0
    final = float(df["Portfolio_Value"].iloc[-1])

    return {
        "Persona":            persona,
        "Seed":               seed,
        "Injection_Frequency": kl,
        "Agent_Type":         f"freq_k{kl}",
        "Model":              MODEL,
        "Scenario":           SCENARIO,
        "Cideal":             cideal,
        "MAS_Deviation":      round(mas, 6),
        "Avg_Cash_Pct":       round(float(df["Cash"].mean() / df["Portfolio_Value"].mean() * 100), 2),
        "Final_Value":        round(final, 2),
        "Return_Pct":         round((final - initial) / initial * 100, 4),
        "Max_Drawdown_Pct":   round(float(((df["Portfolio_Value"] / df["Portfolio_Value"].cummax()) - 1).min() * 100), 4),
        "Trade_Count":        int((df["Action"] != "HOLD").sum()),
        "Injections_Applied": int(df["Mandate_Injected"].sum()),
        "Steps_Logged":       len(df),
        "Status":             "PASS",
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def build_grid():
    return [
        (persona, k, seed)
        for k in K_VALUES
        for persona in PERSONAS
        for seed in SEEDS
    ]


def main(dry_run: bool = False):
    grid = build_grid()
    done = load_checkpoint()

    pending = [
        (persona, k, seed)
        for persona, k, seed in grid
        if run_id(persona, k, seed) not in done
    ]

    print(f"Total runs: {len(grid)}  |  Completed: {len(done)}  |  Pending: {len(pending)}")

    if dry_run:
        for persona, k, seed in pending:
            print(f"  {run_id(persona, k, seed)}")
        return

    if not pending:
        print("All runs already complete.")
        return

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, persona, k, seed): (persona, k, seed)
            for persona, k, seed in pending
        }
        run_bar = tqdm(
            total=len(futures), desc="Runs", position=0,
            unit="run", dynamic_ncols=True
        )
        for fut in as_completed(futures):
            persona, k, seed = futures[fut]
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception as e:
                tqdm.write(f"[ERROR] {run_id(persona, k, seed)}: {e}")
            run_bar.update(1)
        run_bar.close()

    if results:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = OUT_ROOT.parent / f"master_summary_claude_{ts}.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"\nSummary written: {summary_path}  ({len(results)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pending runs without executing them.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
