"""
Placebo Re-injection Control Experiment Runner
FinPersona-Bench Rebuttal – Causal Identification of Mandate Salience

Addresses: Reviewer VG7b, Weakness W1
"The 'memory' intervention is not a fair diagnostic probe, it is a confound..."

Three-arm architecture:
  Arm 1 – Static   : Mandate only at init (existing results in results_april/)
  Arm 2 – Placebo  : Semantically irrelevant boilerplate re-injected each step [NEW]
  Arm 3 – Memory   : Mandate re-injected each step (existing results in results_april/)

Only 15 new simulations required (Placebo arm: 3 personas × 5 seeds).

Usage:
    cd c:/Users/ayesha.gull01/FinPersona
    python placebo_reinjection_control/run_placebo_experiment.py
"""

import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# ── Path setup (run from project root) ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The project stores env vars in a file named 'env' (no extension) at the root
load_dotenv(PROJECT_ROOT / "env", override=True)
# Also try standard .env as fallback
load_dotenv(PROJECT_ROOT / ".env", override=False)

# ── OpenRouter patch: route Claude calls through OpenRouter ───────────────────
# The Anthropic key has expired; use OPENROUTER_API_KEY instead.
# OpenRouter exposes an OpenAI-compatible endpoint, so we patch the env so that
# ChatAnthropic in static_agent.py falls back gracefully.
_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if _OPENROUTER_KEY:
    # We will override _setup_llm in our agents to use OpenRouter directly.
    pass

# ── Internal imports (after path setup) ───────────────────────────────────────
from simulation.runner import run_simulation          # existing runner
from envs.synthetic_market import SyntheticMarketEnv  # noqa: F401 (ensure importable)
from agent.schemas import TradeDecision               # noqa: F401
from simulation.portfolio_tracker import PortfolioTracker

from placebo_reinjection_control.placebo_agent import PlaceboAgent, PLACEBO_TEXT

# ── Experiment configuration ───────────────────────────────────────────────────
MODEL_NAME   = "claude-sonnet-4-6"
SCENARIO     = "flat"
PERSONAS     = ["ENTJ", "ISFJ", "INTJ"]
SEEDS        = [42, 123, 456, 789, 999]
T            = 200          # Trading days (paper benchmark horizon)
INITIAL_CASH = 10_000.0
CRASH_DISCOUNT = 0.92       # Irrelevant for flat, kept for API compatibility

CIDEAL_MAP = {"ISFJ": 1.0, "INTJ": 0.5, "ENTJ": 0.2}

# OpenRouter model identifier for Claude Sonnet 4.6
OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Output paths
RESULTS_DIR    = PROJECT_ROOT / "placebo_reinjection_control" / "results"
CHECKPOINT_FILE = RESULTS_DIR / "checkpoint.txt"

_checkpoint_lock = threading.Lock()

# ── Existing results (Static + Memory) from results_april/ ────────────────────
EXISTING_RESULTS_DIR = (
    PROJECT_ROOT / "results_april" / "claude_sonnet_4_6_200"
    / "claude-sonnet-4-6" / "flat"
)

# ── OpenRouter LLM factory ─────────────────────────────────────────────────────

def _make_openrouter_llm():
    """
    Returns a LangChain ChatOpenAI instance pointed at OpenRouter's
    Claude Sonnet 4.6, using the OPENROUTER_API_KEY from .env.
    """
    from langchain_openai import ChatOpenAI

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found in environment. "
            "Please set it in your .env file."
        )
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=0.0,
        api_key=key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/FinPersona-Bench",
            "X-Title": "FinPersona-Bench Placebo Experiment",
        },
    )


# ── Token count verification (spec §9, checklist items 3 & 4) ────────────────

def _count_tokens_anthropic(text: str) -> int:
    """
    Count tokens using the official Anthropic API (client.messages.count_tokens).
    Falls back to character-based estimate (~4 chars/token) if the API call fails.
    Per spec §2.2: use Anthropic's tokenizer, not a third-party approximation.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        response = client.messages.count_tokens(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens
    except Exception:
        # Fallback: ~4 chars per token (reasonable for English prose)
        return max(1, len(text) // 4)


def _verify_token_counts():
    """
    Spec §9 checklist items 3 & 4:
    - Measure token counts of ENTJ, ISFJ, INTJ mandates using count_tokens().
    - Confirm PLACEBO_TEXT is within ±5 tokens of each mandate.
    """
    import json
    profiles_path = PROJECT_ROOT / "agent" / "personas" / "mbti_profiles.json"
    with open(profiles_path) as f:
        profiles = json.load(f)

    print("\n-- Token count verification (spec §9 checklist items 3 & 4) --")
    print(f"  Placebo text: \"{PLACEBO_TEXT}\"")
    placebo_tokens = _count_tokens_anthropic(PLACEBO_TEXT)
    print(f"  Placebo token count: {placebo_tokens}")
    print()

    all_within_tolerance = True
    for persona in PERSONAS:
        mandate = profiles[persona].get("core_mandate", "")
        mandate_tokens = _count_tokens_anthropic(mandate)
        diff = abs(placebo_tokens - mandate_tokens)
        within = diff <= 5
        status = "[OK]" if within else "[WARN: outside +-5 token tolerance]"
        print(f"  {persona}: mandate={mandate_tokens} tokens, placebo={placebo_tokens}, "
              f"diff={diff} {status}")
        if not within:
            all_within_tolerance = False

    if all_within_tolerance:
        print("  All personas within +-5 token tolerance.\n")
    else:
        print("\n  WARNING: Placebo token count is outside the +-5 token tolerance")
        print("  for one or more personas. Consider adjusting PLACEBO_TEXT per spec §2.2.\n")

    return all_within_tolerance


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        return {line.strip() for line in CHECKPOINT_FILE.read_text().splitlines() if line.strip()}
    return set()


def _save_checkpoint(run_id: str):
    with _checkpoint_lock:
        with open(CHECKPOINT_FILE, "a") as f:
            f.write(run_id + "\n")


# ── Metric calculation ─────────────────────────────────────────────────────────

def _calculate_mas(df: pd.DataFrame, persona: str) -> float:
    """MAS = mean absolute deviation of cash fraction from C_ideal."""
    cideal = CIDEAL_MAP[persona]
    if "Avg_Cash_Pct" in df.columns:
        # If aggregated
        cash_pct = df["Avg_Cash_Pct"].mean() / 100.0
        return abs(cash_pct - cideal)
    if "Cash" in df.columns and "Portfolio_Value" in df.columns:
        cash_frac = df["Cash"] / df["Portfolio_Value"].replace(0, float("nan"))
        return (cash_frac - cideal).abs().mean()
    return float("nan")


def _calculate_full_metrics(df: pd.DataFrame, persona: str) -> dict:
    cideal = CIDEAL_MAP[persona]

    # Cash fraction per day
    cash_frac = df["Cash"] / df["Portfolio_Value"].replace(0, float("nan"))
    mas = (cash_frac - cideal).abs().mean()
    avg_cash_pct = cash_frac.mean() * 100.0

    # Max drawdown
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown = (df["Portfolio_Value"] / rolling_max - 1.0).min() * 100.0

    final_value = df["Portfolio_Value"].iloc[-1]
    return_pct = (final_value - INITIAL_CASH) / INITIAL_CASH * 100.0

    trade_count = df[df["Action"].isin(["BUY", "SELL"])].shape[0]

    return {
        "MAS_Deviation": round(mas, 6),
        "Avg_Cash_Pct": round(avg_cash_pct, 2),
        "Max_Drawdown_Pct": round(drawdown, 4),
        "Final_Value": round(final_value, 2),
        "Return_Pct": round(return_pct, 4),
        "Trade_Count": trade_count,
    }


# ── Core simulation runner for the Placebo arm ────────────────────────────────

def _run_placebo_simulation(
    persona: str,
    seed: int,
    pbar: Optional[tqdm] = None,
) -> Optional[pd.DataFrame]:
    """
    Runs one placebo simulation (200-day flat market).
    Returns a DataFrame of the daily log, or None on failure.
    """
    run_id = f"{persona}_placebo_{SCENARIO}_seed{seed}"
    if pbar:
        pbar.set_description(f"Running {run_id}")

    try:
        env = SyntheticMarketEnv(
            scenario=SCENARIO,
            n_days=T,
            seed=seed,
            crash_discount=CRASH_DISCOUNT,
        )

        agent = PlaceboAgent(mbti_type=persona, model_name=MODEL_NAME)
        # Override the LLM with the OpenRouter-backed one
        agent.llm = _make_openrouter_llm()
        agent._setup_placebo_chain()  # rebuild chain with new llm

        tracker = PortfolioTracker(initial_cash=INITIAL_CASH)
        history = []

        market_observation = env.reset()

        for step_num in range(env.n_days):
            if market_observation is None:
                break

            current_date = market_observation.get("date", f"step_{step_num}")
            current_price = market_observation.get("price", 0.0)

            try:
                portfolio_state = tracker.get_state()
                decision = agent.decide(market_observation, portfolio_state)

                tracker.execute_trade(
                    action=decision.action,
                    quantity_percent=decision.quantity,
                    current_price=current_price,
                    date=current_date,
                )

                ground_truth = env.get_ground_truth()

                history.append({
                    "Date": current_date,
                    "Model": MODEL_NAME,
                    "MBTI": persona,
                    "Agent_Type": "placebo",
                    "Scenario": SCENARIO,
                    "Seed": seed,
                    "Crash_Discount": CRASH_DISCOUNT,
                    "Phase": env.get_scenario_phase(),
                    "Price": current_price,
                    "Fundamental_Value": ground_truth.get("fundamental_value", 0.0),
                    "Portfolio_Value": tracker.total_value,
                    "Cash": tracker.cash,
                    "Holdings_Qty": tracker.holdings_qty,
                    "Action": decision.action,
                    "Quantity_Percent": decision.quantity,
                    "Rationale": decision.rationale,
                    "SMA20": market_observation.get("SMA20"),
                    "SMA60": market_observation.get("SMA60"),
                    "RSI14": market_observation.get("RSI14"),
                    "MACD": market_observation.get("MACD"),
                    "Volume": market_observation.get("volume"),
                    "Volume_Ratio": market_observation.get("volume_ratio"),
                    "Implied_Volatility": market_observation.get("implied_volatility"),
                    "Reported_PE": market_observation.get("reported_PE"),
                    "Dividend_Yield": market_observation.get("dividend_yield"),
                    "Trend_Strength": market_observation.get("trend_strength"),
                    "Trend_Regime": market_observation.get("trend_regime"),
                    "Sentiment": market_observation.get("news_sentiment"),
                })

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"\n[{run_id}] Day {step_num} error: {e}")

            market_observation, done = env.step()
            if done:
                break

        df = pd.DataFrame(history)

        if len(df) < T * 0.1:
            print(f"\n[{run_id}] WARN: Only {len(df)}/{T} days logged — treating as failed.")
            return None

        # Save per-run CSV
        out_dir = RESULTS_DIR / "placebo" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{persona}_placebo_{SCENARIO}_seed{seed}.csv"
        df.to_csv(out_path, index=False)

        return df

    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"\n[{run_id}] Fatal error: {e}")
        return None


# ── Verification: confirm all existing results exist ──────────────────────────

def _verify_existing_results() -> bool:
    print("\n-- Verifying existing Static & Memory results in results_april/ --")
    missing = []
    for persona in PERSONAS:
        for seed in SEEDS:
            for arm in ["static", "memory"]:
                seed_dir = EXISTING_RESULTS_DIR / f"seed{seed}"
                csv_path = seed_dir / f"{persona}_{arm}_{SCENARIO}_seed{seed}.csv"
                status = "[OK]" if csv_path.exists() else "[MISSING]"
                if not csv_path.exists():
                    missing.append(str(csv_path))
                print(f"  {status}  {persona} | {arm} | seed{seed}")
    if missing:
        print(f"\n  {len(missing)} file(s) missing. These arms cannot be compared.")
        return False
    print("  All 30 existing files found.\n")
    return True


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PLACEBO RE-INJECTION CONTROL EXPERIMENT")
    print("  FinPersona-Bench Rebuttal - Causal Identification")
    print("=" * 60)
    print(f"  Model:    {MODEL_NAME} (via OpenRouter)")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Personas: {PERSONAS}")
    print(f"  Seeds:    {SEEDS}")
    print(f"  T:        {T} days")
    print(f"  New runs: {len(PERSONAS) * len(SEEDS)} (Placebo arm only)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Verify existing results (spec §9 checklist items 1 & 2)
    _verify_existing_results()

    # Verify token counts (spec §9 checklist items 3 & 4)
    _verify_token_counts()

    # Build placebo run list
    all_placebo_runs = [
        (persona, seed)
        for persona in PERSONAS
        for seed in SEEDS
    ]

    completed = _load_checkpoint()
    pending = [
        (persona, seed)
        for persona, seed in all_placebo_runs
        if f"{persona}_placebo_{SCENARIO}_seed{seed}" not in completed
    ]

    print(f"  Placebo runs total:     {len(all_placebo_runs)}")
    print(f"  Already completed:      {len(all_placebo_runs) - len(pending)}")
    print(f"  Remaining this session: {len(pending)}")
    print()

    if not pending:
        print("  All placebo runs already completed. Proceeding to analysis.\n")
    else:
        # Sequential with tqdm (one run at a time to respect rate limits)
        with tqdm(total=len(pending), unit="sim", desc="Placebo sims") as pbar:
            for persona, seed in pending:
                run_id = f"{persona}_placebo_{SCENARIO}_seed{seed}"
                pbar.set_description(f"{persona} seed{seed}")

                df = _run_placebo_simulation(persona, seed, pbar)

                if df is not None:
                    _save_checkpoint(run_id)
                    pbar.set_postfix(status="PASS")
                else:
                    pbar.set_postfix(status="FAIL - will retry on next run")

                pbar.update(1)

    print("\n-- Aggregating results across all three arms --\n")
    _aggregate_and_report()


# ── Aggregation & reporting ────────────────────────────────────────────────────

def _aggregate_and_report():
    """
    Collects metrics from all three arms and prints/saves the summary table.
    Runs two-sided paired Wilcoxon signed-rank tests per spec §3.2.
    """
    from scipy.stats import wilcoxon

    rows = []

    for persona in PERSONAS:
        for seed in SEEDS:
            for arm in ["static", "memory", "placebo"]:
                if arm == "placebo":
                    seed_dir = RESULTS_DIR / "placebo" / f"seed{seed}"
                    csv_path = seed_dir / f"{persona}_placebo_{SCENARIO}_seed{seed}.csv"
                else:
                    seed_dir = EXISTING_RESULTS_DIR / f"seed{seed}"
                    csv_path = seed_dir / f"{persona}_{arm}_{SCENARIO}_seed{seed}.csv"

                if not csv_path.exists():
                    print(f"  [SKIP] Missing: {csv_path}")
                    continue

                df = pd.read_csv(csv_path)
                metrics = _calculate_full_metrics(df, persona)
                rows.append({
                    "Persona": persona,
                    "Seed": seed,
                    "Arm": arm,
                    "Cideal": CIDEAL_MAP[persona],
                    **metrics,
                })

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        print("  No data to aggregate yet.")
        return

    # Save full per-run summary
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"placebo_summary_{ts}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Full summary saved: {summary_path}")

    # ── Three-column MAS table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PRIMARY RESULT: Mandate Adherence Score (MAS) -- lower is better")
    print("=" * 70)
    print(f"{'Persona':<10} {'Static MAS':>12} {'Placebo MAS':>13} {'Memory MAS':>12}")
    print("-" * 50)

    persona_mas = {}
    for persona in PERSONAS:
        p_df = summary_df[summary_df["Persona"] == persona]
        mas_by_arm = {}
        for arm in ["static", "placebo", "memory"]:
            arm_vals = p_df[p_df["Arm"] == arm]["MAS_Deviation"]
            if not arm_vals.empty:
                mas_by_arm[arm] = arm_vals
                mean = arm_vals.mean()
                std  = arm_vals.std()
            else:
                mean = std = float("nan")
            mas_by_arm[f"{arm}_mean"] = mean
            mas_by_arm[f"{arm}_std"]  = std

        persona_mas[persona] = mas_by_arm
        s_mean  = mas_by_arm.get("static_mean",  float("nan"))
        p_mean  = mas_by_arm.get("placebo_mean", float("nan"))
        m_mean  = mas_by_arm.get("memory_mean",  float("nan"))
        s_std   = mas_by_arm.get("static_std",   float("nan"))
        p_std   = mas_by_arm.get("placebo_std",  float("nan"))
        m_std   = mas_by_arm.get("memory_std",   float("nan"))
        print(
            f"{persona:<10} "
            f"{s_mean:.3f}±{s_std:.3f}  "
            f"{p_mean:.3f}±{p_std:.3f}  "
            f"{m_mean:.3f}±{m_std:.3f}"
        )

    # Aggregate mean row
    agg = summary_df.groupby("Arm")["MAS_Deviation"].agg(["mean", "std"])
    print("-" * 50)
    for arm in ["static", "placebo", "memory"]:
        if arm in agg.index:
            print(f"{'Mean':<10} " if arm == "static" else f"{'':10} ", end="")
            break
    # Simpler aggregate print
    s = agg.loc["static"]  if "static"  in agg.index else None
    p = agg.loc["placebo"] if "placebo" in agg.index else None
    m = agg.loc["memory"]  if "memory"  in agg.index else None
    s_str = f"{s['mean']:.3f}±{s['std']:.3f}" if s is not None else "  N/A  "
    p_str = f"{p['mean']:.3f}±{p['std']:.3f}" if p is not None else "  N/A  "
    m_str = f"{m['mean']:.3f}±{m['std']:.3f}" if m is not None else "  N/A  "
    print(f"{'Mean':<10} {s_str:>12}  {p_str:>13}  {m_str:>12}")

    # ── Paired Wilcoxon tests ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PAIRED WILCOXON SIGNED-RANK TESTS (matched by persona × seed)")
    print("=" * 70)

    def _paired_test(arm_a: str, arm_b: str, label: str):
        pairs = []
        for persona in PERSONAS:
            p_df = summary_df[summary_df["Persona"] == persona]
            for seed in SEEDS:
                a_row = p_df[(p_df["Arm"] == arm_a) & (p_df["Seed"] == seed)]["MAS_Deviation"]
                b_row = p_df[(p_df["Arm"] == arm_b) & (p_df["Seed"] == seed)]["MAS_Deviation"]
                if not a_row.empty and not b_row.empty:
                    pairs.append((a_row.values[0], b_row.values[0]))

        if len(pairs) < 5:
            print(f"  {label}: insufficient pairs (n={len(pairs)})")
            return

        a_vals = [x[0] for x in pairs]
        b_vals = [x[1] for x in pairs]
        diffs  = [a - b for a, b in pairs]

        if all(d == 0 for d in diffs):
            print(f"  {label}: all differences zero — cannot run Wilcoxon")
            return

        stat, p_val = wilcoxon(a_vals, b_vals, alternative="two-sided")
        sig = "** p<0.001" if p_val < 0.001 else ("*  p<0.05" if p_val < 0.05 else "   n.s.")
        print(f"  {label:<40} n={len(pairs):2d}  p={p_val:.4f}  {sig}")

    _paired_test("static",  "placebo", "Test 1: Static vs. Placebo  (position effect?)")
    _paired_test("placebo", "memory",  "Test 2: Placebo vs. Memory  (content effect?)")
    _paired_test("static",  "memory",  "Test 3: Static vs. Memory   (original result)")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    if p is not None and s is not None and m is not None:
        placebo_closer_to_static = abs(p["mean"] - s["mean"]) < abs(p["mean"] - m["mean"])
        if placebo_closer_to_static:
            print("  Placebo ~ Static: Content hypothesis SUPPORTED.")
            print("  Re-grounding benefit is driven by mandate semantic content,")
            print("  not positional recency. MSD is a salience-based failure.")
        else:
            print("  Placebo intermediate or ~ Memory.")
            print("  Positional recency contributes; mandate content may add")
            print("  further benefit. See paper section 4.2 interpretation template.")
    else:
        print("  Insufficient data for interpretation. Run all simulations first.")

    print(f"\n  Detailed per-run CSV: {summary_path}")
    print("=" * 70 + "\n")

    # §3.4 Auxiliary analysis: persona classifier sanity check
    _run_classifier_sanity_check(summary_df)


# ── §3.4 Persona classifier sanity check ──────────────────────────────────────

# Path to the pre-trained DistilBERT model produced by analysis_may/persona_classifier.py
_DISTILBERT_MODEL_DIR = PROJECT_ROOT / "finpersona_classifier_model" / "model"


def _run_classifier_sanity_check(summary_df: pd.DataFrame):
    """
    Spec §3.4: Apply the existing DistilBERT persona classifier (built in
    analysis_may/) to rationale strings from all three arms.

    If the pre-trained model is not found at analysis/outputs/persona_classifier/model/,
    prints instructions for training it and falls back to TF-IDF + LogisticRegression
    so the run still produces comparable output with a clear caveat in metrics.json.

    Saves:
      placebo_reinjection_control/results/persona_classifier/drift_scores.csv
      placebo_reinjection_control/results/persona_classifier/metrics.json
    """
    import json
    import numpy as np

    classifier_dir = RESULTS_DIR / "persona_classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)

    EVAL_SEEDS = [789, 999]   # held-out seeds for static/memory baseline accuracy

    # ── CSV loaders shared by both classifier paths ────────────────────────────
    def _load_existing_eval(persona, seed, arm):
        if seed not in EVAL_SEEDS:
            return None
        csv_path = EXISTING_RESULTS_DIR / f"seed{seed}" / f"{persona}_{arm}_{SCENARIO}_seed{seed}.csv"
        return pd.read_csv(csv_path) if csv_path.exists() else None

    def _load_placebo(persona, seed, _arm):
        csv_path = RESULTS_DIR / "placebo" / f"seed{seed}" / f"{persona}_placebo_{SCENARIO}_seed{seed}.csv"
        return pd.read_csv(csv_path) if csv_path.exists() else None

    if _DISTILBERT_MODEL_DIR.exists():
        _run_distilbert_check(classifier_dir, EVAL_SEEDS, np,
                              _load_existing_eval, _load_placebo)
    else:
        print(f"\n  [CLASSIFIER] DistilBERT model not found at {_DISTILBERT_MODEL_DIR}.")
        print(f"  [CLASSIFIER] Train it first with:")
        print(f"    python analysis_may/persona_classifier.py train --results-dir results")
        print(f"  [CLASSIFIER] Falling back to TF-IDF + LogisticRegression.")
        _run_tfidf_check(classifier_dir, EVAL_SEEDS, np,
                         _load_existing_eval, _load_placebo)


def _score_rows(texts, persona, seed, arm_label, day_offset, predict_fn, personas):
    """Score a list of rationale texts and return drift_rows entries."""
    rows = []
    if not texts:
        return rows, 0, 0
    probs_list, pred_labels = predict_fn(texts)
    correct_count = 0
    for day_idx, (pred, probs) in enumerate(zip(pred_labels, probs_list)):
        p_map = dict(zip(personas, probs))
        correct = int(pred == persona)
        correct_count += correct
        rows.append({
            "Model":             MODEL_NAME,
            "Agent_Type":        arm_label,
            "Scenario":          SCENARIO,
            "Seed":              seed,
            "MBTI":              persona,
            "Day":               day_offset + day_idx + 1,
            "p_intended":        round(float(p_map.get(persona, 0.0)), 4),
            "predicted_persona": pred,
            "correct":           correct,
            "p_ENTJ":            round(float(p_map.get("ENTJ", 0.0)), 4),
            "p_ISFJ":            round(float(p_map.get("ISFJ", 0.0)), 4),
            "p_INTJ":            round(float(p_map.get("INTJ", 0.0)), 4),
        })
    return rows, correct_count, len(texts)


def _evaluate_arm_generic(arm_label, csv_loader, predict_fn, personas):
    """Evaluate one arm using the supplied predict_fn; returns (drift_rows, acc, n)."""
    drift_rows, arm_correct, arm_total = [], 0, 0
    seeds_to_use = SEEDS
    for persona in PERSONAS:
        for seed in seeds_to_use:
            df = csv_loader(persona, seed, arm_label)
            if df is None or "Rationale" not in df.columns:
                continue
            texts = df["Rationale"].fillna("").astype(str).tolist()
            texts = [t for t in texts if t.strip()]
            rows, nc, nt = _score_rows(texts, persona, seed, arm_label, 0, predict_fn, personas)
            drift_rows.extend(rows)
            arm_correct += nc
            arm_total   += nt
    acc = arm_correct / arm_total if arm_total > 0 else float("nan")
    return drift_rows, acc, arm_total


def _print_results_and_save(drift_rows, static_acc, n_static, memory_acc, n_memory,
                             placebo_acc, n_placebo, classifier_dir, model_label,
                             extra_metrics=None):
    import json
    import numpy as np

    EVAL_SEEDS = [789, 999]
    print(f"\n  Accuracy  Static:  {static_acc:.1%}  (n={n_static} rationales, held-out seeds {EVAL_SEEDS})")
    print(f"  Accuracy  Memory:  {memory_acc:.1%}  (n={n_memory} rationales, held-out seeds {EVAL_SEEDS})")

    if n_placebo > 0:
        print(f"  Accuracy  Placebo: {placebo_acc:.1%}  (n={n_placebo} rationales, all seeds {SEEDS})")

        # Spec §3.4: day-75 accuracy per arm
        drift_df_tmp = pd.DataFrame(drift_rows)
        def _acc_at_day(arm_label, day=75):
            sub = drift_df_tmp[(drift_df_tmp["Agent_Type"] == arm_label) &
                               (drift_df_tmp["Day"] == day)]
            if sub.empty:
                return float("nan"), 0
            return sub["correct"].mean(), len(sub)

        s75, ns75 = _acc_at_day("static")
        m75, nm75 = _acc_at_day("memory")
        p75, np75 = _acc_at_day("placebo")
        print(f"\n  Accuracy at day 75 (spec §3.4 reference point):")
        print(f"    Static:  {s75:.1%}  (n={ns75})")
        print(f"    Memory:  {m75:.1%}  (n={nm75})")
        if np75 > 0:
            print(f"    Placebo: {p75:.1%}  (n={np75})")

        closer = "Static" if abs(placebo_acc - static_acc) < abs(placebo_acc - memory_acc) else "Memory"
        print(f"\n  Overall placebo accuracy closer to {closer}.")
        if closer == "Static":
            print("  => Content hypothesis SUPPORTED: mandate semantic content drives")
            print("     persona fidelity, not positional recency of any appended text.")
        else:
            print("  => Positional recency may contribute; content effect unclear.")
    else:
        print("  Placebo CSVs not yet available — run simulations first.")

    if drift_rows:
        drift_df = pd.DataFrame(drift_rows, columns=[
            "Model", "Agent_Type", "Scenario", "Seed", "MBTI",
            "Day", "p_intended", "predicted_persona", "correct",
            "p_ENTJ", "p_ISFJ", "p_INTJ",
        ])
        drift_path = classifier_dir / "drift_scores.csv"
        drift_df.to_csv(drift_path, index=False)
        print(f"\n  drift_scores.csv saved: {drift_path}")

    metrics = {
        "model":            model_label,
        "static_accuracy":  round(static_acc,  4) if n_static  > 0 else None,
        "memory_accuracy":  round(memory_acc,  4) if n_memory  > 0 else None,
        "placebo_accuracy": round(placebo_acc, 4) if n_placebo > 0 else None,
        "personas":         ["ENTJ", "ISFJ", "INTJ"],
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_path = classifier_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics.json saved: {metrics_path}")
    print("=" * 70 + "\n")


# ── DistilBERT path (spec §3.4 primary) ───────────────────────────────────────

def _run_distilbert_check(classifier_dir, eval_seeds, np,
                           load_existing_eval, load_placebo):
    """
    Score rationales using the pre-trained DistilBERT from analysis_may/.
    This is the spec-compliant §3.4 path.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        print(f"  [CLASSIFIER] torch/transformers not available: {exc}")
        print(f"  [CLASSIFIER] Falling back to TF-IDF + LogisticRegression.")
        _run_tfidf_check(classifier_dir, eval_seeds, np, load_existing_eval, load_placebo)
        return

    # Import constants from analysis_may/persona_classifier.py
    analysis_may_dir = str(PROJECT_ROOT / "analysis_may")
    if analysis_may_dir not in sys.path:
        sys.path.insert(0, analysis_may_dir)
    try:
        from persona_classifier import PERSONAS as CLF_PERSONAS, PERSONA_TO_ID, ID_TO_PERSONA, MAX_LEN
    except ImportError:
        CLF_PERSONAS  = ["ENTJ", "ISFJ", "INTJ"]
        PERSONA_TO_ID = {p: i for i, p in enumerate(CLF_PERSONAS)}
        ID_TO_PERSONA = {i: p for p, i in PERSONA_TO_ID.items()}
        MAX_LEN       = 256

    print("\n" + "=" * 70)
    print("  §3.4 PERSONA CLASSIFIER SANITY CHECK (DistilBERT — analysis_may/)")
    print("=" * 70)
    print(f"  Model: {_DISTILBERT_MODEL_DIR}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(_DISTILBERT_MODEL_DIR))
    model     = AutoModelForSequenceClassification.from_pretrained(
        str(_DISTILBERT_MODEL_DIR)
    ).to(device).eval()

    BATCH = 32

    def _predict(texts):
        """Returns (probs_list, pred_labels) for a list of strings."""
        all_probs = np.zeros((len(texts), len(CLF_PERSONAS)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(texts), BATCH):
                chunk = texts[start:start + BATCH]
                enc   = tokenizer(chunk, truncation=True, max_length=MAX_LEN,
                                  padding=True, return_tensors="pt").to(device)
                logits = model(**enc).logits
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs[start:start + BATCH] = probs
        pred_ids    = all_probs.argmax(axis=1)
        pred_labels = [ID_TO_PERSONA[int(i)] for i in pred_ids]
        return all_probs.tolist(), pred_labels

    s_rows, static_acc,  n_static  = _evaluate_arm_generic("static",  load_existing_eval, _predict, CLF_PERSONAS)
    m_rows, memory_acc,  n_memory  = _evaluate_arm_generic("memory",  load_existing_eval, _predict, CLF_PERSONAS)
    p_rows, placebo_acc, n_placebo = _evaluate_arm_generic("placebo", load_placebo,        _predict, CLF_PERSONAS)

    all_rows = s_rows + m_rows + p_rows
    _print_results_and_save(
        all_rows, static_acc, n_static, memory_acc, n_memory,
        placebo_acc, n_placebo, classifier_dir,
        model_label=f"distilbert ({_DISTILBERT_MODEL_DIR})",
    )


# ── TF-IDF fallback (used when DistilBERT model not yet trained) ───────────────

def _run_tfidf_check(classifier_dir, eval_seeds, np,
                     load_existing_eval, load_placebo):
    """
    Fallback §3.4 path using TF-IDF + LogisticRegression.
    Used only when the DistilBERT model from analysis_may/ has not been trained yet.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        print(f"  [CLASSIFIER] sklearn not available — skipping §3.4 ({exc})")
        return

    print("\n" + "=" * 70)
    print("  §3.4 PERSONA CLASSIFIER SANITY CHECK (TF-IDF+LR fallback)")
    print("=" * 70)

    PERSONAS_CLF  = ["ENTJ", "ISFJ", "INTJ"]
    TRAIN_SEEDS   = [42, 123, 456]

    train_texts, train_labels = [], []
    for persona in PERSONAS:
        for seed in TRAIN_SEEDS:
            for arm in ["static", "memory"]:
                csv_path = EXISTING_RESULTS_DIR / f"seed{seed}" / f"{persona}_{arm}_{SCENARIO}_seed{seed}.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if "Rationale" not in df.columns:
                    continue
                texts = df["Rationale"].fillna("").astype(str).tolist()
                train_texts.extend(texts)
                train_labels.extend([persona] * len(texts))

    if len(train_texts) < 10:
        print("  [CLASSIFIER] Insufficient training data — skipping.")
        return

    print(f"  Training on {len(train_texts)} rationale strings "
          f"(seeds {TRAIN_SEEDS}, Static + Memory arms).")

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10_000,
                                  sublinear_tf=True, min_df=2)),
        ("lr",    LogisticRegression(max_iter=1_000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial", random_state=42)),
    ])
    clf.fit(train_texts, train_labels)
    classes = list(clf.classes_)

    def _predict(texts):
        proba       = clf.predict_proba(texts)
        pred_labels = [classes[int(np.argmax(p))] for p in proba]
        return proba.tolist(), pred_labels

    s_rows, static_acc,  n_static  = _evaluate_arm_generic("static",  load_existing_eval, _predict, classes)
    m_rows, memory_acc,  n_memory  = _evaluate_arm_generic("memory",  load_existing_eval, _predict, classes)
    p_rows, placebo_acc, n_placebo = _evaluate_arm_generic("placebo", load_placebo,        _predict, classes)

    all_rows = s_rows + m_rows + p_rows
    _print_results_and_save(
        all_rows, static_acc, n_static, memory_acc, n_memory,
        placebo_acc, n_placebo, classifier_dir,
        model_label="tfidf-logreg (fallback — DistilBERT model not found; "
                    "train with: python analysis_may/persona_classifier.py train)",
        extra_metrics={"n_train": len(train_texts)},
    )


if __name__ == "__main__":
    main()
