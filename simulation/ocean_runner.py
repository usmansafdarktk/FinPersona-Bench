"""
Simulation Runner — OCEAN / Big Five variant.

Identical orchestration logic to simulation/runner.py. The only structural
change is that agents are loaded from the OCEAN agent module and the persona
identifier column is 'Persona' (using OCEAN labels like 'O1_conservative').
All market environment parameters are held constant (spec Section 3.2).
Reuses the shared SyntheticMarketEnv and PortfolioTracker unchanged.
"""

import os
import pandas as pd
from typing import Optional, Literal

from envs.synthetic_market import SyntheticMarketEnv
from agent.ocean_static_agent import OceanStaticAgent
from agent.ocean_memory_agent import OceanMemoryAgent
from simulation.portfolio_tracker import PortfolioTracker


def run_ocean_simulation(
    persona_type: str,
    agent_type: Literal["static", "memory"] = "static",
    scenario: str = "flat",
    model_name: str = "claude-sonnet-4-6",
    initial_cash: float = 10_000.0,
    output_dir: str = "results_ocean",
    max_days: Optional[int] = None,
    seed: int = 42,
    crash_discount: float = 0.92,
) -> Optional[pd.DataFrame]:
    """
    Runs one OCEAN benchmark simulation.

    Args:
        persona_type:   One of 'O1_conservative', 'O2_aggressive',
                        'O3_conservative', 'O3_aggressive'.
        agent_type:     'static' (baseline) or 'memory' (active mandate injection).
        scenario:       'flat', 'bull_trap', or 'crash'.
        model_name:     API model identifier string.
        initial_cash:   Starting portfolio value.
        output_dir:     Directory for per-step CSV output.
        max_days:       Simulation horizon (default 200 per spec Section 3.2).
        seed:           RNG seed for market generation.
        crash_discount: Panic discount δ (only used in crash scenario).

    Returns:
        DataFrame with one row per trading day, or None on failure.
    """
    discount_suffix = f"_discount{crash_discount}" if scenario == "crash" else ""
    run_id = f"{persona_type}_{agent_type}_{scenario}_seed{seed}{discount_suffix}"
    print(f"\n[OceanRunner] Initializing: {run_id}")

    # 1. Market environment — parameters identical to FinPersona-Bench main panel
    try:
        env_days = max_days if max_days else 200
        env = SyntheticMarketEnv(
            scenario=scenario,
            n_days=env_days,
            seed=seed,
            crash_discount=crash_discount,
        )
        print(f"[OceanRunner] Environment '{scenario}' ready ({env.n_days} days, seed={seed}).")
    except Exception as e:
        print(f"[OceanRunner] Environment init failed: {e}")
        return None

    # 2. Agent
    try:
        if agent_type == "memory":
            agent = OceanMemoryAgent(persona_type=persona_type, model_name=model_name)
        else:
            agent = OceanStaticAgent(persona_type=persona_type, model_name=model_name)
        print(f"[OceanRunner] Agent '{agent.get_uid()}' ready.")
    except Exception as e:
        print(f"[OceanRunner] Agent init failed: {e}")
        return None

    # 3. Portfolio accounting
    tracker = PortfolioTracker(initial_cash=initial_cash)

    # 4. Logging setup
    history = []
    os.makedirs(output_dir, exist_ok=True)

    # 5. Main simulation loop
    print(f"[OceanRunner] Starting {env.n_days}-day simulation...")
    market_observation = env.reset()

    for step_num in range(env.n_days):
        if step_num % 20 == 0:
            print(f"[{run_id}] Day {step_num}/{env.n_days}")
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

            daily_log = {
                "Date": current_date,
                "Model": model_name,
                "Persona": persona_type,
                "Agent_Type": agent_type,
                "Scenario": scenario,
                "Seed": seed,
                "Crash_Discount": crash_discount,
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
                "Sentiment_MA5": market_observation.get("sentiment_MA5"),
                "Sentiment_Change": market_observation.get("sentiment_change"),
            }
            history.append(daily_log)

        except KeyboardInterrupt:
            print("\n[OceanRunner] Stopped by user.")
            break
        except Exception as e:
            print(f"\n[OceanRunner] Error on {current_date}: {e}")

        market_observation, done = env.step()
        if done:
            break

    # 6. Save results
    results_df = pd.DataFrame(history)

    if len(results_df) < env.n_days * 0.1:
        print(
            f"[OceanRunner] WARNING: Only {len(results_df)}/{env.n_days} days logged — "
            f"treating as failed run."
        )
        return None

    filename = os.path.join(output_dir, f"{run_id}.csv")
    results_df.to_csv(filename, index=False)
    print(f"[OceanRunner] Complete. Saved to {filename}")

    final_value = tracker.total_value
    return_pct = ((final_value - initial_cash) / initial_cash) * 100
    print(f"  Return: {return_pct:.2f}%  |  Final: ${final_value:,.2f}")

    return results_df
