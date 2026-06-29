"""
Simulation Runner Module (FinPersona-Bench)

Orchestrates the 'Double-Blind' Tournament.
- Connects the Synthetic Market (Gym) to the Agent (Contestant).
- Manages the daily loop.
- Logs both the 'Public Observation' (Price) and 'Hidden Truth' (Value) for analysis.
"""

import os
import pandas as pd
from typing import Optional, Literal

# Internal Modules
from envs.synthetic_market import SyntheticMarketEnv  
from agent.static_agent import StaticAgent           
from agent.memory_agent import ActiveMemoryAgent     
from simulation.portfolio_tracker import PortfolioTracker

def run_simulation(
    mbti_type: str,
    agent_type: Literal["static", "memory"] = "static",
    scenario: str = "bull_trap",
    model_name: str = "gemini-2.5-flash",
    initial_cash: float = 10000.0,
    output_dir: str = "results",
    max_days: Optional[int] = None,
    seed: int = 42,
    crash_discount: float = 0.92,   
) -> Optional[pd.DataFrame]:
    """
    Runs a rigorous benchmark simulation.

    Args:
        mbti_type: The personality code (e.g., "ENTJ").
        agent_type: "static" (Baseline) or "memory" (SOTA Active Injection).
        scenario: "flat", "bull_trap", or "crash".
        initial_cash: Starting capital.
        output_dir: Folder to save results.
        max_days: Optional limit (defaults to environment max).
    """
    discount_suffix = f"_discount{crash_discount}" if scenario == "crash" else ""
    run_id = f"{mbti_type}_{agent_type}_{scenario}_seed{seed}{discount_suffix}"
    print(f"\n[Runner] Initializing Benchmark: {run_id}")

    # 1. Initialize The Gym (Environment)
    try:
        # Default to 100 days for standard FinPersona benchmarks
        env_days = max_days if max_days else 100
        env = SyntheticMarketEnv(
            scenario=scenario,
            n_days=env_days,
            seed=seed,
            crash_discount=crash_discount,   
        )
        print(f"[Runner] Environment '{scenario}' created ({env.n_days} days).")
    except Exception as e:
        print(f"[Runner] Error initializing environment: {e}")
        return None

    # 2. Initialize The Contestant (Agent)
    try:
        if agent_type == "memory":
            agent = ActiveMemoryAgent(mbti_type=mbti_type, model_name=model_name)
        else:
            agent = StaticAgent(mbti_type=mbti_type, model_name=model_name)
        print(f"[Runner] Agent '{agent.get_uid()}' ready.")
    except Exception as e:
        print(f"[Runner] Error initializing agent: {e}")
        return None

    # 3. Initialize Accounting
    tracker = PortfolioTracker(initial_cash=initial_cash)

    # 4. Prepare Logging
    history = []
    os.makedirs(output_dir, exist_ok=True)

    # 5. The Main Loop
    print(f"[Runner] Starting {env.n_days}-day simulation...")
    
    # We use env.step() logic manually here to handle the "Observation" vs "Truth" split
    # Reset environment first
    market_observation = env.reset()
    
    # Iterate through the environment
    for step_num in range(env.n_days):
        if step_num % 10 == 0:
            print(f"[{run_id}] Day {step_num}/{env.n_days}")
        if market_observation is None:
            break

        current_date = market_observation.get("date", f"step_{step_num}")
        current_price = market_observation.get("price", 0.0)

        try:
            # A. Get Agent Decision (Agent ONLY sees market_observation)
            portfolio_state = tracker.get_state()
            decision = agent.decide(market_observation, portfolio_state)

            # B. Execute Trade
            trade_value = tracker.execute_trade(
                action=decision.action,
                quantity_percent=decision.quantity,
                current_price=current_price,
                date=current_date,
            )

            # C. Capture "Hidden Truth" for the Referee (Evaluator)
            # This data was NOT shown to the agent
            ground_truth = env.get_ground_truth()

            # D. Log Everything (Public + Private Data)
            daily_log = {
                "Date": current_date,
                "Model": model_name,
                "MBTI": mbti_type,
                "Agent_Type": agent_type,
                "Scenario": scenario,
                "Seed": seed,
                "Crash_Discount": crash_discount,
                "Phase": env.get_scenario_phase(),
                "Price": current_price,
                "Fundamental_Value": ground_truth.get("fundamental_value", 0.0), # The Truth
                "Portfolio_Value": tracker.total_value,
                "Cash": tracker.cash,
                "Holdings_Qty": tracker.holdings_qty,
                "Action": decision.action,
                "Quantity_Percent": decision.quantity,
                "Rationale": decision.rationale,
                # Log the new metrics for debugging
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
                "Sentiment_Change": market_observation.get("sentiment_change")
            }
            history.append(daily_log)

        except KeyboardInterrupt:
            print("\n[Runner] Stopped by user.")
            break
        except Exception as e:
            print(f"\n[Runner] Error on {current_date}: {e}")
            # Don't skip the step — always advance the environment

        # Always step forward regardless of whether this day's logging succeeded
        market_observation, done = env.step()
        if done:
            break

    # 6. Save Results
    results_df = pd.DataFrame(history)

    # Guard: if fewer than 10% of days were logged, treat as a failed run
    if len(results_df) < env.n_days * 0.1:
        print(f"[Runner] WARNING: Only {len(results_df)}/{env.n_days} days logged. "
              f"Run may have failed silently.")
        return None

    filename = os.path.join(output_dir, f"{run_id}.csv")
    results_df.to_csv(filename, index=False)

    print(f"[Runner] Benchmark complete. Log saved to {filename}")

    # Final Stats
    final_value = tracker.total_value
    return_pct = ((final_value - initial_cash) / initial_cash) * 100
    print(f"\n--- Result: {mbti_type} ({agent_type}) in {scenario} ---")
    print(f"Initial: ${initial_cash:,.2f}")
    print(f"Final:   ${final_value:,.2f}")
    print(f"Return:  {return_pct:.2f}%")

    return results_df
