"""
Simulation Runner Module

This module orchestrates the backtest. It connects the MarketEnvironment,
the FinPersonaAgent, and the PortfolioTracker into a time-stepped loop.
"""

import os
import pandas as pd
from tqdm import tqdm
from typing import Optional

# Internal Modules
from legacy_market_data.market_environment import MarketEnvironment
from agent.legacy_agent import FinPersonaAgent
from simulation.portfolio_tracker import PortfolioTracker


def run_simulation(
    mbti_type: str,
    initial_cash: float = 10000.0,
    output_dir: str = "results",
    max_days: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Runs a full backtest for a specific agent personality.

    Args:
        mbti_type: The personality code (e.g., "ENTJ").
        initial_cash: Starting capital.
        output_dir: Folder to save the resulting CSV.
        max_days: Optional limit on trading days (for testing).

    Returns:
        A DataFrame containing the daily log of the simulation.
    """
    print(f"\n[Runner] Starting simulation for agent: {mbti_type}")

    # 1. Initialize Components
    try:
        env = MarketEnvironment()  # Loads data based on config.py
        agent = FinPersonaAgent(mbti_type=mbti_type)
        tracker = PortfolioTracker(initial_cash=initial_cash)
    except Exception as e:
        print(f"[Runner] Critical Error initializing components: {e}")
        return None

    # 2. Prepare Logging
    history = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Determine Loop Range
    total_available = len(env)
    if max_days is not None and max_days < total_available:
        days_to_run = max_days
        print(f"[Runner] Capping simulation at {days_to_run} days (Testing Mode).")
    else:
        days_to_run = total_available
        print(f"[Runner] Simulating full {days_to_run} trading days...")

    # 4. The Main Loop (Day-by-Day)
    for day_index in tqdm(range(days_to_run)):
        try:
            # A. Observe Environment (No look-ahead bias)
            market_state = env.get_state_for_day(day_index)
            current_date = market_state["date"]
            current_price = market_state["price"]

            # B. Get Agent Decision
            portfolio_state = tracker.get_state()
            decision = agent.decide(market_state, portfolio_state)

            # C. Execute Trade
            trade_value = tracker.execute_trade(
                action=decision.action,
                quantity_percent=decision.quantity,
                current_price=current_price,
                date=current_date,
            )

            # D. Log Everything
            daily_log = {
                "Date": current_date,
                "MBTI": mbti_type,
                "Price": current_price,
                "Portfolio_Value": tracker.total_value,
                "Cash": tracker.cash,
                "Holdings_Qty": tracker.holdings_qty,
                "Action": decision.action,
                "Quantity_Percent": decision.quantity,
                "Trade_Value": trade_value,
                "Rationale": decision.rationale,
            }
            history.append(daily_log)

        except KeyboardInterrupt:
            print("\n[Runner] Simulation stopped by user. Saving current progress...")
            break
        except Exception as e:
            print(f"\n[Runner] Error on day {day_index}: {e}")
            continue

    # 5. Save Results
    results_df = pd.DataFrame(history)
    filename = os.path.join(output_dir, f"{mbti_type}_simulation.csv")
    results_df.to_csv(filename, index=False)

    print(f"[Runner] Simulation complete. Results saved to {filename}")

    # Print final stats
    final_value = tracker.total_value
    return_pct = ((final_value - initial_cash) / initial_cash) * 100
    print(f" Final Performance ({mbti_type}) ")
    print(f"Initial: ${initial_cash:,.2f}")
    print(f"Final:   ${final_value:,.2f}")
    print(f"Return:  {return_pct:.2f}%")

    return results_df
