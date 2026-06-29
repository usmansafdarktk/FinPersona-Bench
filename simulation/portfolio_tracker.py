"""
Portfolio Tracker Module

This module handles the accounting for the simulation. It tracks
cash, shares held, and total portfolio value day-to-day.
"""


class PortfolioTracker:
    def __init__(self, initial_cash: float = 10000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings_qty = 0.0
        self.total_value = initial_cash
        # We keep a simplified transaction log for debugging
        self.trades = []

    def update_valuation(self, current_price: float):
        """Updates the total portfolio value based on current market price."""
        self.total_value = self.cash + (self.holdings_qty * current_price)

    def execute_trade(
        self, action: str, quantity_percent: float, current_price: float, date: str
    ) -> float:
        """
        Executes a trade based on the agent's decision.

        Args:
            action: "BUY", "SELL", or "HOLD"
            quantity_percent: 0.0 to 1.0
            current_price: The price at which to execute
            date: The date of the trade

        Returns:
            The actual executed value (dollars) of the trade
            (positive for buy, negative for sell).
        """
        trade_value = 0.0

        if action == "BUY" and quantity_percent > 0:
            # Calculate max we can buy
            budget = self.cash * quantity_percent
            shares_to_buy = budget / current_price

            # Update state
            self.cash -= budget
            self.holdings_qty += shares_to_buy
            trade_value = budget

            self.trades.append(
                {
                    "date": date,
                    "action": "BUY",
                    "shares": shares_to_buy,
                    "price": current_price,
                    "value": budget,
                }
            )

        elif action == "SELL" and quantity_percent > 0:
            # Calculate max we can sell
            shares_to_sell = self.holdings_qty * quantity_percent
            proceeds = shares_to_sell * current_price

            # Update state
            self.cash += proceeds
            self.holdings_qty -= shares_to_sell
            trade_value = -proceeds  # Negative indicates money entering cash
            # (convention) or just outflow of stock

            self.trades.append(
                {
                    "date": date,
                    "action": "SELL",
                    "shares": shares_to_sell,
                    "price": current_price,
                    "value": proceeds,
                }
            )

        # Recalculate total value immediately
        self.update_valuation(current_price)

        return trade_value

    def get_state(self) -> dict:
        """Returns the dictionary expected by the Agent's prompt."""
        return {
            "cash": self.cash,
            "holdings_value": self.total_value - self.cash,
        }
