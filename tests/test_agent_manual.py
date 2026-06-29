"""
Manual Test Script for FinPersonaAgent
Run: python test_agent_manual.py
"""

from agent.legacy_agent import FinPersonaAgent


def test_entj_agent():
    print(" Testing ENTJ Agent ")

    # 1. Initialize Agent
    agent = FinPersonaAgent(mbti_type="ENTJ")

    # 2. Mock Data (What the MarketEnvironment would provide)
    mock_market = {
        "date": "2020-03-15",
        "price": 150.00,
        "SMA20": 145.00,
        "SMA60": 140.00,  # Uptrend
        "RSI14": 65.0,
        "MACD": 1.5,
    }

    mock_portfolio = {"cash": 10000.0, "holdings_value": 0.0}

    # 3. Ask for a decision
    print("Thinking...")
    decision = agent.decide(mock_market, mock_portfolio)

    # 4. Print Result
    print("\n Decision Received ")
    print(f"Action: {decision.action}")
    print(f"Quantity: {decision.quantity * 100}%")
    print(f"Rationale: {decision.rationale}")


if __name__ == "__main__":
    test_entj_agent()
