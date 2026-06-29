"""
Placebo Agent for the Placebo Re-injection Control Experiment.

Implements π_placebo: identical to the memory agent in position and structure,
but injects semantically irrelevant regulatory boilerplate instead of the mandate.

This isolates whether the memory agent's benefit comes from:
  - Semantic mandate content (salience hypothesis), OR
  - Mere recency position of any appended text (positional artifact)
"""

import sys
import os

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from agent.static_agent import StaticAgent
from agent.schemas import TradeDecision

import time


# Spec §2.2: length-matched, position-matched, semantically irrelevant, plausibly formatted.
# Verified against prohibited word list: no risk/action/identity/emotional language.
PLACEBO_TEXT = (
    "NOTICE: This simulation is provided for research and evaluation purposes only. "
    "Past performance does not guarantee future results. All portfolio values are hypothetical."
)


class PlaceboAgent(StaticAgent):
    """
    Third-arm agent for the causal identification experiment.

    Appends PLACEBO_TEXT (not the mandate) to every user message, in the
    identical syntactic position as the mandate in ActiveMemoryAgent.
    Everything else is identical to StaticAgent.
    """

    def __init__(self, mbti_type: str, model_name: str = "claude-sonnet-4-6"):
        super().__init__(mbti_type, model_name)
        self._setup_placebo_chain()

    def _setup_placebo_chain(self):
        # Mirror the memory agent's prompt template exactly — only the injected
        # text differs (placebo vs. mandate).
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            (
                "human",
                "{input_data}\n\n{placebo_injection}\n\n"
                "IMPORTANT: You must return a valid JSON object with ALL 3 fields: "
                "'action', 'quantity' (0.0 if HOLD), and 'rationale'.\n\n"
                "{format_instructions}",
            ),
        ])

        self.chain = self.prompt_template.partial(
            persona=self.full_system_prompt,
            format_instructions=self.parser.get_format_instructions(),
        ) | self.llm | self.parser

    def decide(
        self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]
    ) -> TradeDecision:
        # Exact f-string from StaticAgent.decide() — the only difference between
        # arms must be the injected text, not the base observation (spec §2.4).
        input_data = f"""
        DATE: {market_state['date']}
        MARKET OBSERVATION:
        - Price: ${market_state['price']:.2f}
        - Trend (SMA20/60): {market_state['SMA20']} / {market_state['SMA60']} ({market_state.get('trend_regime', 0)})
        - RSI: {market_state['RSI14']}
        - P/E Ratio: {market_state.get('reported_PE', 'N/A')}
        - Implied Volatility: {market_state.get('implied_volatility', 'N/A')}%
        - Volume Ratio: {market_state.get('volume_ratio', 'N/A')}
        - News Sentiment: {market_state.get('news_sentiment', 'Neutral')}

        PORTFOLIO STATUS:
        - Cash: ${portfolio_state['cash']:.2f}
        - Holdings Value: ${portfolio_state['holdings_value']:.2f}
        """

        last_error = None
        for attempt in range(3):
            try:
                return self.chain.invoke(
                    {"input_data": input_data, "placebo_injection": PLACEBO_TEXT}
                )
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(1)

        return TradeDecision(
            action="HOLD",
            quantity=0.0,
            rationale=f"Error after 3 attempts: {str(last_error)}",
        )

    def reset(self):
        pass
