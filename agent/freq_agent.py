"""
FrequencyAgent — mandate re-injection at every k steps.

k=1   → identical to ActiveMemoryAgent (inject every step)
k=INF → identical to StaticAgent (never re-inject after init)
k=N   → inject on steps 1, 1+N, 1+2N, ...

Inherits model routing from StaticAgent so it works with any model
already supported there (Claude, GPT, Gemini, OpenRouter).
"""

import math
import time
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from agent.static_agent import StaticAgent
from agent.schemas import TradeDecision


class FrequencyAgent(StaticAgent):
    """
    Periodic mandate re-injection agent.

    Args:
        mbti_type:  MBTI persona code (e.g. "ENTJ").
        model_name: LLM identifier (same strings as StaticAgent).
        k:          Injection period. Mandate is appended to the user turn
                    on every step where (step_number % k == 1), i.e. steps
                    1, 1+k, 1+2k, ... Step numbering starts at 1.
                    Pass math.inf (or a very large int) for static behaviour.
    """

    def __init__(self, mbti_type: str, model_name: str, k: float):
        super().__init__(mbti_type, model_name)
        self.k = k
        self._step = 0
        self._load_core_mandate()
        self._setup_freq_chain()

    def _load_core_mandate(self):
        import json, os
        json_path = os.path.join(os.path.dirname(__file__), "personas", "mbti_profiles.json")
        try:
            with open(json_path) as f:
                profiles = json.load(f)
        except FileNotFoundError:
            profiles = {}
        profile = profiles.get(self.mbti_type, {})
        self.core_mandate = profile.get(
            "core_mandate", "REMINDER: Act according to your personality."
        )

    def _setup_freq_chain(self):
        """One chain with an optional context_refresh slot."""
        self.parser  # already created by StaticAgent._setup_chain
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            (
                "human",
                "{input_data}{context_refresh}\n\n"
                "IMPORTANT: You must return a valid JSON object with ALL 3 fields: "
                "'action', 'quantity' (0.0 if HOLD), and 'rationale'.\n\n"
                "{format_instructions}",
            ),
        ])
        self.chain = self.prompt_template.partial(
            persona=self.full_system_prompt,
            format_instructions=self.parser.get_format_instructions(),
        ) | self.llm | self.parser

    def _should_inject(self) -> bool:
        """Return True if the mandate should be appended this step."""
        if math.isinf(self.k):
            return False
        # Steps are 1-indexed; inject on step 1 and every k steps after.
        return (self._step - 1) % self.k == 0

    @property
    def mandate_injected_this_step(self) -> bool:
        """Expose for the runner to log Mandate_Injected flag."""
        return self._last_injected

    def decide(self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]) -> TradeDecision:
        self._step += 1
        inject = self._should_inject()
        self._last_injected = inject

        input_data = (
            f"\n        DATE: {market_state['date']}\n"
            f"        MARKET OBSERVATION:\n"
            f"        - Price: ${market_state['price']:.2f}\n"
            f"        - Trend (SMA20/60): {market_state['SMA20']} / {market_state['SMA60']} "
            f"({market_state.get('trend_regime', 0)})\n"
            f"        - RSI: {market_state['RSI14']}\n"
            f"        - P/E Ratio: {market_state.get('reported_PE', 'N/A')}\n"
            f"        - Implied Volatility: {market_state.get('implied_volatility', 'N/A')}%\n"
            f"        - Volume Ratio: {market_state.get('volume_ratio', 'N/A')}\n"
            f"        - News Sentiment: {market_state.get('news_sentiment', 'Neutral')}\n\n"
            f"        PORTFOLIO STATUS:\n"
            f"        - Cash: ${portfolio_state['cash']:.2f}\n"
            f"        - Holdings Value: ${portfolio_state['holdings_value']:.2f}\n"
        )

        context_refresh = (
            f"\n\n        *** ACTIVE MEMORY REFRESH ***\n"
            f"        Strictly adhere to your core mandate:\n"
            f"        {self.core_mandate}\n"
            f"        Evaluate this trade ONLY through the lens of this mandate.\n"
            if inject else ""
        )

        last_error = None
        for attempt in range(3):
            try:
                return self.chain.invoke(
                    {"input_data": input_data, "context_refresh": context_refresh}
                )
            except Exception as e:
                last_error = e
                print(f"[{self.get_uid()}] Parse error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
        return TradeDecision(
            action="HOLD", quantity=0.0,
            rationale=f"Error after 3 attempts: {str(last_error)}",
        )

    def reset(self):
        self._step = 0
        self._last_injected = False
