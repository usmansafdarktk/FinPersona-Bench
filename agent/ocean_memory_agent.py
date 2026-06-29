"""
Ocean Active Memory Agent — OCEAN / Big Five memory-augmented variant.

Identical architecture to OceanStaticAgent except the Core Mandate (M) is
appended to the end of the user message at every time step t, immediately
after the market observation Ot. This positions M as the most proximal
token sequence prior to generation, leveraging the LLM's recency bias
(Liu et al., 2024) to maintain mandate salience over the 200-day horizon.

For O1/O2: mandate text is content-identical to ISFJ/ENTJ mandates.
For O3: mandate is purely numerical ("Your target cash allocation is X%").
"""

import time
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate

from agent.ocean_static_agent import OceanStaticAgent
from agent.schemas import TradeDecision
from agent.ocean_prompts import get_ocean_persona

load_dotenv()


class OceanMemoryAgent(OceanStaticAgent):
    """
    OCEAN memory-augmented agent.
    Re-injects the Core Mandate M at every decision step via recency-bias injection.
    """

    def __init__(self, persona_type: str, model_name: str = "claude-sonnet-4-6"):
        super().__init__(persona_type, model_name)
        self.core_mandate = get_ocean_persona(persona_type)["core_mandate"]
        self._setup_memory_chain()

    def _rebuild_chain(self):
        """Override: memory agent must rebuild with mandate slot, not the static chain."""
        self._setup_memory_chain()

    def _setup_memory_chain(self):
        """Overwrites the static chain with one that includes the mandate injection slot."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            (
                "human",
                "{input_data}\n\n"
                "{mandate_injection}\n\n"
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
        input_data = self._build_input_data(market_state, portfolio_state)

        # Recency-bias injection: mandate appended after market observation
        mandate_injection = (
            "*** ACTIVE MEMORY REFRESH ***\n"
            "Strictly adhere to your core mandate:\n"
            f"{self.core_mandate}\n\n"
            "Evaluate this trade ONLY through the lens of this mandate."
        )

        last_error = None
        for attempt in range(3):
            try:
                return self.chain.invoke({
                    "input_data": input_data,
                    "mandate_injection": mandate_injection,
                })
            except Exception as e:
                last_error = e
                print(f"[{self.get_uid()}] Parse error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
        return TradeDecision(
            action="HOLD",
            quantity=0.0,
            rationale=f"Error after 3 attempts: {str(last_error)}",
        )
