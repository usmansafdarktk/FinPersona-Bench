"""
Ocean Static Agent — OCEAN / Big Five variant of the baseline agent.

Receives the OCEAN persona system prompt once at initialization.
No mandate re-injection during the run (stateless baseline condition).
Temperature = 0.2, no top_p — identical client config to StaticAgent (static_agent.py).

Claude routing: ANTHROPIC_API_KEY is used as primary. If unavailable or if an
authentication error is detected at inference time, the agent falls back to
OPENROUTER_API_KEY (OpenAI-compatible endpoint, model = anthropic/<model_name>).
"""

import os
import time
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from agent.base import BaseAgent
from agent.schemas import TradeDecision
from agent.ocean_prompts import get_ocean_persona

load_dotenv()

# Keywords that indicate an API key / authentication failure worth falling back on
_AUTH_KEYWORDS = (
    "authentication", "401", "invalid api key", "invalid_api_key",
    "unauthorized", "permission denied", "authenticationerror",
    "invalid x-api-key", "access denied",
)


class OceanStaticAgent(BaseAgent):
    """
    OCEAN baseline agent.
    Persona prompt injected once at init; no mandate re-injection during the run.
    """

    def __init__(self, persona_type: str, model_name: str = "claude-sonnet-4-6"):
        super().__init__(persona_type)
        self.model_name = model_name
        persona = get_ocean_persona(persona_type)
        self.full_system_prompt = persona["system_prompt"]
        self._fallback_llm = None  # populated for Claude if OPENROUTER_API_KEY is set
        self._setup_llm()
        self._setup_chain()

    def _setup_llm(self):
        # temperature=0.2, no top_p — identical to StaticAgent in static_agent.py
        temperature = 0.2

        name = self.model_name.lower()
        if "claude" in name:
            anthropic_key  = os.getenv("ANTHROPIC_API_KEY")
            openrouter_key = os.getenv("OPENROUTER_API_KEY")

            if anthropic_key:
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    temperature=temperature,
                    anthropic_api_key=anthropic_key,
                )
                # Pre-build the fallback so there's no delay when we need it
                if openrouter_key:
                    self._fallback_llm = ChatOpenAI(
                        model=f"anthropic/{self.model_name}",
                        temperature=temperature,
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
            elif openrouter_key:
                print(
                    f"  [agent] ANTHROPIC_API_KEY absent — "
                    f"routing {self.model_name} via OpenRouter."
                )
                self.llm = ChatOpenAI(
                    model=f"anthropic/{self.model_name}",
                    temperature=temperature,
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            else:
                raise ValueError(
                    "Claude model requested but neither ANTHROPIC_API_KEY "
                    "nor OPENROUTER_API_KEY is set."
                )

        elif "gpt" in name or "o1" in name or "o3" in name:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set.")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                api_key=api_key,
            )
        elif "gemini" in name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set.")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                google_api_key=api_key,
            )
        elif "deepseek" in name:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set.")
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _setup_chain(self):
        self.parser = PydanticOutputParser(pydantic_object=TradeDecision)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            (
                "human",
                "{input_data}\n\n"
                "IMPORTANT: You must return a valid JSON object with ALL 3 fields: "
                "'action', 'quantity' (0.0 if HOLD), and 'rationale'.\n\n"
                "{format_instructions}",
            ),
        ])
        self.chain = self.prompt_template.partial(
            persona=self.full_system_prompt,
            format_instructions=self.parser.get_format_instructions(),
        ) | self.llm | self.parser

    def _rebuild_chain(self):
        """Rebuild the LangChain chain after self.llm has been swapped."""
        self._setup_chain()

    def _activate_fallback(self) -> bool:
        """
        Switch self.llm to the OpenRouter fallback and rebuild the chain.
        Returns True if fallback was available and activated, False otherwise.
        """
        if self._fallback_llm is None:
            return False
        self.llm = self._fallback_llm
        self._fallback_llm = None  # one-time switch; prevents infinite loop
        self._rebuild_chain()
        print(f"[{self.get_uid()}] Switched to OpenRouter fallback for Claude.")
        return True

    def _is_auth_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in _AUTH_KEYWORDS)

    def _build_input_data(
        self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]
    ) -> str:
        return (
            f"DATE: {market_state['date']}\n"
            f"MARKET OBSERVATION:\n"
            f"- Price: ${market_state['price']:.2f}\n"
            f"- Trend (SMA20/SMA60): {market_state['SMA20']} / {market_state['SMA60']} "
            f"(regime: {market_state.get('trend_regime', 0)})\n"
            f"- RSI: {market_state['RSI14']}\n"
            f"- P/E Ratio: {market_state.get('reported_PE', 'N/A')}\n"
            f"- Implied Volatility: {market_state.get('implied_volatility', 'N/A')}%\n"
            f"- Volume Ratio: {market_state.get('volume_ratio', 'N/A')}\n"
            f"- News Sentiment: {market_state.get('news_sentiment', 'Neutral')}\n"
            f"\nPORTFOLIO STATUS:\n"
            f"- Cash: ${portfolio_state['cash']:.2f}\n"
            f"- Holdings Value: ${portfolio_state['holdings_value']:.2f}"
        )

    def decide(
        self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]
    ) -> TradeDecision:
        input_data = self._build_input_data(market_state, portfolio_state)
        last_error = None
        for attempt in range(3):
            try:
                return self.chain.invoke({"input_data": input_data})
            except Exception as e:
                last_error = e
                print(f"[{self.get_uid()}] Error (attempt {attempt + 1}/3): {e}")
                # Auth failure → try OpenRouter fallback immediately (no sleep)
                if self._is_auth_error(e) and self._activate_fallback():
                    continue
                if attempt < 2:
                    time.sleep(1)
        return TradeDecision(
            action="HOLD",
            quantity=0.0,
            rationale=f"Error after 3 attempts: {str(last_error)}",
        )

    def reset(self):
        pass
