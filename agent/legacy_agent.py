"""
FinPersona Agent Module

This module defines the FinPersonaAgent class, which represents a single
AI trading agent. It combines a specific MBTI personality with a
financial trading context to make decisions based on market data.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Internal Imports
from agent.schemas import TradeDecision
from agent.prompts import get_financial_persona

# Load environment variables (API Keys)
load_dotenv()


class FinPersonaAgent:
    """
    A financial trading agent powered by an LLM and conditioned with
    a specific MBTI personality.
    """

    def __init__(self, mbti_type: str, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the agent with a specific personality.

        Args:
            mbti_type: The MBTI type code (e.g., 'ENTJ', 'INFP').
            model_name: The name of the LLM to use (default: gemini-2.5-flash).
        """
        self.mbti_type = mbti_type
        self.model_name = model_name

        # 1. Setup the LLM
        # We use temperature=0.2 for consistent, deterministic trading decisions,
        # as recommended in the FinPersona proposal.
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.2, google_api_key=api_key
        )

        # 2. Setup the Output Parser (Enforces JSON Schema)
        self.parser = PydanticOutputParser(pydantic_object=TradeDecision)

        # 3. Setup the Prompt Template
        self.chain = self._build_chain()

        print(f"[FinPersonaAgent] Initialized {mbti_type} agent.")

    def _build_chain(self):
        """
        Constructs the LangChain processing pipeline (Prompt -> LLM -> Parser).
        """
        # Get the scientifically validated persona text
        system_persona = get_financial_persona(self.mbti_type)

        # Define the prompt structure
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{persona}"),
                (
                    "human",
                    """
            DATE: {date}

            YOUR CURRENT PORTFOLIO:
            - Cash Available: ${cash:.2f}
            - Current Holdings Value: ${holdings_value:.2f}

            MARKET DATA (AAPL):
            - Price: ${price:.2f}
            - SMA20 (Trend): {SMA20:.2f}
            - SMA60 (Trend): {SMA60:.2f}
            - RSI (Momentum): {RSI14:.2f}
            - MACD (Signal): {MACD:.4f}

            INSTRUCTIONS:
            Based strictly on your personality and the market data above,
            make a trading decision.
            {format_instructions}
            """,
                ),
            ]
        )

        # Partial application: Bake the persona and format instructions in now
        # so we don't have to pass them every time we call the chain.
        formatted_prompt = prompt.partial(
            persona=system_persona,
            format_instructions=self.parser.get_format_instructions(),
        )

        # Create the chain: Prompt -> Model -> JSON Parser
        return formatted_prompt | self.llm | self.parser

    def decide(
        self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]
    ) -> TradeDecision:
        """
        The core 'Policy' function. It observes the environment and outputs an action.

        Args:
            market_state: A dictionary containing price, date, and indicators.
            portfolio_state: A dictionary containing 'cash' and 'holdings_value'.

        Returns:
            A structured TradeDecision object (Action, Quantity, Rationale).
        """
        try:
            # Combine market and portfolio data into a single input dictionary
            inputs = {**market_state, **portfolio_state}

            # Invoke the LLM chain
            decision = self.chain.invoke(inputs)
            return decision

        except Exception as e:
            # Fallback mechanism: In production, we might want to log this
            # and return a safe "HOLD" decision rather than crashing the simulation.
            print(f"[FinPersonaAgent] Error making decision: {e}")

            # Return a safe default
            return TradeDecision(
                action="HOLD",
                quantity=0.0,
                rationale=f"Error during decision process: {str(e)}",
            )
