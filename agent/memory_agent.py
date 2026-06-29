import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Internal Imports
from agent.base import BaseAgent
from agent.static_agent import StaticAgent
from agent.schemas import TradeDecision

load_dotenv()

class ActiveMemoryAgent(StaticAgent):
    """
    SOTA Agent (ReMemR1 Inspired).
    
    Architecture:
    - Inherits base setup from StaticAgent.
    - Implements 'Periodic Mandate Retrieval' (Pillar I).
    - Injects the Core Mandate into the ACTIVE CONTEXT (User Prompt) 
      before every decision to prevent attention decay.
      
    Hypothesis:
    - This agent will maintain a high Mandate Adherence Score (MAS) 
      and resist drifting into neutrality.
    """

    def __init__(self, mbti_type: str, model_name: str = "gemini-2.0-flash"):
        # Initialize the base StaticAgent to get the LLM and Parser setup
        super().__init__(mbti_type, model_name)
        
        # Load the specific 'Core Mandate' for active injection
        self._load_core_mandate()
        
        # Re-build the chain to accept the extra 'context_refresh' variable
        self._setup_memory_chain()

    def _load_core_mandate(self):
        """
        Loads the short, imperative 'Core Mandate' from the JSON.
        """
        json_path = os.path.join(os.path.dirname(__file__), "personas", "mbti_profiles.json")
        try:
            with open(json_path, 'r') as f:
                profiles = json.load(f)
        except FileNotFoundError:
            profiles = {}
            
        profile = profiles.get(self.mbti_type, profiles.get("DEFAULT", {}))
        
        # This is the "Needle" we inject every step
        self.core_mandate = profile.get("core_mandate", "REMINDER: Act according to your personality.")

    def _setup_memory_chain(self):
        """
        Overwrites the StaticAgent chain with one that accepts 'context_refresh'.
        """
        # We modify the prompt template to include a slot for the mandate injection
        # at the VERY END of the user input (Recency Bias).
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            ("human", "{input_data}\n\n{context_refresh}\n\nIMPORTANT: You must return a valid JSON object with ALL 3 fields: 'action', 'quantity' (0.0 if HOLD), and 'rationale'.\n\n{format_instructions}")
        ])

        self.chain = self.prompt_template.partial(
            persona=self.full_system_prompt,
            format_instructions=self.parser.get_format_instructions()
        ) | self.llm | self.parser

    def decide(self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]) -> TradeDecision:
        """
        Overrides the decision logic to inject memory.
        """
        # 1. Standard Observation (Same as Static)
        input_data = f"""
        DATE: {market_state['date']}
        
        MARKET OBSERVATION:
        - Price: ${market_state['price']:.2f}
        - Trend: {market_state['SMA20']} / {market_state['SMA60']} ({market_state.get('trend_regime', 0)})
        - RSI: {market_state['RSI14']}
        - P/E Ratio: {market_state.get('reported_PE', 'N/A')}
        - Implied Volatility: {market_state.get('implied_volatility', 'N/A')}%
        - Volume Ratio: {market_state.get('volume_ratio', 'N/A')}
        - News Sentiment: {market_state.get('news_sentiment', 'Neutral')}
        
        PORTFOLIO:
        - Cash: ${portfolio_state['cash']:.2f}
        - Holdings: ${portfolio_state['holdings_value']:.2f}
        """

        # 2. THE PILLAR I FIX: "Context Refresh"
        # We inject the mandate explicitly at the END of the prompt.
        context_refresh = f"""
        *** ACTIVE MEMORY REFRESH ***
        Strictly adhere to your core mandate:
        {self.core_mandate}
        
        Evaluate this trade ONLY through the lens of this mandate.
        """

        last_error = None
        for attempt in range(3):
            try:
                return self.chain.invoke({
                    "input_data": input_data,
                    "context_refresh": context_refresh
                })
            except Exception as e:
                last_error = e
                print(f"[{self.get_uid()}] Parse error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
        return TradeDecision(action="HOLD", quantity=0.0, rationale=f"Error after 3 attempts: {str(last_error)}")
