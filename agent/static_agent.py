import os
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# LLM Providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Internal Imports
from agent.base import BaseAgent
from agent.schemas import TradeDecision
from agent.prompts import get_financial_persona

load_dotenv()

class StaticAgent(BaseAgent):
    """
    Baseline Agent (The 'Old Way').
    Multi-Model Support Added: Gemini, Claude, GPT.
    """

    def __init__(self, mbti_type: str, model_name: str = "gemini-2.0-flash"):
        super().__init__(mbti_type)
        self.model_name = model_name
        
        # 1. Load the specific financial extension
        self._load_persona_data()
        
        # 2. Setup the LLM (Now Dynamic)
        self._setup_llm()
        
        # 3. Setup Chain
        self._setup_chain()

    def _load_persona_data(self):
        json_path = os.path.join(os.path.dirname(__file__), "personas", "mbti_profiles.json")
        try:
            with open(json_path, 'r') as f:
                profiles = json.load(f)
        except FileNotFoundError:
            profiles = {}
        
        profile = profiles.get(self.mbti_type, profiles.get("DEFAULT", {}))
        self.financial_extension = profile.get("financial_extension", "")
        self.full_system_prompt = get_financial_persona(self.mbti_type, self.financial_extension)

    def _setup_llm(self):
        """
        Factory method to initialize the correct LLM based on model_name.
        """
        temperature = 0.2
        
        if "gemini" in self.model_name.lower():
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key: raise ValueError("GOOGLE_API_KEY missing.")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name, temperature=temperature, google_api_key=api_key
            )

        elif self.model_name.startswith("openrouter/"):
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key: raise ValueError("OPENROUTER_API_KEY missing.")
            # Strip the "openrouter/" prefix to get the actual model id
            actual_model = self.model_name[len("openrouter/"):]
            self.llm = ChatOpenAI(
                model=actual_model, temperature=temperature, api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        elif "claude" in self.model_name.lower():
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key: raise ValueError("ANTHROPIC_API_KEY missing.")
            self.llm = ChatAnthropic(
                model=self.model_name, temperature=temperature, anthropic_api_key=api_key
            )
            
        elif "gpt" in self.model_name.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key: raise ValueError("OPENAI_API_KEY missing.")
            self.llm = ChatOpenAI(
                model=self.model_name, temperature=temperature, api_key=api_key
            )
        elif "deepseek" in self.model_name.lower():
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key: raise ValueError("DEEPSEEK_API_KEY missing.")
            self.llm = ChatOpenAI(
                model=self.model_name, temperature=temperature, api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _setup_chain(self):
        self.parser = PydanticOutputParser(pydantic_object=TradeDecision)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "{persona}"),
            ("human", "{input_data}\n\nIMPORTANT: You must return a valid JSON object with ALL 3 fields: 'action', 'quantity' (0.0 if HOLD), and 'rationale'.\n\n{format_instructions}")
        ])
        self.chain = self.prompt_template.partial(
            persona=self.full_system_prompt,
            format_instructions=self.parser.get_format_instructions()
        ) | self.llm | self.parser

    def decide(self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]) -> TradeDecision:
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
                return self.chain.invoke({"input_data": input_data})
            except Exception as e:
                last_error = e
                print(f"[{self.get_uid()}] Parse error (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(1)
        return TradeDecision(action="HOLD", quantity=0.0, rationale=f"Error after 3 attempts: {str(last_error)}")

    def reset(self):
        pass
