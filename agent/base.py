from abc import ABC, abstractmethod
from typing import Dict, Any
from agent.schemas import TradeDecision

class BaseAgent(ABC):
    """
    Abstract Base Class for all FinPersona Agents.
    Enforces a strict interface for the Simulation Runner.
    """
    def __init__(self, mbti_type: str):
        self.mbti_type = mbti_type
        # Default name, should be overwritten by child classes
        self.name = f"Unknown-{mbti_type}"

    @abstractmethod
    def decide(self, market_state: Dict[str, Any], portfolio_state: Dict[str, float]) -> TradeDecision:
        """
        The Core Policy Function.
        
        Args:
            market_state: Dict containing 'price', 'SMA20', 'news_sentiment', etc.
            portfolio_state: Dict containing 'cash', 'holdings_value'.
            
        Returns:
            TradeDecision: A Pydantic model with action, quantity, and rationale.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state (memory, history) for a fresh experiment.
        """
        pass

    def get_uid(self) -> str:
        """
        Returns a unique identifier for logging purposes.
        Example: 'ENTJ-StaticAgent'
        """
        return f"{self.mbti_type}-{self.__class__.__name__}"
