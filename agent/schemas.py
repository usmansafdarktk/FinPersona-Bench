"""
Schemas Module

This module defines the strict data structures (schemas) for the agent's output.
It acts as a contract/interface between the LLM and the execution engine.
"""

from pydantic import BaseModel, Field
from typing import Literal


class TradeDecision(BaseModel):
    """
    Represents a single trading decision made by an agent.

    The LLM must populate this structure exactly.
    """

    action: Literal["BUY", "SELL", "HOLD"] = Field(
        ...,
        description="The action to take. BUY to enter/add, SELL to exit/reduce, "
        "HOLD to do nothing.",
    )

    quantity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The quantity to trade expressed as a percentage (0.0 to 1.0). "
        "For BUY: % of available cash. For SELL: % of current holdings. "
        "REQUIRED: If action is HOLD, you MUST set this to 0.0.",
    )

    rationale: str = Field(
        ...,
        description="A concise explanation (max 2-3 sentences) linking the "
        "decision to the agent's personality and the current market indicators. "
        "REQUIRED: You MUST provide a reason even if the decision is HOLD.",
    )
