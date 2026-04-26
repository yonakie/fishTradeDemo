"""Debate layer — bull / bear / judge."""

from .bear import debate_opening_bear_node, debate_rebuttal_bear_node
from .bull import debate_opening_bull_node, debate_rebuttal_bull_node
from .judge import debate_judge_node

__all__ = [
    "debate_judge_node",
    "debate_opening_bear_node",
    "debate_opening_bull_node",
    "debate_rebuttal_bear_node",
    "debate_rebuttal_bull_node",
]
