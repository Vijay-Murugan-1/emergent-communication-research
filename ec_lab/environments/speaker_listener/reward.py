"""
Reward computation.
"""

from .constants import REWARD_CORRECT, REWARD_INCORRECT


def compute_reward(goal: str, action: str) -> int:
    """
    Compute reward for the listener action.
    """
    if goal == action:
        return REWARD_CORRECT

    return REWARD_INCORRECT