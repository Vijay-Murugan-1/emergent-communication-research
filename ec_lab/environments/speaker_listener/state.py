"""
State management for the Speaker-Listener environment.
"""

from dataclasses import dataclass
import random

from .constants import GOALS, SPEAKER


@dataclass
class EnvironmentState:
    """
    Stores the current environment state.
    """

    goal: str | None = None
    current_agent: str = SPEAKER
    current_message: int | None = None
    step_count: int = 0

    def reset(self, seed: int | None = None) -> None:
        """
        Reset the environment state.
        """
        rng = random.Random(seed)

        self.goal = rng.choice(GOALS)
        self.current_agent = SPEAKER
        self.current_message = None
        self.step_count = 0
        