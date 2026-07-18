"""
Configuration schemas for the Speaker-Listener environment.
"""

from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    """
    Configuration values for the environment.
    """

    max_steps: int = 2
    num_messages: int = 2
    num_actions: int = 2
    reward_correct: int = 10
    reward_incorrect: int = -10