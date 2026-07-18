"""
Observation generation.
"""

import numpy as np
from .constants import SPEAKER, LISTENER, LEFT
from .state import EnvironmentState
from .communication import CommunicationChannel


def generate_observation(state: EnvironmentState, channel: CommunicationChannel, agent: str) -> np.int64 | None:
    """
    Generate an observation for the given agent.
    """
    if agent == SPEAKER:
        val = 0 if state.goal == LEFT else 1
        return np.int64(val)

    if agent == LISTENER:
        msg = channel.receive()
        val = msg if msg is not None else 0
        return np.int64(val)

    return None