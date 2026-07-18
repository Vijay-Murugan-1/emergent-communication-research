"""
Gymnasium space definitions.
"""

from gymnasium.spaces import Discrete

from .constants import NUM_ACTIONS, NUM_GOALS, NUM_MESSAGES


SPEAKER_ACTION_SPACE = Discrete(NUM_MESSAGES)

LISTENER_ACTION_SPACE = Discrete(NUM_ACTIONS)

SPEAKER_OBSERVATION_SPACE = Discrete(NUM_GOALS)

LISTENER_OBSERVATION_SPACE = Discrete(NUM_MESSAGES)