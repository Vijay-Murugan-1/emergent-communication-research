"""
Constants for the Speaker-Listener environment.
"""

# Agents
SPEAKER = "speaker"
LISTENER = "listener"

# Goals
LEFT = "LEFT"
RIGHT = "RIGHT"

GOALS = (LEFT, RIGHT)

# Spaces
NUM_MESSAGES = 2
NUM_ACTIONS = 2
NUM_GOALS = 2

# Rewards
REWARD_CORRECT = 10
REWARD_INCORRECT = -10

# Episode
MAX_STEPS = 2