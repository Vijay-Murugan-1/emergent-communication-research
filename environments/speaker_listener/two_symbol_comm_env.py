"""
Speaker-Listener communication environment for EC-Lab.

This environment simulates communication between two agents in a single Gymnasium Env:
- Speaker sends a message.
- Listener is simulated to act LEFT if message is 0, RIGHT if message is 1.
"""

import gymnasium as gym
from typing import Tuple, Dict, Any, Optional

from .constants import LEFT, RIGHT, SPEAKER
from .state import EnvironmentState
from .communication import CommunicationChannel
from .reward import compute_reward
from .logger import EnvironmentLogger
from .observation import generate_observation
from .spaces import SPEAKER_ACTION_SPACE, SPEAKER_OBSERVATION_SPACE


class TwoSymbolCommEnv(gym.Env):
    """
    A simple Gymnasium wrapper for the two-symbol Speaker-Listener task.
    In this environment, the listener action is hardcoded to map message 0 to LEFT
    and message 1 to RIGHT.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = SPEAKER_ACTION_SPACE
        self.observation_space = SPEAKER_OBSERVATION_SPACE

        self.state = EnvironmentState()
        self.channel = CommunicationChannel()
        self.logger = EnvironmentLogger()

    @property
    def goal(self) -> Optional[str]:
        """
        Expose the current hidden goal.
        """
        return self.state.goal

    @property
    def message_log(self) -> list:
        """
        Expose the logged messages.
        """
        return self.logger.message_log

    @property
    def episode_log(self) -> list:
        """
        Expose the logged episodes.
        """
        return self.logger.episode_log

    @property
    def metrics_log(self) -> list:
        """
        Dummy property to preserve compatibility with old evaluation metrics.
        """
        return [{"success": log["reward"] > 0, "reward": log["reward"]} for log in self.logger.episode_log]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict[str, Any]]:
        super().reset(seed=seed)
        self.state.reset(seed)
        self.channel.reset()
        self.logger.reset()

        obs = generate_observation(self.state, self.channel, SPEAKER)
        obs_val = obs if obs is not None else 0
        return obs_val, {}

    def step(self, action: int) -> Tuple[int, int, bool, bool, Dict[str, Any]]:
        # Speaker action is the message
        self.channel.transmit(action)
        self.logger.log_message(action)

        # Receiver maps message 0 to LEFT, 1 to RIGHT
        receiver_action = LEFT if action == 0 else RIGHT
        reward = compute_reward(self.state.goal, receiver_action)

        terminated = True
        truncated = False

        obs = generate_observation(self.state, self.channel, SPEAKER)
        obs_val = obs if obs is not None else 0

        self.logger.log_episode({
            "goal": self.state.goal,
            "message": action,
            "receiver_action": receiver_action,
            "reward": reward
        })

        return obs_val, reward, terminated, truncated, {}

    def render(self) -> None:
        print(f"Goal: {self.state.goal}")

    def close(self) -> None:
        pass