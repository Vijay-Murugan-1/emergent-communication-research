"""
PettingZoo Speaker-Listener Environment.
"""

from typing import Dict, List, Optional, Tuple, Any
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.spaces import Discrete

from .constants import (
    SPEAKER,
    LISTENER,
    LEFT,
    RIGHT,
    GOALS,
)
from .state import EnvironmentState
from .communication import CommunicationChannel
from .reward import compute_reward
from .logger import EnvironmentLogger
from .observation import generate_observation
from .spaces import (
    SPEAKER_ACTION_SPACE,
    LISTENER_ACTION_SPACE,
    SPEAKER_OBSERVATION_SPACE,
    LISTENER_OBSERVATION_SPACE,
)
from .renderer import render_env
from .game import SpeakerListenerGame


class SpeakerListenerEnv(AECEnv):
    """
    PettingZoo AEC Speaker-Listener Environment.
    
    A cooperative multi-agent environment where:
    - The speaker observes the hidden goal and sends a message symbol.
    - The listener receives the message symbol and chooses a direction.
    """

    metadata = {
        "name": "speaker_listener_v0",
        "render_modes": ["human"],
    }

    def __init__(self) -> None:
        super().__init__()

        self.possible_agents = [
            SPEAKER,
            LISTENER,
        ]
        self.agents = self.possible_agents[:]

        # Environment sub-components
        self.state = EnvironmentState()
        self.channel = CommunicationChannel()
        self.logger = EnvironmentLogger()
        self.game = SpeakerListenerGame(self.state, self.channel)

        # Action and Observation spaces mapping
        self._action_spaces = {
            SPEAKER: SPEAKER_ACTION_SPACE,
            LISTENER: LISTENER_ACTION_SPACE,
        }
        self._observation_spaces = {
            SPEAKER: SPEAKER_OBSERVATION_SPACE,
            LISTENER: LISTENER_OBSERVATION_SPACE,
        }

        # PettingZoo standard API attributes
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = SPEAKER

    def action_space(self, agent: str) -> Discrete:
        """
        Return the action space for the given agent.
        """
        return self._action_spaces[agent]

    def observation_space(self, agent: str) -> Discrete:
        """
        Return the observation space for the given agent.
        """
        return self._observation_spaces[agent]

    def observe(self, agent: str) -> int:
        """
        Return the current observation for the given agent.
        """
        obs = generate_observation(self.state, self.channel, agent)
        return obs if obs is not None else 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Reset the environment state and setup agent selector.
        """
        self.agents = self.possible_agents[:]
        self.state.reset(seed)
        self.channel.reset()
        self.logger.reset()

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: int) -> None:
        """
        Perform a step in the environment.
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection

        # Zero rewards for this step before calculating new ones
        self.rewards = {agent: 0 for agent in self.possible_agents}

        if current_agent == SPEAKER:
            # Speaker selects a message symbol (0 or 1)
            self.game.speaker_step(action)
            self.logger.log_message(action)

            # Clear speaker cumulative rewards
            self._cumulative_rewards[SPEAKER] = 0

            # Cycle selection
            self.state.current_agent = LISTENER
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()

        elif current_agent == LISTENER:
            # Listener performs direction action (0 -> LEFT, 1 -> RIGHT)
            listener_action = LEFT if action == 0 else RIGHT
            reward = self.game.listener_step(listener_action)

            # Cooperative game: same reward for speaker and listener
            self.rewards[SPEAKER] = reward
            self.rewards[LISTENER] = reward

            # Clear listener cumulative rewards
            self._cumulative_rewards[LISTENER] = 0

            # Increment step count
            self.state.step_count += 1

            # End of interaction round: terminate episode
            self.terminations[SPEAKER] = True
            self.terminations[LISTENER] = True

            # Log episode summary
            self.logger.log_episode({
                "goal": self.state.goal,
                "message": self.channel.receive(),
                "listener_action": listener_action,
                "reward": reward,
            })

            # Cycle selection
            self.state.current_agent = SPEAKER
            self.agent_selection = self._agent_selector.next()
            self._accumulate_rewards()

    def render(self) -> None:
        """
        Render environment state.
        """
        render_env(self.state, self.channel)

    def close(self) -> None:
        """
        Clean up resources.
        """
        pass