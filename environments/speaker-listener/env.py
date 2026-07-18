"""
PettingZoo Speaker-Listener Environment.
"""

from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
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

class SpeakerListenerEnv(AECEnv):
    """
    PettingZoo AEC Speaker-Listener Environment.
    """

    metadata = {
        "name": "speaker_listener_v0",
        "render_modes": ["human"],
    }

    def __init__(self):
        super().__init__()

        self.possible_agents = [
            SPEAKER,
            LISTENER,
        ]

        self.agents = self.possible_agents[:]

        self._action_spaces = {
            SPEAKER: Discrete(2),
            LISTENER: Discrete(2),
        }

        self._observation_spaces = {
            SPEAKER: Discrete(2),
            LISTENER: Discrete(2),
        }

        self.state = EnvironmentState()
        self.channel = CommunicationChannel()
        self.logger = EnvironmentLogger()

             self.rewards = {
            agent: 0 for agent in self.possible_agents
        }

        self._cumulative_rewards = {
            agent: 0 for agent in self.possible_agents
        }

        self.terminations = {
            agent: False for agent in self.possible_agents
        }

        self.truncations = {
            agent: False for agent in self.possible_agents
        }

        self.infos = {
            agent: {} for agent in self.possible_agents
        }

        self.agent_selection = SPEAKER   
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        """
        self.agents = self.possible_agents[:]

        self.state.reset(seed)

        self.channel.reset()

        self.logger.reset()

        self.rewards = {
            agent: 0 for agent in self.possible_agents
        }

        self._cumulative_rewards = {
            agent: 0 for agent in self.possible_agents
        }

        self.terminations = {
            agent: False for agent in self.possible_agents
        }

        self.truncations = {
            agent: False for agent in self.possible_agents
        }

        self.infos = {
            agent: {} for agent in self.possible_agents
        }

        self.agent_selection = SPEAKER

    def observe(self, agent):
        """
        Return the observation for the given agent.
        """

        if agent == SPEAKER:
            return 0 if self.state.goal == LEFT else 1

        if agent == LISTENER:
            message = self.channel.receive()

            if message is None:
                return 0

            return message

        raise ValueError(f"Unknown agent: {agent}")
    
    def step(self, action):
        """
        Execute one environment step.
        """

        if self.agent_selection == SPEAKER:
            self.channel.transmit(action)
            self.logger.log_message(action)

            self.agent_selection = LISTENER
            return

        listener_action = LEFT if action == 0 else RIGHT

        reward = compute_reward(
            self.state.goal,
            listener_action,
        )

        self.rewards[SPEAKER] = reward
        self.rewards[LISTENER] = reward
    