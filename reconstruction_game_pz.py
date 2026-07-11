import functools
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from typing import Optional, Dict, Any

from configs.schemas import ReconstructionGameConfig

def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    """
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
    """
    PettingZoo AEC environment for the Reconstruction Game.
    
    Agents:
    - sender_0: Observes the target vector and emits a discrete symbol (message).
    - receiver_0: Observes the message and reconstructs the target vector.
    
    This is an episodic turn-based game:
    1. sender_0 gets target, outputs message.
    2. receiver_0 gets message, outputs reconstruction.
    3. Reward is computed and episode ends.
    """
    
    metadata = {
        "render_modes": ["human"], 
        "name": "reconstruction_game_v0",
        "is_parallelizable": False
    }

    def __init__(self, config: Optional[ReconstructionGameConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        if config is None:
            config = ReconstructionGameConfig()
            
        self.config = config
        self.render_mode = render_mode
        
        self.feature_dim = self.config.feature_dim
        self.vocab_size = self.config.vocab_size
        
        self.agents = ["sender_0", "receiver_0"]
        self.possible_agents = self.agents[:]
        
        # Action spaces
        self.action_spaces = {
            "sender_0": spaces.Discrete(self.vocab_size),
            "receiver_0": spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)
        }
        
        # Observation spaces
        # Sender observes target, Receiver observes message
        if self.config.continuous_target:
            sender_obs = spaces.Box(low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)
        else:
            sender_obs = spaces.MultiBinary(self.feature_dim)
            
        self.observation_spaces = {
            "sender_0": sender_obs,
            "receiver_0": spaces.Discrete(self.vocab_size)
        }

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = None
        
        self.current_target = None
        self.current_message = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
        
    def _generate_target(self) -> np.ndarray:
        if self.config.continuous_target:
            # We use gym.spaces.Box sample or just numpy
            # Seed behavior should ideally rely on self.np_random but keeping simple here
            return np.random.uniform(-1.0, 1.0, size=(self.feature_dim,)).astype(np.float32)
        else:
            return np.random.integers(0, 2, size=(self.feature_dim,), dtype=np.int8)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.current_target = self._generate_target()
        self.current_message = 0 # Default message before sender acts

    def observe(self, agent: str):
        """Returns the observation for the given agent."""
        if agent == "sender_0":
            return self.current_target
        elif agent == "receiver_0":
            return self.current_message

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        self._cumulative_rewards[agent] = 0

        if agent == "sender_0":
            # Sender emits a message
            self.current_message = int(action)
            
            # Move to receiver
            self.agent_selection = self._agent_selector.next()
            
        elif agent == "receiver_0":
            # Receiver attempts reconstruction
            reconstruction = action
            
            # Compute reward
            mse = np.mean(np.square(self.current_target - reconstruction))
            reward = -float(mse)
            
            # Assign collaborative reward
            self.rewards["sender_0"] = reward
            self.rewards["receiver_0"] = reward
            
            self.infos["sender_0"]["mse"] = mse
            self.infos["receiver_0"]["mse"] = mse
            
            # End episode
            self.terminations = {a: True for a in self.agents}
            self.agent_selection = self._agent_selector.next()
            
        self._accumulate_rewards()

    def render(self):
        if self.render_mode == "human":
            print(f"Target: {self.current_target} | Message: {self.current_message}")

    def close(self):
        pass
