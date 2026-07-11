import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from configs.schemas import ReconstructionGameConfig

class ReconstructionGameGymEnv(gym.Env):
    """
    Gymnasium environment for the Reconstruction Game.
    
    This acts as a centralized environment where a single policy (or a joint Sender-Receiver 
    network) interacts with the environment. 
    
    The environment provides the target vector as observation.
    The agent must provide a joint action: both the 'message' and the 'reconstruction'.
    This assumes the RL agent internally passes the sender's message to the receiver before
    taking the environment step.
    
    Purpose: Information Compression, Representation Learning.
    Measures: Reconstruction Accuracy, Communication Efficiency.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, config: Optional[ReconstructionGameConfig] = None, render_mode: str = None):
        super().__init__()
        if config is None:
            config = ReconstructionGameConfig()
            
        self.config = config
        self.render_mode = render_mode
        
        self.feature_dim = self.config.feature_dim
        self.vocab_size = self.config.vocab_size
        
        # Observation is the target vector to be communicated
        if self.config.continuous_target:
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.MultiBinary(self.feature_dim)
            
        # Action is a dictionary containing both the sender's message and receiver's reconstruction
        self.action_space = spaces.Dict({
            "message": spaces.Discrete(self.vocab_size),
            "reconstruction": spaces.Box(
                low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
            )
        })
        
        self.current_target = None
        self.current_step = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.current_target = self._generate_target()
        
        info = {}
        return self.current_target, info

    def _generate_target(self) -> np.ndarray:
        """Generates a new target vector."""
        if self.config.continuous_target:
            return self.np_random.uniform(-1.0, 1.0, size=(self.feature_dim,)).astype(np.float32)
        else:
            return self.np_random.integers(0, 2, size=(self.feature_dim,), dtype=np.int8)

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Takes a step in the environment.
        
        Args:
            action: A dictionary containing 'message' and 'reconstruction'.
            
        Returns:
            obs, reward, terminated, truncated, info
        """
        reconstruction = action["reconstruction"]
        message = action["message"]
        
        # Compute reward: Negative Mean Squared Error
        mse = np.mean(np.square(self.current_target - reconstruction))
        reward = -float(mse)
        
        # Logging metrics
        info = {
            "success": mse < 0.05,  # Arbitrary threshold for success
            "mse": mse,
            "message": int(message)
        }
        
        self.current_step += 1
        terminated = True  # Episodic: 1 step per target by default
        truncated = False
        
        # In a continuous running mode, we might not terminate immediately
        if self.config.max_steps > 1 and self.current_step < self.config.max_steps:
            terminated = False
            self.current_target = self._generate_target()
        
        return self.current_target, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Target: {self.current_target}")

    def close(self):
        pass
