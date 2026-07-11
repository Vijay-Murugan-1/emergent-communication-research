import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from datasets.manager import DatasetManager
from communication.protocol import CommunicationProtocol
from plugins.registry import REWARD_REGISTRY

class ReconstructionGameEnv(gym.Env):
    """
    A Gymnasium-compatible environment for the Reconstruction Game.
    This environment handles dataset sampling, computing rewards, and the 
    interaction between Sender and Receiver.
    """
    def __init__(self, config, dataset_manager: DatasetManager):
        super().__init__()
        self.config = config
        self.dataset_manager = dataset_manager
        self.dataloader = iter(self.dataset_manager.get_dataloader("train"))
        
        # Setup Spaces
        # Abstracting image vs text observation spaces
        # Note: Actual RL agents usually operate in PyTorch directly.
        # This gym wrapper provides standard interfaces.
        stats = self.dataset_manager.get_statistics()
        if "channels" in stats:  # Image
            ch, h, w = stats["channels"], stats["resolution"][0], stats["resolution"][1]
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(ch, h, w), dtype=np.float32)
        else:
            self.observation_space = spaces.Discrete(stats.get("vocab_size", 10000))
            
        self.action_space = spaces.Discrete(config.communication.vocab_size)
        
        # Initialize reward function
        self.reward_fn = REWARD_REGISTRY.build(
            config.reward.name if hasattr(config.reward, 'name') else "hybrid",
            alpha_reconstruction=config.reward.alpha_reconstruction,
            beta_compression=config.reward.beta_compression,
            gamma_diversity=config.reward.gamma_diversity,
            delta_efficiency=config.reward.delta_efficiency
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            batch = next(self.dataloader)
        except StopIteration:
            self.dataloader = iter(self.dataset_manager.get_dataloader("train"))
            batch = next(self.dataloader)
            
        # batch is usually (data, target)
        self.current_observation = batch[0]
        return self.current_observation, {}

    def step(self, actions):
        """
        Step expects actions (the predicted reconstructions from the Receiver).
        The environment computes the reward based on how close the reconstruction 
        is to the current_observation.
        """
        # actions here are the Reconstructed Objects from the Receiver
        reconstructed = actions
        
        # Compute Reconstruction Loss (MSE for images, CrossEntropy for text)
        loss_fn = torch.nn.MSELoss(reduction='none')
        
        # Assuming reconstructed and observation are tensors
        # Flatten for loss computation
        rec_flat = reconstructed.view(reconstructed.size(0), -1)
        obs_flat = self.current_observation.view(self.current_observation.size(0), -1)
        
        loss_reconstruction = loss_fn(rec_flat, obs_flat).mean(dim=1)
        
        # Using a dummy message for now to compute reward (length max_len)
        class DummyMessage:
            lengths = torch.full((reconstructed.size(0),), self.config.communication.max_length)
            entropy = None
            
        reward = self.reward_fn(
            loss_reconstruction, 
            DummyMessage(), 
            self.config.communication.max_length, 
            self.config.communication.vocab_size
        )
        
        terminated = True
        truncated = False
        
        info = {
            "loss": loss_reconstruction.mean().item(),
            "reward": reward.mean().item()
        }
        
        # Reset internally for continuous sampling
        next_obs, _ = self.reset()
        
        # For PettingZoo / standard RL, rewards and next obs are returned
        return next_obs, reward.detach().cpu().numpy(), terminated, truncated, info
