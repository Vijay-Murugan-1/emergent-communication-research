import torch
import torch.nn as nn
from plugins.registry import register_reward

@register_reward("hybrid")
class CompositeReward(nn.Module):
    def __init__(
        self, 
        alpha_reconstruction: float = 1.0, 
        beta_compression: float = 0.1,
        gamma_diversity: float = 0.1,
        delta_efficiency: float = 0.0,
        penalties: float = 0.5
    ):
        super().__init__()
        self.alpha = alpha_reconstruction
        self.beta = beta_compression
        self.gamma = gamma_diversity
        self.delta = delta_efficiency
        self.penalty_weight = penalties

    def forward(self, loss_reconstruction, message, max_length: int, vocab_size: int):
        batch_size = loss_reconstruction.size(0)
        
        # 1. Reconstruction Reward (Normalized, higher is better)
        reward_reconstruction = -loss_reconstruction 
        
        # 2. Compression Reward (Shorter messages are better)
        lengths = message.lengths.float()
        reward_compression = 1.0 - (lengths / max_length)
        
        # 3. Diversity Reward (Entropy of the message, if discrete)
        reward_diversity = torch.zeros_like(reward_reconstruction)
        if message.entropy is not None:
            ent = message.entropy
            if ent.dim() > 1:
                ent = ent.mean(dim=1)
            reward_diversity = ent
            
        # 4. Efficiency
        reward_efficiency = torch.zeros_like(reward_reconstruction)
        
        # 5. Penalties
        penalty_invalid = torch.zeros_like(reward_reconstruction) # E.g., for out-of-bounds symbols
        penalty_length = (lengths > max_length).float()
        total_penalties = self.penalty_weight * (penalty_invalid + penalty_length)
        
        # Total Reward
        total_reward = (
            self.alpha * reward_reconstruction +
            self.beta * reward_compression +
            self.gamma * reward_diversity +
            self.delta * reward_efficiency -
            total_penalties
        )
        
        return {
            "reward/total": total_reward,
            "reward/reconstruction": self.alpha * reward_reconstruction,
            "reward/compression": self.beta * reward_compression,
            "reward/diversity": self.gamma * reward_diversity,
            "reward/efficiency": self.delta * reward_efficiency,
            "penalty/total": total_penalties
        }
