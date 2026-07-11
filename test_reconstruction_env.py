import os
import torch
from omegaconf import OmegaConf
from envs.reconstruction_env import ReconstructionGameEnv
from plugins.registry import DATASET_REGISTRY

# 1. Load the default configuration from yaml
config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
config = OmegaConf.load(config_path)

# 2. Load dataset using the registry
dataset_class = DATASET_REGISTRY.get(config.dataset.name)
dataset_manager = dataset_class(config.dataset)

# 3. Create environment
env = ReconstructionGameEnv(config, dataset_manager)

# 4. Reset the environment
obs, info = env.reset()

print("="*40)
print("Initial Observation Shape:", obs.shape)

# 5. Take a dummy step with a terrible reconstruction (all zeros)
bad_reconstruction = torch.zeros_like(obs)
next_obs, reward_bad, terminated, truncated, info_bad = env.step(bad_reconstruction)

print("="*40)
print("Taking a step with a BAD reconstruction (all zeros):")
print(f"Reward Shape: {reward_bad.shape}")
print(f"Mean Reward:  {reward_bad.mean():.4f}")
print(f"Mean Loss:    {info_bad['loss']:.4f}")

# 6. Take a step with a perfect reconstruction (exact copy of the new observation)
perfect_reconstruction = next_obs.clone()
_, reward_perfect, _, _, info_perfect = env.step(perfect_reconstruction)

print("="*40)
print("Taking a step with a PERFECT reconstruction:")
print(f"Mean Reward:  {reward_perfect.mean():.4f}")
print(f"Mean Loss:    {info_perfect['loss']:.4f}")
print("="*40)

env.close()
print("\nEnvironment reward test executed successfully!")
