import gymnasium as gym

class BaseReconstructionEnv(gym.Env):
    """Base class for Emergent Communication environments."""
    def __init__(self):
        super().__init__()
        
    def reset(self, seed=None, options=None):
        raise NotImplementedError
        
    def step(self, action):
        raise NotImplementedError
