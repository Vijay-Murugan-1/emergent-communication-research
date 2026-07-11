import collections
import torch
from dataclasses import dataclass

@dataclass
class Transition:
    observation: torch.Tensor
    message: torch.Tensor
    reconstruction: torch.Tensor
    reward: float

class ReplayBuffer:
    """Experience replay buffer for storing emergent communication episodes."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, transition: Transition):
        self.buffer.append(transition)
        
    def sample(self, batch_size: int):
        import random
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)
