from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Message:
    """Encapsulates the communication between Sender and Receiver."""
    content: torch.Tensor
    lengths: torch.Tensor
    log_probs: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    discrete_content: Optional[torch.Tensor] = None
    
    @property
    def batch_size(self):
        return self.content.size(0)
    
    @property
    def max_length(self):
        if self.content.dim() > 1:
            return self.content.size(1)
        return 1
