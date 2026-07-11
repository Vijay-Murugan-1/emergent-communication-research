import torch
import torch.nn as nn
from .message import Message
from .channel import CommunicationChannel
from .constraints import CommunicationConstraints
from .noise import CommunicationNoise
from .eos import EOSProcessor
from .padding import PaddingProcessor

class CommunicationProtocol(nn.Module):
    """
    First-class object orchestrating the entire communication pipeline:
    Channel -> Constraints -> Noise -> EOS detection -> Padding.
    """
    def __init__(
        self, 
        channel: CommunicationChannel,
        vocab_size: int,
        max_length: int,
        noise_prob: float = 0.0,
        eos_id: int = 0,
        pad_id: int = 0
    ):
        super().__init__()
        self.channel = channel
        self.constraints = CommunicationConstraints(vocab_size, max_length)
        self.noise = CommunicationNoise(noise_prob)
        self.eos_processor = EOSProcessor(eos_id)
        self.padding = PaddingProcessor(pad_id)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.is_continuous = "Continuous" in channel.__class__.__name__

    def forward(self, sender_output: torch.Tensor) -> Message:
        # 1. Transmit through channel (sampling, Gumbel, continuous, etc.)
        message = self.channel(sender_output)
        
        # 2. Apply Constraints (Only for pure discrete indices, not one-hot or continuous)
        # We skip these for now since the channel implementations natively handle lengths and vocab bounds
        # (e.g. Reinforce samples from Categorical(vocab), Gumbel uses one-hot)
        
        # 3. Apply Noise
        if not self.is_continuous:
            # message.content = self.noise.apply_discrete_noise(message.content, self.vocab_size)
            pass # Skip noise for now to get baselines running cleanly
        else:
            message.content = self.noise.apply_continuous_noise(message.content)
            
        # 4. Resolve Lengths and Pad (only for discrete sequence communication)
        # Skip for continuous and Gumbel
        if not self.is_continuous and message.content.dtype == torch.long:
            lengths = self.eos_processor.find_lengths(message.content, self.max_length)
            message.lengths = lengths
            message.content = self.padding.mask_padding(message.content, lengths)
            
        return message
