import torch.nn as nn
import torch
from plugins.registry import register_encoder
from .factory import EncoderFactory

class Sender(nn.Module):
    def __init__(self, encoder_name: str, encoder_kwargs: dict, communication_dim: int):
        super().__init__()
        self.encoder = EncoderFactory.create(encoder_name, **encoder_kwargs)
        # Projection layer to map encoder output to communication dimension
        # (vocab_size * max_length) for discrete, or continuous vector dim.
        self.fc = nn.Linear(encoder_kwargs.get("hidden_dim", 256), communication_dim)

    def forward(self, observation):
        encoded = self.encoder(observation)
        logits_or_vector = self.fc(encoded)
        return logits_or_vector
