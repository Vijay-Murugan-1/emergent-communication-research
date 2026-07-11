import torch.nn as nn
from plugins.registry import register_decoder
from .factory import DecoderFactory
from communication.message import Message

class Receiver(nn.Module):
    def __init__(self, decoder_name: str, decoder_kwargs: dict, communication_dim: int):
        super().__init__()
        hidden_dim = decoder_kwargs.get("hidden_dim", 256)
        self.fc = nn.Linear(communication_dim, hidden_dim)
        self.decoder = DecoderFactory.create(decoder_name, **decoder_kwargs)

    def forward(self, message: Message):
        # Flatten message content if necessary
        content = message.content
        if content.dim() > 2:
            content = content.view(content.size(0), -1)
            
        projected = self.fc(content.float())
        reconstructed = self.decoder(projected)
        return reconstructed
