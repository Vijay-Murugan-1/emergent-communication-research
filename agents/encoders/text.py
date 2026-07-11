import torch.nn as nn
from plugins.registry import register_encoder

@register_encoder("lstm")
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # hidden is [num_layers * num_directions, batch, hidden_size]
        return hidden[-1]
