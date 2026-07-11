import torch
import torch.nn as nn
from plugins.registry import register_decoder

@register_decoder("lstm")
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, max_length: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_state):
        batch_size = hidden_state.size(0)
        
        # Prepare inputs for generation
        inputs = hidden_state.unsqueeze(1).expand(-1, self.max_length, -1)
        
        # h_0, c_0
        h_0 = hidden_state.unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        outputs, _ = self.lstm(inputs, (h_0, c_0))
        logits = self.fc(outputs)
        
        return logits
