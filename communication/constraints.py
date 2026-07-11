import torch

class CommunicationConstraints:
    def __init__(self, vocab_size: int, max_length: int):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def enforce_length(self, symbols: torch.Tensor) -> torch.Tensor:
        """Truncates symbols to max_length."""
        if symbols.dim() > 1 and symbols.size(1) > self.max_length:
            return symbols[:, :self.max_length]
        return symbols

    def enforce_vocab(self, symbols: torch.Tensor) -> torch.Tensor:
        """Clips symbols to valid vocabulary indices (if discrete)."""
        if symbols.dtype in [torch.int, torch.long]:
            return torch.clamp(symbols, 0, self.vocab_size - 1)
        return symbols
