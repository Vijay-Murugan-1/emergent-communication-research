import torch
import torch.nn.functional as F

class CommunicationNoise:
    def __init__(self, noise_prob: float):
        self.noise_prob = noise_prob
        
    def apply_discrete_noise(self, symbols: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Randomly flips symbols with noise_prob."""
        if self.noise_prob <= 0.0:
            return symbols
            
        mask = torch.rand_like(symbols, dtype=torch.float) < self.noise_prob
        random_symbols = torch.randint_like(symbols, 0, vocab_size)
        return torch.where(mask, random_symbols, symbols)

    def apply_continuous_noise(self, vectors: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Adds Gaussian noise to continuous vectors."""
        if self.noise_prob <= 0.0:
            return vectors
            
        noise = torch.randn_like(vectors) * std
        return vectors + noise
