import torch

class PaddingProcessor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        
    def mask_padding(self, symbols: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Masks out symbols beyond the specified lengths with pad_id."""
        batch_size, max_len = symbols.size(0), symbols.size(1)
        
        # Create a mask where True indicates a valid position (index < length)
        arange = torch.arange(max_len, device=symbols.device).unsqueeze(0).expand(batch_size, max_len)
        mask = arange < lengths.unsqueeze(1)
        
        # Replace invalid positions with pad_id
        padded_symbols = symbols.clone()
        padded_symbols[~mask] = self.pad_id
        
        return padded_symbols
