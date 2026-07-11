import torch

class EOSProcessor:
    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        
    def find_lengths(self, symbols: torch.Tensor, max_len: int) -> torch.Tensor:
        """Finds the length of each sequence up to the first EOS token."""
        is_eos = (symbols == self.eos_id)
        # argmax returns the first index of the maximum value
        # If EOS is present, it returns its index. If not, it returns 0.
        eos_indices = is_eos.float().argmax(dim=-1)
        
        # If no EOS is in the sequence, the length is the full max length
        no_eos = ~is_eos.any(dim=-1)
        lengths = eos_indices + 1
        lengths[no_eos] = max_len
        return lengths
