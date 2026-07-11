import torch
from collections import defaultdict
import numpy as np

class ProtocolAnalyzer:
    """Analyzes emergent properties of the communication protocol."""
    def __init__(self, vocabulary_size: int):
        self.vocabulary_size = vocabulary_size
        self.token_frequencies = torch.zeros(vocabulary_size)
        self.total_messages = 0
        self.total_tokens = 0
        self.message_lengths = []
        
    def track_batch(self, messages, lengths=None):
        """Updates statistics with a batch of messages."""
        if messages.dim() > 1:
            batch_size = messages.size(0)
            
            # Count token frequencies
            for i in range(self.vocabulary_size):
                self.token_frequencies[i] += (messages == i).sum().item()
                
            self.total_tokens += messages.numel()
            self.total_messages += batch_size
            
            if lengths is not None:
                self.message_lengths.extend(lengths.cpu().tolist())

    def get_statistics(self):
        """Returns computed statistics for diagnostics."""
        stats = {}
        
        # Dead symbols
        unused = (self.token_frequencies == 0).sum().item()
        stats["comm/dead_symbols"] = unused
        stats["comm/vocab_utilization"] = (self.vocabulary_size - unused) / self.vocabulary_size
        
        # Vocab Collapse (if 1 symbol is used for >90% of tokens)
        if self.total_tokens > 0:
            max_freq = self.token_frequencies.max().item()
            stats["comm/collapse_ratio"] = max_freq / self.total_tokens
            stats["comm/is_collapsed"] = 1.0 if stats["comm/collapse_ratio"] > 0.9 else 0.0
            
            # Entropy of vocabulary distribution
            probs = self.token_frequencies / self.total_tokens
            probs = probs[probs > 0]
            stats["comm/vocab_entropy"] = -(probs * torch.log(probs)).sum().item()
        else:
            stats["comm/collapse_ratio"] = 0.0
            stats["comm/is_collapsed"] = 0.0
            stats["comm/vocab_entropy"] = 0.0
            
        # Message Lengths
        if self.message_lengths:
            stats["comm/avg_msg_length"] = np.mean(self.message_lengths)
        else:
            stats["comm/avg_msg_length"] = 0.0
            
        return stats
        
    def reset(self):
        self.token_frequencies = torch.zeros(self.vocabulary_size)
        self.total_messages = 0
        self.total_tokens = 0
        self.message_lengths = []
