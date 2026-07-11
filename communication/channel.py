import torch
import torch.nn as nn
from plugins.registry import register_channel
from .message import Message

class CommunicationChannel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, sender_output: torch.Tensor) -> Message:
        """Processes sender output and transmits it over the channel."""
        raise NotImplementedError

@register_channel("discrete")
class DiscreteChannel(CommunicationChannel):
    """A standard discrete channel that expects categorical distributions or logits."""
    def forward(self, sender_output: torch.Tensor) -> Message:
        # Placeholder for complex sampling (e.g. Gumbel or STE logic)
        # This will be orchestrated by the protocol
        pass

@register_channel("continuous")
class ContinuousChannel(CommunicationChannel):
    """A channel for transmitting continuous latent vectors."""
    def forward(self, sender_output: torch.Tensor) -> Message:
        batch_size = sender_output.size(0)
        lengths = torch.full((batch_size,), sender_output.size(1) if sender_output.dim() > 1 else 1, device=sender_output.device)
        return Message(content=sender_output, lengths=lengths)

@register_channel("gumbel")
class GumbelSoftmaxChannel(CommunicationChannel):
    """A channel for discrete symbols using the Gumbel-Softmax estimator."""
    def __init__(self, config):
        super().__init__(config)
        self.temperature = getattr(config, "temperature", 1.0)
        
    def forward(self, sender_output: torch.Tensor) -> Message:
        # sender_output is raw logits [batch_size, max_length, vocab_size]
        batch_size = sender_output.size(0)
        
        # Apply Gumbel-Softmax (hard=True for straight-through estimator)
        if self.training:
            symbols_one_hot = torch.nn.functional.gumbel_softmax(
                sender_output, tau=self.temperature, hard=True, dim=-1
            )
        else:
            # During evaluation, just take argmax
            indices = torch.argmax(sender_output, dim=-1)
            symbols_one_hot = torch.nn.functional.one_hot(indices, num_classes=sender_output.size(-1)).float()
            
        # For compatibility with entropy/diagnostics, calculate standard probs
        probs = torch.softmax(sender_output, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        entropy = dist.entropy()
        
        # The content here is the continuous one-hot vector (allowing gradients to flow)
        # We also store the discrete indices for diagnostics
        discrete_indices = torch.argmax(symbols_one_hot, dim=-1)
        
        lengths = torch.full((batch_size,), sender_output.size(1) if sender_output.dim() > 2 else 1, device=sender_output.device)
        
        # We pass the one_hot continuous representation as content so backprop works
        return Message(
            content=symbols_one_hot, 
            lengths=lengths,
            entropy=entropy,
            discrete_content=discrete_indices # Added for diagnostics
        )

@register_channel("reinforce")
class ReinforceChannel(CommunicationChannel):
    """Samples discrete symbols and stores log probabilities for Policy Gradient (REINFORCE)."""
    def forward(self, sender_output: torch.Tensor) -> Message:
        # sender_output is expected to be raw logits [batch_size, max_length, vocab_size]
        # or [batch_size, vocab_size] for single-step communication
        
        # Calculate probabilities
        probs = torch.softmax(sender_output, dim=-1)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs=probs)
        
        # Sample symbols
        sampled_symbols = dist.sample()
        
        # Calculate log probabilities of the sampled symbols
        log_probs = dist.log_prob(sampled_symbols)
        
        # Calculate entropy for diversity bonuses/regularization
        entropy = dist.entropy()
        
        batch_size = sender_output.size(0)
        lengths = torch.full((batch_size,), sender_output.size(1) if sender_output.dim() > 2 else 1, device=sender_output.device)
        
        return Message(
            content=sampled_symbols, 
            lengths=lengths,
            log_probs=log_probs,
            entropy=entropy
        )
