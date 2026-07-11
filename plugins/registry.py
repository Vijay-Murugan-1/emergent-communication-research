from typing import Callable, Dict, Any

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Callable] = {}

    def register(self, name: str):
        """Decorator to register a class or function."""
        def decorator(cls_or_fn: Callable):
            if name in self._registry:
                raise ValueError(f"'{name}' is already registered in {self.name}.")
            self._registry[name] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def build(self, name: str, **kwargs) -> Any:
        cls_or_fn = self.get(name)
        return cls_or_fn(**kwargs)

# Core Registries
ENCODER_REGISTRY = Registry("Encoders")
DECODER_REGISTRY = Registry("Decoders")
REWARD_REGISTRY = Registry("Rewards")
DATASET_REGISTRY = Registry("Datasets")
CHANNEL_REGISTRY = Registry("Channels")

# Public Decorators
def register_encoder(name: str): return ENCODER_REGISTRY.register(name)
def register_decoder(name: str): return DECODER_REGISTRY.register(name)
def register_reward(name: str): return REWARD_REGISTRY.register(name)
def register_dataset(name: str): return DATASET_REGISTRY.register(name)
def register_channel(name: str): return CHANNEL_REGISTRY.register(name)
