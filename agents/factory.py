from plugins.registry import ENCODER_REGISTRY, DECODER_REGISTRY
import torch.nn as nn

class EncoderFactory:
    @staticmethod
    def create(name: str, **kwargs) -> nn.Module:
        return ENCODER_REGISTRY.build(name, **kwargs)

class DecoderFactory:
    @staticmethod
    def create(name: str, **kwargs) -> nn.Module:
        return DECODER_REGISTRY.build(name, **kwargs)
