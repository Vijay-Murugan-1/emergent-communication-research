import torch
import torch.nn as nn
from plugins.registry import register_decoder

@register_decoder("cnn")
class CNNDecoder(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int, output_size: int = 28):
        super().__init__()
        self.output_size = output_size
        self.fc = nn.Linear(hidden_dim, 64 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 4, 4)
        out = self.net(x)
        # Interpolate to match exact output resolution
        out = nn.functional.interpolate(out, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        return out
