import torch
from torch import Tensor, nn


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size, stride, padding, bias) -> None:
        super().__init__()
        self.redisual_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.redisual_block(x)
        x = torch.add(x, identity)

        return x
