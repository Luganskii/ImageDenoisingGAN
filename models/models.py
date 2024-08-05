from basic_block import _Basic_block
from torch import Tensor, nn


class Generator(nn.Module):
    pass


class Discriminator(nn.Module):
    """

    """

    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, bias: bool, alpha: float = 0.2, hidden_channel: int = 1024, output_channel: int = 1):

        self.input_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias),
            nn.LeakyReLU(alpha),
            nn.Conv2d(channels, channels, kernel_size, stride * 2, padding, bias),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(alpha)
        )

        self.base_block = self._make_layer(channels, kernel_size, stride, padding, bias, alpha)

        self.output_block = nn.Sequential(
            nn.Linear(..., hidden_channel),
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_channel, output_channel)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        x = self.base_block(x)

        return self.output_block(x)

    def _make_layer(self, channels: int, kernel_size: int, stride: int, padding: int, bias: bool, alpha: float = 0.2):
        layers = []

        for i in range(6):
            k = 2 ** (i // 2 + 1)
            layers.append(_Basic_block(k * channels, kernel_size, stride, padding, bias, alpha))

        return nn.Sequential(layers)
