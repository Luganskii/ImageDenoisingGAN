import torch
from torch import Tensor, nn

from .basic_block import _Basic_block
from .residual_block import _ResidualConvBlock


class Generator(nn.Module):

    """

    """

    def __init__(self, channels: int, kernel_size: int, stride: int = 1, padding: int = 1, bias: bool = False, B: int = 16):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size * 3, stride, padding, bias=bias),
            nn.PReLU(),
        )

        self.base_block = self._make_layer(channels, kernel_size, stride, padding, bias, B)

        self.skip_connection_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(channels)
        )

        self.output_conv = nn.Conv2d(channels, 3, kernel_size * 3, stride, padding, bias=bias)

    def forward(self, x):
        identity = x
        x = self.input_block(x)
        x = self.base_block(x)
        x = self.skip_connection_block(x)
        x = torch.add(x, identity)

        return self.output_conv(x)

    def _make_layer(self, channels: int, kernel_size: int, stride: int, padding: int, bias: bool, B: int):
        layers = []

        for _ in range(B):
            layers.append(_ResidualConvBlock(channels, kernel_size, stride, padding, bias))

        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    """

    """

    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, bias: bool, alpha: float = 0.2, hidden_channel: int = 1024, output_channel: int = 1):
        super().__init__()

        # input: batch_size, 3, 96, 96
        self.input_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias=bias),
            # batch_size, channels, 96, 96
            nn.LeakyReLU(alpha),
            nn.Conv2d(channels, channels, kernel_size, stride * 2, padding, bias=bias),
            # batch_size, channels, 48, 48
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(alpha)
        )
        # batch_size, channels, 48, 48
        self.base_block = nn.Sequential(
            _Basic_block(channels, channels * 2, kernel_size, stride, padding, bias=bias, alpha=alpha),
            _Basic_block(channels * 2, channels * 2, kernel_size, stride * 2, padding, bias=bias, alpha=alpha),
            _Basic_block(channels * 2, channels * 4, kernel_size, stride, padding, bias=bias, alpha=alpha),
            _Basic_block(channels * 4, channels * 4, kernel_size, stride * 2, padding, bias=bias, alpha=alpha),
            _Basic_block(channels * 4, channels * 8, kernel_size, stride, padding, bias=bias, alpha=alpha),
            _Basic_block(channels * 8, channels * 8, kernel_size, stride * 2, padding, bias=bias, alpha=alpha)
        )
        # batch_size, channels, 48, 48
        self.output_block = nn.Sequential(
            # batch_size, channels * 48 * 48
            nn.Linear(channels * 48 * 48, hidden_channel),
            # batch_size, hidden_channel
            nn.LeakyReLU(alpha),
            nn.Linear(hidden_channel, output_channel)
            # batch_size, output_channel
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_block(x)
        x = self.base_block(x)
        x = x.view(x.shape[0], -1)

        return self.output_block(x)
