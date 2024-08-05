from torch import Tensor, nn


class _Basic_block(nn.Module):
    """
    Basic module for discriminator
    """

    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, bias: bool, alpha: float) -> None:
        self.redisual_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.redisual_block(x)
