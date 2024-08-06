from torch import Tensor, nn


class _Basic_block(nn.Module):
    """
    Basic module for discriminator
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, bias: bool, alpha: float) -> None:
        super().__init__()
        self.redisual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.redisual_block(x)
