""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Sequential, Module, Parameter, SyncBatchNorm, LeakyReLU, UpsamplingNearest2d
from torch import Tensor

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, rezero=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            StackedBottleNeck(mid_channels, out_channels, rezero)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, rezero):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 2, 1),
            StackedBottleNeck(out_channels, out_channels, rezero),
        )

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, rezero):
        super().__init__()

        self.up = UpsamplingNearest2d(scale_factor=2)
        self.conv = Sequential(
            StackedBottleNeck(in_channels, out_channels, rezero),
        )
        self.out = StackedBottleNeck(out_channels*2, out_channels, rezero)

    def forward(self, x1, x2):
        # x1 (h, w)    x2 (2h, 2w)
        x1 = self.up(x1)
        # print(f'up_x1shape={x1.shape}\nup.x2.shape={x2.shape}')
        x1 = self.conv(x1)
        # print(f'up_x1shape={x1.shape}\nup.x2.shape={x2.shape}')
        x = torch.cat([x2, x1], dim=1)
        # print(f'up_x.shape={x.shape}')
        return self.out(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class StackedBottleNeck(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            rezero: bool,
    ) -> None:
        super(StackedBottleNeck, self).__init__()

        self.block = Sequential(
            BottleNeck(in_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
            BottleNeck(out_channels, out_channels, rezero),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.block(x)
        return x1
    

class BottleNeck(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rezero: bool,
    ) -> None:
        super(BottleNeck, self).__init__()
        self.rezero = rezero
        mid_channels = int(in_channels / 4.0)
        if self.rezero:
            self.left = Sequential(
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),   # 1*1卷积
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),  # 3*3卷积
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                SyncBatchNorm(in_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channels, mid_channels, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channels, 1, 1),
            )

        if in_channels == out_channels:
            self.right = None
        else:
            self.right = Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        
        x1 = factor * self.left(x) + (x
                                      if self.right is None else self.right(x))
        return x1