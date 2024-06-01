from torch import Tensor
from torch.nn import Conv2d, Sequential, Module, Parameter, SyncBatchNorm, LeakyReLU
import torch
from torchvision.ops import StochasticDepth
import torch.nn as nn

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        s, u = torch.std_mean(x, dim=1, keepdim=True)
        x = (x - u) / (s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x



class NormLayer(nn.Module):
    def __init__(self, normalized_shape, norm_type):
        super(NormLayer, self).__init__()
        self.norm_type = norm_type

        if self.norm_type == 'LN':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'BN':
            self.norm = nn.BatchNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'RZ':
            self.norm = nn.Identity()
        elif self.norm_type in ['CNX', 'CN+X', 'GRN']:
            self.norm = LayerNorm(normalized_shape, affine=False)
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        elif self.norm_type == 'NX':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'CX':
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        else:
            raise ValueError('norm_type error')

    def forward(self, x):
        # save_featuremaps(self.conv(self.norm(x)), '/home/WangHT/python/G2-V2/featuremaps/w.png')
        # save_featuremaps(x, '/home/WangHT/python/G2-V2/featuremaps/before.png')
        if self.norm_type in ['LN', 'BN', 'IN', 'RZ']:
            x = self.norm(x)
        elif self.norm_type in ['CNX', 'GRN']:
            x = self.conv(self.norm(x)) * x
        elif self.norm_type == 'CN+X':
            x = self.conv(self.norm(x)) + x
        elif self.norm_type == 'NX':
            x = self.norm(x) * x
        elif self.norm_type == 'CX':
            x = self.conv(x) * x
        else:
            raise ValueError('norm_type error')
        # save_featuremaps(x, '/home/WangHT/python/G2-V2/featuremaps/after.png')
        return x


class CNBlock(nn.Module):
    def __init__(self, dim: int, norm_type: str, dp_rate: float):
        super(CNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            NormLayer(dim, norm_type),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            GRN(4 * dim) if norm_type == 'GRN' else nn.Identity(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = StochasticDepth(dp_rate, mode='batch')
        self.norm_type = norm_type
        if self.norm_type == 'RZ':
            self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        res = self.block(x)
        if self.norm_type == 'RZ':
            res = self.alpha * res
        x = x + self.drop_path(res)
        return x




class ResNeXtBottleneck(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            rezero: bool,
    ):
        super(ResNeXtBottleneck, self).__init__()
        self.rezero = rezero
        mid_channels = int(in_channel / 2)

        if self.rezero:
            self.left = Sequential(
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channel, mid_channels, 1, 1),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=32),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channel, 1, 1),
            )
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.left = Sequential(
                SyncBatchNorm(in_channel),
                LeakyReLU(0.2, inplace=True),
                Conv2d(in_channel, mid_channels, 1, 1),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=32),
                SyncBatchNorm(mid_channels),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, out_channel, 1, 1),
            )
        if in_channel == out_channel:
            self.right = None
        else:
            self.right = Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x: Tensor) -> Tensor:
        factor = 1.0
        if self.rezero:
            factor = self.alpha
        x1 = factor * self.left(x) + (x if self.right is None else self.right(x))
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
                Conv2d(in_channels, mid_channels, 1, 1),
                LeakyReLU(0.2, inplace=True),
                Conv2d(mid_channels, mid_channels, 3, 1, 1),
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
        x1 = factor * self.left(x) + (x if self.right is None else self.right(x))
        return x1
