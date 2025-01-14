
import sys
sys.path.extend(['/data1/Chenbingyuan/Depth-Completion/src'])
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Conv2d, init, Sequential, ModuleDict
from custom_blocks import *

from .modules import FirstModule, UNetModule, StackedBottleNeck
from einops import rearrange
from torch.cuda.amp import autocast
from .utils import save_feature_as_uint8colored
from .cby_unet import *

rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]

sf_chan = [512, 512, 512, 512, 256, 128, 64]

# class UNet(Module):
#     def __init__(self,rezero: bool = True):
#         super(UNet, self).__init__()
#         from src.baselines.BPnet.models.BPNet import Net as BPnetModel
#         self.network = BPnetModel() 
        
#     def forward(self,
#         rgb: Tensor,
#         point: Tensor,
#         hole_point: Tensor, ):
#         fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
#         K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#         K = K.unsqueeze(0)
#         depth = self.network(rgb.cuda(), point.cuda(), K.cuda())  
#         return depth,depth,depth,depth

class UNet(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = False):
        super(UNet, self).__init__()
        # rgb+x channel
        layer = FirstModule(rgb_x_chan[1], rgb_x_chan[0], rezero)
        for i in range(2, rgb_x_layer_num):
            layer = UNetModule(layer, rgb_x_chan[i], rgb_x_chan[i - 1], rezero)

        self.rgb_x_block = Sequential(
            Conv2d(5, rgb_x_chan[i], 3, 1, 1),
            StackedBottleNeck(rgb_x_chan[i], rgb_x_chan[i], rezero),
            layer,
            StackedBottleNeck(2 * rgb_x_chan[i], rgb_x_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 1, 3, 1, 1),
        )
        
        # sfmap channel
        sf_layer = FirstModule(sf_chan[1], sf_chan[0], rezero)
        for i in range(2, rgb_x_layer_num):
            sf_layer = UNetModule(sf_layer, sf_chan[i], sf_chan[i - 1], rezero)

        # # 使用G2时注释
        self.sfmap_block = Sequential(
            Conv2d(3, sf_chan[i], 3, 1, 1),
            StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
            sf_layer,
            StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
            # Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
        )
        
        self.softmax = nn.Softmax(dim=1)
        
        # initializing
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, a=0.2)
                init.zeros_(m.bias)
        
 
    @autocast()
    def forward(
        self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, 
    ) -> Tensor:

        x1 = torch.cat((rgb, point, hole_point), dim=1)
        depth = self.rgb_x_block(x1)


        
        # # 使用G2时注释
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )  #sfv2 点云 + depth预测sf，注意要改block里面的in_channel数量  

        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  # sfv2或者不使用p
        # s = torch.ones_like(sfmap[:,0,:,:].unsqueeze(1))  # 去掉s
        
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob
        # f = sfmap[:,0,:,:].unsqueeze(1)  # 去掉s
        f = torch.nn.functional.hardtanh(f, min_val= -0.5, max_val=0.5)  # sfv2或者不适用prob
        # f = torch.zeros_like(sfmap[:,0,:,:].unsqueeze(1))  # 不使用f
        
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        # prob = self.softmax(sfmap[:,1,:,:]).unsqueeze(1)  # 去掉s
        # prob = torch.ones_like(f)  # 不使用p
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        # prob = f
        return depth,s,f,prob


# # 直接将参数量翻倍
# class UNet(Module):
#     def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = False):
#         super(UNet, self).__init__()
#         # rgb+x channel
#         layer = FirstModule(rgb_x_chan[1], rgb_x_chan[0], rezero)
#         for i in range(2, rgb_x_layer_num):
#             layer = UNetModule(layer, rgb_x_chan[i], rgb_x_chan[i - 1], rezero)

#         self.rgb_x_block = Sequential(
#             Conv2d(5, rgb_x_chan[i], 3, 1, 1),
#             StackedBottleNeck(rgb_x_chan[i], rgb_x_chan[i], rezero),
#             layer,
#             StackedBottleNeck(2 * rgb_x_chan[i], rgb_x_chan[i], rezero),
#             Conv2d(rgb_x_chan[i], 1, 3, 1, 1),
#         )
        
#         # sfmap channel
#         sf_layer = FirstModule(sf_chan[1], sf_chan[0], rezero)
#         for i in range(2, rgb_x_layer_num):
#             sf_layer = UNetModule(sf_layer, sf_chan[i], sf_chan[i - 1], rezero)

#         self.sfmap_block = Sequential(
#             Conv2d(3, sf_chan[i], 3, 1, 1),
#             StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
#             sf_layer,
#             StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
#             Conv2d(rgb_x_chan[i], 1, 3, 1, 1),  #直接将参数量翻倍
#         )
        
#         # self.softmax = nn.Softmax(dim=1)
        
#         # initializing
#         for m in self.modules():
#             if isinstance(m, Conv2d):
#                 init.kaiming_normal_(m.weight, a=0.2)
#                 init.zeros_(m.bias)
        
 
#     @autocast()
#     def forward(
#         self,
#         rgb: Tensor,
#         point: Tensor,
#         hole_point: Tensor, 
#     ) -> Tensor:

#         x1 = torch.cat((rgb, point, hole_point), dim=1)
#         depth = self.rgb_x_block(x1)
#         x2 = torch.cat((depth, point, hole_point), dim=1)
#         output = self.sfmap_block(x2)
#         return output,output,output,output

# # 直接将参数量翻倍_2
# class UNet(Module):
#     def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = False):
#         super(UNet, self).__init__()
#         # rgb+x channel
#         layer = FirstModule(rgb_x_chan[1], rgb_x_chan[0], rezero)
#         for i in range(2, rgb_x_layer_num):
#             layer = UNetModule(layer, rgb_x_chan[i], rgb_x_chan[i - 1], rezero)

#         self.rgb_x_block = Sequential(
#             Conv2d(5, rgb_x_chan[i], 3, 1, 1),
#             StackedBottleNeck(rgb_x_chan[i], rgb_x_chan[i], rezero),
#             layer,
#             StackedBottleNeck(2 * rgb_x_chan[i], rgb_x_chan[i], rezero),
#             Conv2d(rgb_x_chan[i], 1, 3, 1, 1),
#         )
        
#         # sfmap channel
#         sf_layer = FirstModule(sf_chan[1], sf_chan[0], rezero)
#         for i in range(2, rgb_x_layer_num):
#             sf_layer = UNetModule(sf_layer, sf_chan[i], sf_chan[i - 1], rezero)

#         self.sfmap_block = Sequential(
#             Conv2d(1, sf_chan[i], 3, 1, 1),
#             StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
#             sf_layer,
#             StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
#             Conv2d(rgb_x_chan[i], 1, 3, 1, 1),  #直接将参数量翻倍
#         )
        
#         # self.softmax = nn.Softmax(dim=1)
        
#         # initializing
#         for m in self.modules():
#             if isinstance(m, Conv2d):
#                 init.kaiming_normal_(m.weight, a=0.2)
#                 init.zeros_(m.bias)
        
 
#     @autocast()
#     def forward(
#         self,
#         rgb: Tensor,
#         point: Tensor,
#         hole_point: Tensor, 
#     ) -> Tensor:

#         x1 = torch.cat((rgb, point, hole_point), dim=1)
#         depth = self.rgb_x_block(x1)
#         output = self.sfmap_block(depth)
#         return output,output,output,output


# # redc
# class Unet(Module):
#     def __init__(self,rezero: bool = True):
#         super(Unet, self).__init__()
#         from baselines.ReDC.redc import ReDC
#         from baselines.ReDC.config import args as args_ReDC
#         self.network = ReDC(args_ReDC) 
        
#     def forward(self,
#         rgb: Tensor,
#         point: Tensor,
#         hole_point: Tensor, ):
#         fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
#         K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#         K = K.unsqueeze(0)
#         depth = self.network(rgb.cuda(), point.cuda(), K.cuda())  
#         return depth,depth,depth,depth
