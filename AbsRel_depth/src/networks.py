import torch
from torch import Tensor
from torch.nn import Module, Conv2d, init, Sequential, Softmax
from .modules import FirstModule, UNetModule, StackedBottleNeck,Encoder, Decoder
import torch.nn as nn
chan = [512, 512, 512, 512, 256, 128, 64]


class UNet(Module):
    def __init__(self, layer_num: int = 7, rezero: bool = True):
        super(UNet, self).__init__()
        self.softmax = Softmax(dim=1)

        rgb_x_chan = chan
        layer = FirstModule(rgb_x_chan[1], rgb_x_chan[0], rezero)
        for i in range(2, layer_num):
            layer = UNetModule(layer, rgb_x_chan[i], rgb_x_chan[i - 1], rezero)

        self.rgb_x_block = Sequential(
            Conv2d(5, rgb_x_chan[i], 3, 1, 1),
            StackedBottleNeck(rgb_x_chan[i], rgb_x_chan[i], rezero),
            layer,
            StackedBottleNeck(2 * rgb_x_chan[i], rgb_x_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 1, 3, 1, 1),
        )
        
        sf_chan = chan
        # sfmap channel
        sf_layer = FirstModule(sf_chan[1], sf_chan[0], rezero)
        for i in range(2, layer_num):
            sf_layer = UNetModule(sf_layer, sf_chan[i], sf_chan[i - 1], rezero)

        # # 使用G2时注释
        self.sfmap_block = Sequential(
            Conv2d(3, sf_chan[i], 3, 1, 1),
            StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
            sf_layer,
            StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
        )

        # initializing
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, a=0.2)
                init.zeros_(m.bias)

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
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob
        f = torch.nn.functional.hardtanh(f, min_val= -0.5, max_val=0.5)  # sfv2或者不适用prob
        
        
        # prob是概率 使用prob                                              
        prob = torch.sigmoid(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth
    
# class V2Net(nn.Module):
#     def __init__(self, dims=[96, 192, 384, 768], depths=[3, 3, 9, 3], dp_rate=0.0, norm_type='CNX'):
#         super(V2Net, self).__init__()

#         self.visual_branch = nn.Sequential(
#                 Encoder(5, dims, depths, dp_rate, norm_type),
#                 Decoder(1, dims, norm_type)
#             )
        
#         self.refine_branch = nn.Sequential(
#                 Encoder(3, dims, depths, dp_rate, norm_type),
#                 Decoder(3, dims, norm_type)
#         )
        
#         self.softmax = nn.Softmax(dim=1)
#         # initializing
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv2d):
#             nn.init.xavier_normal_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, rgb, raw, hole_raw):
#         x = torch.cat((rgb, raw, hole_raw), dim=1)
        
#         rel_depth = self.visual_branch(x)
#         refine_branch_output = self.refine_branch(torch.cat((x[:, 3:, :, :], raw), dim=1))
        
#         s = torch.sigmoid(refine_branch_output[:,0,:,:]).unsqueeze(1) * 2  # sfv2或者不使用p
#         # s = torch.ones_like(sfmap[:,0,:,:].unsqueeze(1))  # 去掉s
        
        
#         # f缩放到-1 1
#         f = refine_branch_output[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob
#         # f = sfmap[:,0,:,:].unsqueeze(1)  # 去掉s
#         f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  # sfv2或者不适用prob
#         # f = torch.zeros_like(sfmap[:,0,:,:].unsqueeze(1))  # 不使用f
        
        
#         # prob是概率 使用prob                                              
#         prob = self.softmax(refine_branch_output[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
#         # prob = self.sigmoid(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，sigmoid
#         # prob = self.softmax(sfmap[:,1,:,:]).unsqueeze(1)  # 去掉s
#         # prob = torch.ones_like(f)  # 不使用p
#         # # 使用G2时注释
#         depth = (rel_depth*s+f)*prob + rel_depth * (1-prob)
#         # prob = f
#         # return depth,s,f,prob
#         return depth

class V2Net(nn.Module):
    # 用来测SPNorm
    def __init__(self, dims=[3,3,27,3], depths=[192,384,768,1536], dp_rate=0.2, norm_type='CNX'):
        super(V2Net, self).__init__()

        self.visual_branch = nn.Sequential(
                Encoder(5, dims, depths, dp_rate, norm_type),
                Decoder(1, dims, norm_type)
            )
        # initializing
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, rgb, raw, hole_raw):
        x = torch.cat((rgb, raw, hole_raw), dim=1)
        
        depth = self.visual_branch(x)
        
        return depth