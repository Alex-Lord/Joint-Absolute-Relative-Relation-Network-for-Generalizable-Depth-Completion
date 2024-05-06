from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module, Conv2d, init, Sequential, Softmax
from .modules import FirstModule, UNetModule, StackedBottleNeck

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
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth
