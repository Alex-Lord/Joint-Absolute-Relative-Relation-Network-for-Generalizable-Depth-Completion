
import sys
sys.path.extend(['/data1/name/JARRN/src'])
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, init, Sequential
from custom_blocks import *

from .modules import FirstModule, UNetModule, StackedBottleNeck
from torch.cuda.amp import autocast


rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]

sf_chan = [512, 512, 512, 512, 256, 128, 64]


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


        self.sfmap_block = Sequential(
            Conv2d(3, sf_chan[i], 3, 1, 1),
            StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
            sf_layer,
            StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  
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
        
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )  
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  

        f = sfmap[:,1,:,:].unsqueeze(1)
        f = torch.nn.functional.hardtanh(f, min_val= -0.5, max_val=0.5) 

        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  

        dense_depth = depth.clone()
        depth = (depth*s+f)*prob + depth * (1-prob)
        
        return depth,s,f,prob,dense_depth
