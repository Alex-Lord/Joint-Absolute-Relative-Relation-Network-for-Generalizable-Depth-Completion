import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.extend(['/data/8T/cby/Trans_G2/src'])
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Conv2d, init, Sequential, ModuleDict
from custom_blocks import *
from src.modules import FirstModule, UNetModule, StackedBottleNeck
from einops import rearrange
from torch.cuda.amp import autocast
from src.utils import save_feature_as_uint8colored
from src.nconv import NConvUNet
from src.cby_unet import *






class G2_Mono(Module):
    def __init__(self, layer_num: int = 7, rezero: bool = True):
        super(G2_Mono, self).__init__()
        chan = [512, 512, 512, 512, 256, 128, 64]
        layer = FirstModule(chan[1], chan[0], rezero)
        for i in range(2, layer_num):
            layer = UNetModule(layer, chan[i], chan[i - 1], rezero)

        self.block = Sequential(
            Conv2d(5, chan[i], 3, 1, 1),
            StackedBottleNeck(chan[i], chan[i], rezero),
            layer,
            StackedBottleNeck(2 * chan[i], chan[i], rezero),
            Conv2d(chan[i], 1, 3, 1, 1),
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
        depth = self.block(x1)
        return depth




class g2_UNet(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
        super(g2_UNet, self).__init__()
        rgb_x_layer_num = len(rgb_x_chan)
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
        
        # 使用Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(
        self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, 
    ) -> Tensor:
        x1 = torch.cat((rgb, point, hole_point), dim=1)
        depth = self.rgb_x_block(x1)
        b, c, h, w = depth.size()
        
        sfmap=torch.ones((b,3,h,w)) # 使用G2
        
        s = torch.ones_like(sfmap)  # 去掉s

        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob

        prob = torch.ones_like(f)  # 不使用p

        return depth,s,f,prob



class NLSPN_DIODE_HRWSI(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        from src.baselines.NLSPN.src.model.nlspnmodel import NLSPNModel
        from src.baselines.NLSPN.src.config import args as args_NLSPN   
        super(NLSPN_DIODE_HRWSI, self).__init__()

        self.rgb_x_block = NLSPNModel(args_NLSPN) 

        # 使用Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)
    def forward(
        self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, 
    ) -> Tensor:

        depth = self.rgb_x_block(rgb, point)      
        b, c, h, w = depth.size()
        
        sfmap=torch.ones((b,3,h,w)) 
        s = torch.ones_like(sfmap)  # 去掉s
        
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob

        prob = torch.ones_like(f)  # 不使用p

        return depth,s,f,prob

class CFormer_DIODE_HRWSI(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(CFormer_DIODE_HRWSI, self).__init__()
        from src.baselines.CFormer.model.completionformer import CompletionFormer, check_args
        from src.baselines.CFormer.model.config import args as args_cformer
        self.rgb_x_block = CompletionFormer(args_cformer) 

        # 使用Xavier初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, 0.2)
                if m.bias is not None:
                    init.zeros_(m.bias)


    # @autocast()
    def forward(
        self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, 
    ) -> Tensor:

        depth = self.rgb_x_block(rgb, point)
        depth = depth['pred']
        
        b, c, h, w = depth.size()
        sfmap=torch.ones((b,3,h,w)) 
        s = torch.ones_like(sfmap)  # 去掉s
        
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob

        prob = torch.ones_like(f)  # 不使用p

        return depth,s,f,prob

class sfv2_UNet_only_p(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_only_p, self).__init__()
        
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]

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
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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
        
        s = torch.ones_like(sfmap[:,0,:,:].unsqueeze(1))  # 去掉s， 只有p
        f = torch.zeros_like(sfmap[:,0,:,:].unsqueeze(1))  # 不使用f
        
        prob = self.softmax(sfmap[:,0,:,:]).unsqueeze(1)  # 只有p
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_only_s(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_only_s, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]

        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 1.5  # sfv2或者不使用p
        
     
        f = torch.zeros_like(sfmap[:,0,:,:].unsqueeze(1))  # 不使用f
        

        prob = torch.ones_like(f)  # 不使用p
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_only_f(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_only_f, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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
        
        s = torch.ones_like(sfmap[:,0,:,:].unsqueeze(1))  # 去掉s
        
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  # sfv2或者不适用prob
        
        
        # prob是概率 使用prob                                              
        prob = torch.ones_like(f)  # 不使用p
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_no_p(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_no_p, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]

        sf_chan = [512, 512, 512, 512, 256, 128, 64]

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
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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

        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  # sfv2 或者不适用prob
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  # sfv2或者不适用prob
        
        prob = torch.ones_like(f)  # 不使用p
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_no_f(Module):

    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_no_f, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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
        
        # f缩放到-1 1
        f = torch.zeros_like(sfmap[:,0,:,:].unsqueeze(1))  # 不使用f
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,1,:,:]).unsqueeze(1)  # 去掉s
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob
class sfv2_UNet_no_s(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_no_s, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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

        sf_layer = FirstModule(sf_chan[1], sf_chan[0], rezero)
        for i in range(2, rgb_x_layer_num):
            sf_layer = UNetModule(sf_layer, sf_chan[i], sf_chan[i - 1], rezero)

        # # 使用G2时注释
        self.sfmap_block = Sequential(
            Conv2d(3, sf_chan[i], 3, 1, 1),
            StackedBottleNeck(sf_chan[i], sf_chan[i], rezero),
            sf_layer,
            StackedBottleNeck(2 * sf_chan[i], sf_chan[i], rezero),
            Conv2d(rgb_x_chan[i], 2, 3, 1, 1),  #不用Prob，s或f
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
        
        s = torch.ones_like(sfmap[:,0,:,:].unsqueeze(1))  # 去掉s
        
        
        # f缩放到-1 1
        f = sfmap[:,0,:,:].unsqueeze(1)  # 去掉s
        f = torch.nn.functional.hardtanh(f, min_val= -0.5, max_val=0.5)  # sfv2或者不适用prob

        prob = self.softmax(sfmap[:,1,:,:]).unsqueeze(1)  # 去掉s
        # # 使用G2时注释
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob


class sfv2_UNet(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob
    


class sfv2_UNet_f22(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_f22, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -2, max_val=2)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_f11(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_f11, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob


class sfv2_UNet_f0505(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_f0505, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -0.5, max_val=0.5)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_s005(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_s005, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 0.5  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_s05(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_s05, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 5 
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob




class sfv2_UNet_KITTI(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_KITTI, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
        )
        
        self.softmax = nn.Softmax(dim=1)
        self.Attention = Fast_dense_to_sparse_Attention(1)
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

class sfv2_UNet_visual(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_visual, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        unscale_depth = depth.detach().clone()
        # # 使用G2时注释
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return unscale_depth,depth,s,f,prob



class sfv2_UNet_unscale(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_unscale, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        unscale_depth = depth.detach().clone()
        # # 使用G2时注释
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return unscale_depth,s,f,prob
    
class sfv2_UNet_bn(Module):
    
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = False):
        super(sfv2_UNet_bn, self).__init__()
        rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]
        sf_chan = [512, 512, 512, 512, 256, 128, 64]
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        unscale_depth = depth.detach().clone()
        # # 使用G2时注释
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return unscale_depth,s,f,prob
    
    

class sfv2_UNet_tiny(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_tiny, self).__init__()
        rgb_x_chan = [512, 512, 128, 64, 32]
        sf_chan = rgb_x_chan
        # rgb+x channel
        rgb_x_layer_num = len(rgb_x_chan)
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob
    

class sfv2_UNet_small(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_small, self).__init__()
        rgb_x_chan = [512, 512, 512 ,256, 128, 64, 32]
        sf_chan = rgb_x_chan
        # rgb+x channel
        rgb_x_layer_num = len(rgb_x_chan)
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob
    

class sfv2_UNet_large(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(sfv2_UNet_large, self).__init__()
        rgb_x_chan = [1024,1024,512,512,256,128,64]
        sf_chan = rgb_x_chan
        # rgb+x channel
        rgb_x_layer_num = len(rgb_x_chan)
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
            Conv2d(rgb_x_chan[i], 3, 3, 1, 1),  #原始sfv2
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
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量  sfv2
        b, c, h, w = depth.size()
        
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0  
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)  
        f = torch.nn.functional.hardtanh(f, min_val= -1, max_val=1)  
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)  # sfv2，使用p
        
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob

# EMDC
class EMDC_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(EMDC_retrain, self).__init__()
        from baselines.EMDC.models.EMDC import emdc
        self.network = emdc(depth_norm=False)
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        
        depth = self.network(rgb.cuda(),point.cuda())
        depth = depth[0]
        return depth,depth,depth,depth


# redc
class SDCM_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(SDCM_retrain, self).__init__()
        
        from baselines.SDCM.model import DepthCompletionNet
        from baselines.SDCM.config import args as args_SDCM
        self.network = DepthCompletionNet(args_SDCM)
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        import torchvision.transforms as transforms
        grayscale_transform = transforms.Grayscale(num_output_channels=1)  # 创建一个灰度转换器
        rgb = grayscale_transform(rgb)  # 使用转换器将RGB图像转换为灰度图像
        depth = self.network(rgb.cuda(), point.cuda())  
        return depth,depth,depth,depth

# GuideNet
class GuideNet_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(GuideNet_retrain, self).__init__()
        from baselines.GuideNet.utils import init_net
        import yaml
        from easydict import EasyDict as edict
        with open('/data/8T/cby/Trans_G2/src/baselines/GuideNet/configs/GNS.yaml', 'r') as file:
            self.config_data = yaml.load(file, Loader=yaml.FullLoader)
        self.GuideNetconfig = edict(self.config_data)
        self.network = init_net(self.GuideNetconfig)
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        
        depth, = self.network(rgb.cuda(), point.cuda())
        return depth,depth,depth,depth

# TWISE
class TWISE_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(TWISE_retrain, self).__init__()
        from baselines.TWISE.model import MultiRes_network_avgpool_diffspatialsizes
        from baselines.TWISE.utils import smooth2chandep
        sys.path.append('/data/8T/cby/Trans_G2/src/baselines/TWISE')
        import baselines.TWISE.metrics
        
        self.network = MultiRes_network_avgpool_diffspatialsizes()
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        from baselines.TWISE.utils import smooth2chandep
        gen_depth,_,_ = self.network(point.cuda(),rgb.cuda()) 
        gen_depth = smooth2chandep(gen_depth, params={'depth_maxrange': 1.0,}, device=None)
        depth = gen_depth
        return depth,depth,depth,depth

# MDAnet
class MDAnet_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(MDAnet_retrain, self).__init__()
        from baselines.MDANet.modules.net import network as MDAnet
        self.network = MDAnet()
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        
        depth = self.network(point.cuda(),rgb.cuda())
        depth = depth[0]
        return depth,depth,depth,depth

# PEnet
class PEnet_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(PEnet_retrain, self).__init__()
        from baselines.PEnet.model.model import PENet_C2
        from baselines.PEnet.model.config import args as args_penet
        self.network = PENet_C2(args_penet) 
        
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        depth = self.network(rgb.cuda(), point.cuda(), K.cuda())  
        return depth,depth,depth,depth


# redc
class ReDC_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(ReDC_retrain, self).__init__()
        from baselines.ReDC.redc import ReDC
        from baselines.ReDC.config import args as args_ReDC
        self.network = ReDC(args_ReDC) 
        
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        depth = self.network(rgb.cuda(), point.cuda(), K.cuda())  
        return depth,depth,depth,depth

class LRRU_retrain(Module):
    def __init__(self,rezero: bool = True):
        super(LRRU_retrain, self).__init__()
        from baselines.LRRU.configs import get as get_cfg
        from baselines.LRRU.model.model_dcnv2 import Model as LRRUModel
        import argparse
        self.arg = argparse.ArgumentParser(description='depth completion')
        self.arg.add_argument('-p', '--project_name', type=str, default='inference')
        self.arg.add_argument('-c', '--configuration', type=str, default='/data/8T/cby/Trans_G2/src/baselines/LRRU/configs/val_lrru_base_kitti.yml')
        self.arg = self.arg.parse_args()
        self.args_LRRU = get_cfg(self.arg)
        self.network = LRRUModel(self.args_LRRU)
    def forward(self,
        rgb: Tensor,
        point: Tensor,
        hole_point: Tensor, ):
        
        depth = self.network(rgb, point)['results'][-1]  
        return depth,depth,depth,depth