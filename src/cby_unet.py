""" Full assembly of the parts to form the complete network """

from .cby_unet_parts import *
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn import init


        
class CBYUNet(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(CBYUNet, self).__init__()
        self.rgb_in_conv2d = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, 1),
            StackedBottleNeck(64, 64, rezero),
        )
        self.rgb_out_conv2d = Conv2d(64, 1, 3, 1, 1)
        self.rgb_down1 = (Down(64, 128, rezero))
        self.rgb_down2 = (Down(128, 256, rezero))
        self.rgb_down3 = (Down(256, 512, rezero))
        self.rgb_down4 = (Down(512, 512, rezero))
        self.rgb_down5 = (Down(512, 512, rezero))
        self.rgb_down6 = (Down(512, 512, rezero))
        self.rgb_up1 = (Up(512, 512, rezero))
        self.rgb_up2 = (Up(512, 512, rezero))
        self.rgb_up3 = (Up(512, 512, rezero))
        self.rgb_up4 = (Up(512, 256, rezero))
        self.rgb_up5 = (Up(256, 128, rezero))
        self.rgb_up6 = (Up(128, 64, rezero))
        

        self.sf_in_conv2d = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            StackedBottleNeck(64, 64, rezero),
        )
        self.sf_out_conv2d = Conv2d(64, 3, 3, 1, 1)
        # self.sf_down1 = (Down(128, 128, rezero))
        # self.sf_down2 = (Down(256, 256, rezero))
        # self.sf_down3 = (Down(512, 512, rezero))
        # self.sf_down4 = (Down(1024, 512, rezero))
        # self.sf_down5 = (Down(1024, 512, rezero))
        # self.sf_down6 = (Down(1024, 512, rezero))
        # self.sf_up1 = (Up(1024, 512, rezero))
        # self.sf_up2 = (Up(512, 512, rezero))
        # self.sf_up3 = (Up(512, 512, rezero))
        # self.sf_up4 = (Up(512, 256, rezero))
        # self.sf_up5 = (Up(256, 128, rezero))
        # self.sf_up6 = (Up(128, 64, rezero))
        
        # self.sf_down1 = (Down(64, 128, rezero))
        # self.sf_down2 = (Down(128, 256, rezero))
        # self.sf_down3 = (Down(256, 512, rezero))
        # self.sf_down4 = (Down(512, 512, rezero))
        # self.sf_down5 = (Down(512, 512, rezero))
        # self.sf_down6 = (Down(512, 512, rezero))
        # self.sf_up1 = (Up(1024, 512, rezero))
        # self.sf_up2 = (Up(512, 512, rezero))
        # self.sf_up3 = (Up(512, 512, rezero))
        # self.sf_up4 = (Up(512, 256, rezero))
        # self.sf_up5 = (Up(256, 128, rezero))
        # self.sf_up6 = (Up(128, 64, rezero))
        
        self.sf_down1 = (Down(64, 128, rezero))
        self.sf_down2 = (Down(128, 256, rezero))
        self.sf_down3 = (Down(256, 512, rezero))
        self.sf_down4 = (Down(512, 512, rezero))
        self.sf_down5 = (Down(512, 512, rezero))
        self.sf_down6 = (Down(512, 512, rezero))
        self.sf_up1 = (Up(512, 512, rezero))
        self.sf_up2 = (Up(512, 512, rezero))
        self.sf_up3 = (Up(512, 512, rezero))
        self.sf_up4 = (Up(512, 256, rezero))
        self.sf_up5 = (Up(256, 128, rezero))
        self.sf_up6 = (Up(128, 64, rezero))
        
        self.softmax = nn.Softmax(dim=1)
        
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

        y1 = torch.cat((rgb, point, hole_point), dim=1)
        x =  self.rgb_in_conv2d(y1)
        x1 = self.rgb_down1(x)
        x2 = self.rgb_down2(x1)
        x3 = self.rgb_down3(x2)
        x4 = self.rgb_down4(x3)
        x5 = self.rgb_down5(x4)
        x6 = self.rgb_down6(x5)
        
        
        x7= self.rgb_up1(x6, x5) 
        x8 = self.rgb_up2(x7, x4) 
        x9 = self.rgb_up3(x8, x3)
        x10 = self.rgb_up4(x9, x2) 
        x11 = self.rgb_up5(x10, x1) 
        x12 = self.rgb_up6(x11, x)
        
        depth = self.rgb_out_conv2d(x12) # 64->1
        # print(f'Done Depth!! depth={depth.shape}')
        
        
        y = torch.cat((y1[:,3:,:,:], depth), dim=1)
        y = self.sf_in_conv2d(y)  # 64
        # y1 = self.sf_down1(torch.cat((y, x12), dim=1))
        # y2 = self.sf_down2(torch.cat((y1, x11), dim=1))
        # y3 = self.sf_down3(torch.cat((y2, x10), dim=1))
        # y4 = self.sf_down4(torch.cat((y3, x9), dim=1))
        # y5 = self.sf_down5(torch.cat((y4, x8), dim=1))
        # y6 = self.sf_down6(torch.cat((y5, x7), dim=1))
        # y_out = self.sf_up1(torch.cat((y6, x6), dim=1), y5)
        # y_out = self.sf_up2(y_out, y4)
        # y_out = self.sf_up3(y_out, y3)
        # y_out = self.sf_up4(y_out, y2)
        # y_out = self.sf_up5(y_out, y1)
        # y_out = self.sf_up6(y_out, y)
        
        y1 = self.sf_down1(y)
        y2 = self.sf_down2(y1)
        y3 = self.sf_down3(y2)
        y4 = self.sf_down4(y3)
        # print(f'y4.shape={y4.shape}')
        y5 = self.sf_down5(y4)
        # print(f'y5.shape={y5.shape}')
        y6 = self.sf_down6(y5)
        # print(f'y6.shape={y1.shape}')
        # y_out = self.sf_up1(torch.cat((y6, x6), dim=1), y5)
        y_out = self.sf_up1(y6, y5)
        # print(f'y_up1_out.shape={y_out.shape}')
        y_out = self.sf_up2(y_out, y4)
        y_out = self.sf_up3(y_out, y3)
        y_out = self.sf_up4(y_out, y2)
        y_out = self.sf_up5(y_out, y1)
        y_out = self.sf_up6(y_out, y)
        
        sfmap = self.sf_out_conv2d(y_out)
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0
        f = sfmap[:,1,:,:].unsqueeze(1)
        f = torch.nn.functional.hardtanh(f, min_val=-1, max_val=1)
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)
        depth = (depth*s+f)*prob + depth * (1-prob)
        return depth,s,f,prob
    
    
    


'''
class cby_rgb_x_block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, rezero=True):
        super(cby_rgb_x_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv2d = nn.Sequential(
            nn.Conv2d(5, 64, 3, 1, 1),
            StackedBottleNeck(64, 64, rezero),
        )
        self.out_conv2d = StackedBottleNeck(64, 1, rezero)
        self.down1 = (Down(64, 128, rezero))
        self.down2 = (Down(128, 256, rezero))
        self.down3 = (Down(256, 512, rezero))
        self.down4 = (Down(512, 512, rezero))
        self.down5 = (Down(512, 512, rezero))
        self.down6 = (Down(512, 512, rezero))
        self.up1 = (Up(512, 512, rezero))
        self.up2 = (Up(512, 512, rezero))
        self.up3 = (Up(512, 256, rezero))
        self.up4 = (Up(256, 128, rezero))
        self.up5 = (Up(128, 64, rezero))

    def forward(self, x):
        print('Now we are in rgb')
        print(f'x.shape = {x.shape}')
        x =  self.in_conv2d(x)
        print(f'x.shape = {x.shape}')
        x1 = self.down1(x)
        print(f'x1.shape = {x1.shape}')
        x2 = self.down2(x1)
        print(f'x2.shape = {x2.shape}')
        x3 = self.down3(x2)
        print(f'x3.shape = {x3.shape}')
        x4 = self.down4(x3)
        print(f'x4.shape = {x4.shape}')
        x5 = self.down5(x4)
        print(f'x5.shape = {x5.shape}')
        x6 = self.down6(x5)
        print(f'x6.shape = {x6.shape}')
        x7= self.up1(x6, x5)  # 512
        print(f'x7.shape = {x7.shape}')
        x8 = self.up2(x7, x4)  # 512
        print(f'x8.shape = {x8.shape}')
        x9 = self.up3(x8, x3)  # 256
        print(f'x9.shape = {x9.shape}')
        x10 = self.up4(x9, x2)  # 128
        print(f'x10.shape = {x10.shape}')
        x11 = self.up5(x10, x1)  # 64
        print(f'x11.shape = {x11.shape}')
        x = self.out_conv2d(x11) # 64->1
        return x, x6, x7, x8, x9, x10, x11

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class cby_sf_block(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, rezero=True):
        super(cby_sf_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_conv2d = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            StackedBottleNeck(64, 64, rezero),
        )
        self.out_conv2d = StackedBottleNeck(64, 3, rezero)
        self.down1 = (Down(128, 128, rezero))
        self.down2 = (Down(256, 256, rezero))
        self.down3 = (Down(512, 512, rezero))
        self.down4 = (Down(1024, 1024, rezero))
        self.down5 = (Down(1024, 1024, rezero))
        self.down6 = (Down(1024, 1024, rezero))
        self.up1 = (Up(1024, 1024, rezero))
        self.up2 = (Up(1024, 1024, rezero))
        self.up3 = (Up(1024, 512, rezero))
        self.up4 = (Up(512, 256, rezero))
        self.up5 = (Up(256, 64, rezero))

    def forward(self, x, y6, y7, y8, y9, y10, y11):
        x =  self.in_conv2d(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

'''