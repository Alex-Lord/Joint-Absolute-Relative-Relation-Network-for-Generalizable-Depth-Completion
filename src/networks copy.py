import sys
sys.path.extend(['/data/4TSSD/cby/Trans_G2/src'])
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Conv2d, init, Sequential, ModuleDict
from custom_blocks import *
from swin_transformer_v2 import SwinTransformerV2
# from mmcls.models import SwinTransformerV2
from .modules import FirstModule, UNetModule, StackedBottleNeck
from einops import rearrange
from torch.cuda.amp import autocast
from .utils import save_feature_as_uint8colored
from .nconv import NConvUNet
from .cby_unet import *
# import timm
kwargs_1 = {"image_size": 320,
          "patch_size": 16,
          "dim": 1024,
          "depth":6,
          "heads":16,
          "mlp_dim":2048,
          "dropout":0.1,
          "emb_dropout":0.1}

kwargs_2 = {
          "dim": 1024,
          "depth":6,
          "heads":16,
          "mlp_dim":2048,
          "dropout":0.1}

kwargs_3 = {
          "dim": 1024,
          "depth":6,
          "heads":16,
          "mlp_dim":2048,
          "dropout":0.1}

kwargs_4 = {
          "dim": 1024,
          "depth":6,
          "heads":16,
          "mlp_dim":2048,
          "dropout":0.1}

# 使用本地的swin
kwargs_swin = {
    'img_size': 320,
    'patch_size': 4,
    'in_chans': 4,
    'num_classes': 0,
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 10,
    'pretrained_window_sizes': [0, 0, 0, 0],
}

# # 使用mmcl swin
# kwargs_swin = {
#     'arch':'tiny',
#     'img_size':320,
#     'window_size':10,
#     'in_channels':4,
#     'pad_small_map':True,
#     'pretrained_window_sizes':[0, 0, 0, 0],
# }
module_dict = ModuleDict({
                'vit_layer1':Sequential(
                    ViT(head_block=Sequential(head1(1024, 256), ), **kwargs_1),
                    ),
                'vit_layer2':Sequential(
                    ViTBlock(head_block=head2(1024, 256), **kwargs_2),
                    ),
                'vit_layer3':Sequential(
                    ViTBlock(head_block=head3(1024, 256), **kwargs_3),
                    ),
                'vit_layer4':Sequential(
                    ViTBlock(head_block=head4(1024, 256), **kwargs_4),
                    ),
                'swin_trans_layer':Sequential(SwinTransformerV2(**kwargs_swin)),
                })

class RefineNet(nn.Module):
    def __init__(self, swin=True):
        super(RefineNet, self).__init__()
        self.swin = swin
        self.swin_model = module_dict['swin_trans_layer']
        # self.swin_model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True, features_only=True, window_size=20 ,img_size=(320, 320),pretrained_cfg_overlay=dict(file='/data/4TSSD/cby/dataset/g2_dataset/pretrained_models/pytorch_model.bin'))
        # for param in self.swin_model.parameters():
        #     param.requires_grad = False

        self.rearrange = rearrange
        self.inplanes = 256
        self.do = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        
        # self.layer1 = module_dict['vit_layer1']
        # self.layer2 = module_dict['vit_layer2']
        # self.layer3 = module_dict['vit_layer3']
        # self.layer4 = module_dict['vit_layer4']
        
        self.p_ims1d2_outl1_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.adapt_stage1_b = self._make_rcu(256, 256, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g1_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        
        self.p_ims1d2_outl2_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.p_ims1d2_outl3_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.p_ims1d2_outl4_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        if self.swin:
            self.p_ims1d2_outl1_dimred = nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False)
            self.adapt_stage1_b = self._make_rcu(256, 256, 2, 2)
            self.mflow_conv_g1_pool = self._make_crp(256, 256, 4)
            self.mflow_conv_g1_b = self._make_rcu(256, 256, 3, 2)
            self.mflow_conv_g1_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            
            self.p_ims1d2_outl2_dimred = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
            self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
            self.adapt_stage2_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
            self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
            self.mflow_conv_g2_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

            self.p_ims1d2_outl3_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
            self.adapt_stage3_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
            self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
            self.mflow_conv_g3_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

            self.p_ims1d2_outl4_dimred = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
            self.adapt_stage4_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
            self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
            self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)
        
        self.head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes,stages)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, blocks, stages):
        layers = [RCUBlock(in_planes, out_planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,
            rgb: Tensor,
            point: Tensor,
            hole_point: Tensor,):
        # x = torch.cat((rgb, point), dim=1)
        # ln是经过Head Resemble后的数据
        # yn是经过transformer之后的数据
        h0 = rgb.shape[2]
        if self.swin:
            l1, l2, l3, _, l4 = self.swin_model(rgb)  # 本地Swin
            l1 = rearrange(l1, 'b (h w) c -> b c h w', h=int(h0/4))  # 本地Swin
            l2 = rearrange(l2, 'b (h w) c -> b c h w', h=int(h0/8))  # 本地Swin
            l3 = rearrange(l3, 'b (h w) c -> b c h w', h=int(h0/16))  # 本地Swin
            l4 = rearrange(l4, 'b (h w) c -> b c h w', h=int(h0/32))  # 本地Swin
            
            # l = self.swin_model(x)  # mmcl Swin
            # l1,l2,l3,l4 = l  # mmcl Swin
        # else:
        #     l1, y1 = self.layer1(x)
        #     l2, y2 = self.layer2(y1)
        #     l3, y3 = self.layer3(y2)
        #     l4, y4 = self.layer4(y3)
        # 2个dropout
        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4) 
        x4 = self.adapt_stage1_b(x4)  # RCU
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)  # CRP
        x4 = self.mflow_conv_g1_b(x4)  #RCU
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)  
        x4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x4)  # 上采样为2倍 10,10->20,20

        x3 = self.p_ims1d2_outl2_dimred(l3) 
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.do(x1)
        x1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x1)
        
        out = self.head(x1)
        return out

rgb_x_chan = [512, 512, 512, 512, 256, 128, 64]

sf_chan = [512, 512, 512, 512, 256, 128, 64]


class UNet(Module):
    def __init__(self, rgb_x_layer_num: int = 7, rezero: bool = True):
        super(UNet, self).__init__()
        # self.Probabilistic_NCNNs = Probabilistic_NCNNs()
        
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

        # self.rgb_x_block = RefineNet()
        

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

        # self.Attention = Fast_dense_to_sparse_Attention(1)
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
        # point[hole_point==0] = 0
        
        # hole_point = torch.zeros_like(point)
        # point[hole_point==0] = 0
        x1 = torch.cat((rgb, point, hole_point), dim=1)
        # depth = self.rgb_x_block(rgb)
        depth = self.rgb_x_block(x1)

        # attention = self.Attention(point, depth)
        
        # s缩放到0-2
        # sfmap = self.sfmap_block(x1[:,3:,:,:])  # 只利用点云预测sf
        sfmap = self.sfmap_block(torch.cat((x1[:,3:,:,:], depth), dim=1) )# 点云 + depth预测sf，注意要改block里面的in_channel数量 
        # sfmap = self.sfmap_block(torch.cat((dense_out[:,:2,:,:],depth),dim=1))  # dense点云(pcnn) + depth预测sf，注意要改block里面的in_channel数量
        # sfmap = self.sfmap_block(torch.cat((attention, depth, point),dim=1))  # attention + depth预测sf，注意要改block里面的in_channel数量
        
        b, c, h, w = depth.size()
        # sfmap=torch.ones((b,3,h,w))
        # s = F.interpolate(torch.sigmoid((F.avg_pool2d(sfmap[:,0,:,:], kernel_size=(h//8, w//8))).unsqueeze(1))* 2.0, size=(h, w), mode='bilinear')  # 让s的分辨率降低为 8 * 8
        s = torch.sigmoid(sfmap[:,0,:,:]).unsqueeze(1) * 2.0
        # if torch.isnan(s).any() or torch.isinf(s).any():
        #     print("s中存在inf或nan")
        
        # f缩放到-1 1
        f = sfmap[:,1,:,:].unsqueeze(1)
        f = torch.nn.functional.hardtanh(f, min_val=-1, max_val=1)
        
        # if torch.isnan(f).any() or torch.isinf(f).any():
        #     print("f中存在inf或nan")
        
        # prob是概率 使用prob                                              
        prob = self.softmax(sfmap[:,2,:,:]).unsqueeze(1)
        depth = (depth*s+f)*prob + depth * (1-prob)

        # return depth,s,f,prob,dense_out
        return depth,s,f,prob

class MyNormalization(nn.Module):
    def __init__(self, max_value=1.0):
        super(MyNormalization, self).__init__()
        self.max_value = max_value
        
    @autocast()
    def forward(self, x):
        # 将张量归一化到 [0, 1] 范围内
        min_val = x.min()
        max_val = x.max()
        x_norm = (x - min_val) / (max_val - min_val) * self.max_value
        return x_norm

class Probabilistic_NCNNs(Module):
    def __init__(self):
        super().__init__() 
        self.__name__ = 'pncnn'

        self.conf_estimator = UNetSP(1, 1)
        self.nconv = NConvUNet(1, 1)
        self.var_estimator = UNetSP(1, 1)

    def forward(self, x0):  
        x0 = x0[:,:1,:,:]  # Use only depth
        c0 = self.conf_estimator(x0)  # Input Confidence Estimation Network
        xout, cout = self.nconv(x0, c0)  # Normalized Convolution Network
        cout = self.var_estimator(cout)  # Noise Variance Estimation Network
        out = torch.cat((xout, cout, c0), 1)
        return out 