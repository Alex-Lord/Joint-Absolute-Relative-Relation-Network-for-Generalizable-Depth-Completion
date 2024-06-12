import sys
sys.path.append('/data1/Chenbingyuan/Depth-Completion/src/baselines/ReDC')
from basic import *
import torch 
from dataloaders_redc import transforms
import torch.nn as nn
import CoordConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

gks = 5
pad = [i for i in range(gks*gks)]
shift = torch.zeros(gks*gks, 4)
for i in range(gks):
    for j in range(gks):
        top = i
        bottom = gks-1-i
        left = j
        right = gks-1-j
        pad[i*gks + j] = torch.nn.ZeroPad2d((left, right, top, bottom))
        #shift[i*gks + j, :] = torch.tensor([left, right, top, bottom])
mid_pad = torch.nn.ZeroPad2d(((gks-1)/2, (gks-1)/2, (gks-1)/2, (gks-1)/2))
zero_pad = pad[0]

gks2 = 3     #guide kernel size
pad2 = [i for i in range(gks2*gks2)]
shift = torch.zeros(gks2*gks2, 4)
for i in range(gks2):
    for j in range(gks2):
        top = i
        bottom = gks2-1-i
        left = j
        right = gks2-1-j
        pad2[i*gks2 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

gks3 = 7     #guide kernel size
pad3 = [i for i in range(gks3*gks3)]
shift = torch.zeros(gks3*gks3, 4)
for i in range(gks3):
    for j in range(gks3):
        top = i
        bottom = gks3-1-i
        left = j
        right = gks3-1-j
        pad3[i*gks3 + j] = torch.nn.ZeroPad2d((left, right, top, bottom))

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def convbnrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def deconvbnrelu(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels),
		nn.ReLU(inplace=True)
	)

def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

def deconvbn(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    return nn.Sequential(
		nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            #norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=bias)

class SparseDownSampleClose(nn.Module):
    def __init__(self, stride):
        super(SparseDownSampleClose, self).__init__()
        self.pooling = nn.MaxPool2d(stride, stride)
        self.large_number = 600
    def forward(self, d, mask):
        encode_d = - (1-mask)*self.large_number - d

        d = - self.pooling(encode_d)
        mask_result = self.pooling(mask)
        d_result = d - (1-mask_result)*self.large_number

        return d_result, mask_result

class CSPNGenerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)

        #padding
        weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size*self.kernel_size):
            zero_pad = 0
            if(self.kernel_size==3):
                zero_pad = pad2[t]
            elif(self.kernel_size==5):
                zero_pad = pad[t]
            elif(self.kernel_size==7):
                zero_pad = pad3[t]
            if(t < int((self.kernel_size*self.kernel_size-1)/2)):
                weight_pad[t] = zero_pad(guide[:, t:t+1, :, :])
            elif(t > int((self.kernel_size*self.kernel_size-1)/2)):
                weight_pad[t] = zero_pad(guide[:, t-1:t, :, :])
            else:
                weight_pad[t] = zero_pad(guide_mid)

        guide_weight = torch.cat([weight_pad[t] for t in range(self.kernel_size*self.kernel_size)], dim=1)
        return guide_weight

class CSPN(nn.Module):
  def __init__(self, kernel_size):
      super(CSPN, self).__init__()
      self.kernel_size = kernel_size

  def forward(self, guide_weight, hn, h0):

        #CSPN
        half = int(0.5 * (self.kernel_size * self.kernel_size - 1))
        result_pad = [i for i in range(self.kernel_size * self.kernel_size)]
        for t in range(self.kernel_size*self.kernel_size):
            zero_pad = 0
            if(self.kernel_size==3):
                zero_pad = pad2[t]
            elif(self.kernel_size==5):
                zero_pad = pad[t]
            elif(self.kernel_size==7):
                zero_pad = pad3[t]
            if(t == half):
                result_pad[t] = zero_pad(h0)
            else:
                result_pad[t] = zero_pad(hn)
        guide_result = torch.cat([result_pad[t] for t in range(self.kernel_size*self.kernel_size)], dim=1)
        #guide_result = torch.cat([result0_pad, result1_pad, result2_pad, result3_pad,result4_pad, result5_pad, result6_pad, result7_pad, result8_pad], 1)

        guide_result = torch.sum((guide_weight.mul(guide_result)), dim=1)
        guide_result = guide_result[:, int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2), int((self.kernel_size-1)/2):-int((self.kernel_size-1)/2)]

        return guide_result.unsqueeze(dim=1)

class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):

        guide = self.generate(feature)

        #normalization in standard CSPN
        #'''
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        #'''
        #weight_pad = [i for i in range(self.kernel_size * self.kernel_size)]

        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output

def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size-1)/2))
    return kernel

class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, kernel, input, input0): #with standard CSPN, an addition input0 port is added
        bs = input.size()[0]
        h, w = input.size()[2], input.size()[3]
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)

        # standard CSPN
        input0 = input0.view(bs, 1, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+1, :] = input0

        #print(input_im2col.size(), kernel.size())
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)

class GeometryFeature(nn.Module):
    def __init__(self):
        super(GeometryFeature, self).__init__()

    def forward(self, z, vnorm, unorm, h, w, ch, cw, fh, fw):
        x = z*(0.5*h*(vnorm+1)-ch)/fh
        y = z*(0.5*w*(unorm+1)-cw)/fw
        return torch.cat((x, y, z),1)

class BasicBlockGeo(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, geoplanes=3):
        super(BasicBlockGeo, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            #norm_layer = encoding.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes + geoplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes+geoplanes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes+geoplanes, planes, stride),
                norm_layer(planes),
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, g1=None, g2=None):
        identity = x
        if g1 is not None:
            x = torch.cat((x, g1), 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if g2 is not None:
            out = torch.cat((g2,out), 1)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




def grid_generator(k, r, n):
    """grid_generator
    Parameters
    ---------
    f : filter_size, int
    k: kernel_size, int
    n: number of grid, int
    Returns
    -------
    torch.Tensor. shape = (n, 2, k, k)
    """
    grid_x, grid_y = torch.meshgrid([torch.linspace(k//2, k//2+r-1, steps=r),
                                     torch.linspace(k//2, k//2+r-1, steps=r)])
    grid = torch.stack([grid_x, grid_y], 2).view(r, r, 2)

    return grid.unsqueeze(0).repeat(n, 1, 1, 1).cuda()
to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()
# **********************************************************************************************************************
# The following is the ENet backbone from the paper PENet: https://github.com/JUGGHM/PENet_ICRA2021
# In ReDC, as mentioned in the paper, we study our deformanle refinement module on top of ENet.
class ENet(nn.Module):
    def __init__(self, args):
        super(ENet, self).__init__()
        self.args = args
        self.geofeature = None
        self.geoplanes = 3
        if self.args.convolutional_layer_encoding == "xyz":
            self.geofeature = GeometryFeature()
        elif self.args.convolutional_layer_encoding == "std":
            self.geoplanes = 0
        elif self.args.convolutional_layer_encoding == "uv":
            self.geoplanes = 2
        elif self.args.convolutional_layer_encoding == "z":
            self.geoplanes = 1
        position = CoordConv.AddCoordsNp(320, 320)
        # print(position.shape)
        self.position = position.call()
        self.position = to_float_tensor(self.position)
        self.position = self.position.unsqueeze(0)
        self.position = self.position.cuda()
        # rgb encoder
        self.rgb_conv_init = convbnrelu(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.rgb_encoder_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer3 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer5 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer7 = BasicBlockGeo(inplanes=256, planes=512, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.rgb_encoder_layer9 = BasicBlockGeo(inplanes=512, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.rgb_encoder_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        self.rgb_decoder_layer8 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer6 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer4 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer2 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_layer0 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.rgb_decoder_output = deconvbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, output_padding=0)


        # depth encoder
        self.depth_conv_init = convbnrelu(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)

        self.depth_layer1 = BasicBlockGeo(inplanes=32, planes=64, stride=2, geoplanes=self.geoplanes)
        self.depth_layer2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.depth_layer3 = BasicBlockGeo(inplanes=128, planes=128, stride=2, geoplanes=self.geoplanes)
        self.depth_layer4 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.depth_layer5 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.depth_layer6 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.depth_layer7 = BasicBlockGeo(inplanes=512, planes=512, stride=2, geoplanes=self.geoplanes)
        self.depth_layer8 = BasicBlockGeo(inplanes=512, planes=512, stride=1, geoplanes=self.geoplanes)
        self.depth_layer9 = BasicBlockGeo(inplanes=1024, planes=1024, stride=2, geoplanes=self.geoplanes)
        self.depth_layer10 = BasicBlockGeo(inplanes=1024, planes=1024, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.decoder_layer1 = deconvbnrelu(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer2 = deconvbnrelu(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer3 = deconvbnrelu(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer4 = deconvbnrelu(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.decoder_layer5 = deconvbnrelu(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.decoder_layer6 = convbnrelu(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)
        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        weights_init(self)

    def forward(self, rgb, d, K):
        #independent input
        # rgb = input['rgb']
        # d = input['d']
        position = self.position
        # K= self.K
        K = K
        # position = input['position']
        # K = input['K']
        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]

        f352 = K[:, 1, 1]
        f352 = f352.unsqueeze(1)
        f352 = f352.unsqueeze(2)
        f352 = f352.unsqueeze(3)
        c352 = K[:, 1, 2]
        c352 = c352.unsqueeze(1)
        c352 = c352.unsqueeze(2)
        c352 = c352.unsqueeze(3)
        f1216 = K[:, 0, 0]
        f1216 = f1216.unsqueeze(1)
        f1216 = f1216.unsqueeze(2)
        f1216 = f1216.unsqueeze(3)
        c1216 = K[:, 0, 2]
        c1216 = c1216.unsqueeze(1)
        c1216 = c1216.unsqueeze(2)
        c1216 = c1216.unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None

        if self.args.convolutional_layer_encoding == "xyz":
            geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352 / 2, 1216 / 2, c352, c1216, f352, f1216)
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352 / 4, 1216 / 4, c352, c1216, f352, f1216)
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352 / 8, 1216 / 8, c352, c1216, f352, f1216)
            geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352 / 16, 1216 / 16, c352, c1216, f352, f1216)
            geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352 / 32, 1216 / 32, c352, c1216, f352, f1216)
        elif self.args.convolutional_layer_encoding == "uv":
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
            geo_s5 = torch.cat((vnorm_s5, unorm_s5), dim=1)
            geo_s6 = torch.cat((vnorm_s6, unorm_s6), dim=1)
        elif self.args.convolutional_layer_encoding == "z":
            geo_s1 = d
            geo_s2 = d_s2
            geo_s3 = d_s3
            geo_s4 = d_s4
            geo_s5 = d_s5
            geo_s6 = d_s6

        # b 1 352 1216
        rgb_feature = self.rgb_conv_init(torch.cat((rgb, d), dim=1))
        rgb_feature1 = self.rgb_encoder_layer1(rgb_feature, geo_s1, geo_s2) # b 32 176 608
        rgb_feature2 = self.rgb_encoder_layer2(rgb_feature1, geo_s2, geo_s2) # b 32 176 608
        rgb_feature3 = self.rgb_encoder_layer3(rgb_feature2, geo_s2, geo_s3) # b 64 88 304
        rgb_feature4 = self.rgb_encoder_layer4(rgb_feature3, geo_s3, geo_s3) # b 64 88 304
        rgb_feature5 = self.rgb_encoder_layer5(rgb_feature4, geo_s3, geo_s4) # b 128 44 152
        rgb_feature6 = self.rgb_encoder_layer6(rgb_feature5, geo_s4, geo_s4) # b 128 44 152
        rgb_feature7 = self.rgb_encoder_layer7(rgb_feature6, geo_s4, geo_s5) # b 256 22 76
        rgb_feature8 = self.rgb_encoder_layer8(rgb_feature7, geo_s5, geo_s5) # b 256 22 76
        rgb_feature9 = self.rgb_encoder_layer9(rgb_feature8, geo_s5, geo_s6) # b 512 11 38
        rgb_feature10 = self.rgb_encoder_layer10(rgb_feature9, geo_s6, geo_s6) # b 512 11 38

        rgb_feature_decoder8 = self.rgb_decoder_layer8(rgb_feature10)
        rgb_feature8_plus = rgb_feature_decoder8 + rgb_feature8

        rgb_feature_decoder6 = self.rgb_decoder_layer6(rgb_feature8_plus)
        rgb_feature6_plus = rgb_feature_decoder6 + rgb_feature6

        rgb_feature_decoder4 = self.rgb_decoder_layer4(rgb_feature6_plus)
        rgb_feature4_plus = rgb_feature_decoder4 + rgb_feature4

        rgb_feature_decoder2 = self.rgb_decoder_layer2(rgb_feature4_plus)
        rgb_feature2_plus = rgb_feature_decoder2 + rgb_feature2   # b 32 176 608

        rgb_feature_decoder0 = self.rgb_decoder_layer0(rgb_feature2_plus)
        rgb_feature0_plus = rgb_feature_decoder0 + rgb_feature

        rgb_output = self.rgb_decoder_output(rgb_feature0_plus)
        rgb_depth = rgb_output[:, 0:1, :, :]
        rgb_conf = rgb_output[:, 1:2, :, :]

        # -----------------------------------------------------------------------
        # mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        # input = torch.cat([d, mask], 1)

        sparsed_feature = self.depth_conv_init(torch.cat((d, rgb_depth), dim=1))
        sparsed_feature1 = self.depth_layer1(sparsed_feature, geo_s1, geo_s2)# b 32 176 608
        sparsed_feature2 = self.depth_layer2(sparsed_feature1, geo_s2, geo_s2) # b 32 176 608

        sparsed_feature2_plus = torch.cat([rgb_feature2_plus, sparsed_feature2], 1)
        sparsed_feature3 = self.depth_layer3(sparsed_feature2_plus, geo_s2, geo_s3) # b 64 88 304
        sparsed_feature4 = self.depth_layer4(sparsed_feature3, geo_s3, geo_s3) # b 64 88 304

        sparsed_feature4_plus = torch.cat([rgb_feature4_plus, sparsed_feature4], 1)
        sparsed_feature5 = self.depth_layer5(sparsed_feature4_plus, geo_s3, geo_s4) # b 128 44 152
        sparsed_feature6 = self.depth_layer6(sparsed_feature5, geo_s4, geo_s4) # b 128 44 152

        sparsed_feature6_plus = torch.cat([rgb_feature6_plus, sparsed_feature6], 1)
        sparsed_feature7 = self.depth_layer7(sparsed_feature6_plus, geo_s4, geo_s5) # b 256 22 76
        sparsed_feature8 = self.depth_layer8(sparsed_feature7, geo_s5, geo_s5) # b 256 22 76

        sparsed_feature8_plus = torch.cat([rgb_feature8_plus, sparsed_feature8], 1)
        sparsed_feature9 = self.depth_layer9(sparsed_feature8_plus, geo_s5, geo_s6) # b 512 11 38
        sparsed_feature10 = self.depth_layer10(sparsed_feature9, geo_s6, geo_s6) # b 512 11 38

        # -----------------------------------------------------------------------------------------

        fusion1 = rgb_feature10 + sparsed_feature10
        decoder_feature1 = self.decoder_layer1(fusion1)

        fusion2 = sparsed_feature8 + decoder_feature1
        decoder_feature2 = self.decoder_layer2(fusion2)

        fusion3 = sparsed_feature6 + decoder_feature2
        decoder_feature3 = self.decoder_layer3(fusion3)

        fusion4 = sparsed_feature4 + decoder_feature3
        decoder_feature4 = self.decoder_layer4(fusion4)

        fusion5 = sparsed_feature2 + decoder_feature4
        decoder_feature5 = self.decoder_layer5(fusion5)

        depth_output = self.decoder_layer6(decoder_feature5)
        d_depth, d_conf = torch.chunk(depth_output, 2, dim=1)

        rgb_conf, d_conf = torch.chunk(self.softmax(torch.cat((rgb_conf, d_conf), dim=1)), 2, dim=1)
        output = rgb_conf*rgb_depth + d_conf*d_depth
        
        return torch.cat((rgb_feature0_plus, decoder_feature5), 1), output
# **********************************************************************************************************************

    
class ReDC(nn.Module):
    def __init__(self, args):
        super(ReDC, self).__init__()
        self.kernel_size = 3
        self.filter_size = 15
        self.residual = True
        
        self.backbone = ENet(args)
        self.dkn_weight = convbnrelu(in_channels=64, out_channels=self.kernel_size**2, kernel_size=3, stride=1, padding=1)
        self.dkn_offset = convbn(in_channels=64, out_channels=2*self.kernel_size**2, kernel_size=3, stride=1, padding=1)
        
    def DKN_Interpolate(self, image, depth, weight, offset):
        if self.residual:
            weight = weight - torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight = weight / torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        h, w = image.size(2), image.size(3)
        b = image.size(0)
        k = self.filter_size
        r = self.kernel_size
        hw = h*w

        offset = offset.permute(0, 2, 3, 1).contiguous().view(b*hw, r, r, 2)
        # (b, r**2, h, w) -> (b, h, w, r**2) -> (b*hw, r**2, 1)
        weight = weight.permute(0, 2, 3, 1).contiguous().view(b*hw, r*r, 1)

        # (b*hw, r, r, 2)
        grid = grid_generator(k, r, b*hw)

        coord = grid + offset
        coord = (coord / k * 2) - 1

        # (b, k**2, hw) -> (b*hw, 1, k, k)
        depth_col = F.unfold(depth, k, padding=k//2).permute(0, 2, 1).contiguous().view(b*hw, 1, k, k)

        # (b*hw, 1, k, k), (b*hw, r, r, 2) => (b*hw, 1, r^2)
        depth_sampled = F.grid_sample(
            depth_col.float(), coord).view(b*hw, 1, -1)

        # (b*w*h, 1, r^2) x (b*w*h, r^2, 1) => (b, 1, h,w)
        out = torch.bmm(depth_sampled, weight).view(b, 1, h, w).float()

        if self.residual:
            out += depth.float()

        return out

    def forward(self, rgb, d, K):
        features, coarse_depth = self.backbone(rgb, d, K)
        weight = torch.sigmoid(self.dkn_weight(features))
        offset = self.dkn_offset(features)
        refined_depth = self.DKN_Interpolate(rgb, coarse_depth, weight, offset)
        return refined_depth