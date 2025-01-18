# append module path
import sys
import io
import datetime
import os

import copy
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
from PIL import Image
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = '4'



from src import str2bool
from application.application_utils import RGBPReader
import torch
from torch.backends import cudnn
from tqdm import tqdm
import numpy as np
import glob
import sys

import torch
import numpy as np
import cv2
from scipy.ndimage import label


# def fill_in_fast_tensor(depth_tensor, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
#                         extrapolate=False, blur_type='bilateral'):
#     """Fast, in-place depth completion for tensor.

#     Args:
#         depth_tensor: projected depths tensor with shape [b, c=1, h, w]
#         max_depth: max depth value for inversion
#         custom_kernel: kernel to apply initial dilation
#         extrapolate: whether to extrapolate by extending depths to top of
#             the frame, and applying a 31x31 full kernel dilation
#         blur_type:
#             'bilateral' - preserves local structure (recommended)
#             'gaussian' - provides lower RMSE

#     Returns:
#         depth_tensor: dense depth tensor with same shape [b, c=1, h, w]
#     """
#     depth_tensor_np = depth_tensor.squeeze(1).cpu().numpy()  # Convert to numpy array with shape [b, h, w]

#     for i in range(depth_tensor_np.shape[0]):  # Iterate over the batch
#         depth_map = depth_tensor_np[i]

#         valid_pixels = (depth_map > 0.1)
#         depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

#         depth_map = cv2.dilate(depth_map, custom_kernel)

#         depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

#         empty_pixels = (depth_map < 0.1)
#         dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
#         depth_map[empty_pixels] = dilated[empty_pixels]

#         if extrapolate:
#             top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
#             top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

#             for pixel_col_idx in range(depth_map.shape[1]):
#                 depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
#                     top_pixel_values[pixel_col_idx]

#         empty_pixels = depth_map < 0.1
#         dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
#         depth_map[empty_pixels] = dilated[empty_pixels]

#         depth_map = depth_map.astype('float32')
#         depth_map = cv2.medianBlur(depth_map, 5)
#         depth_map = depth_map.astype('float64')

#         if blur_type == 'bilateral':
#             depth_map = depth_map.astype('float32')
#             depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
#             depth_map = depth_map.astype('float64')
#         elif blur_type == 'gaussian':
#             valid_pixels = (depth_map > 0.1)
#             blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
#             depth_map[valid_pixels] = blurred[valid_pixels]

#         valid_pixels = (depth_map > 0.1)
#         depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

#         mask = (depth_map <= 0.1)
#         if np.sum(mask) != 0:
#             labeled_array, num_features = label(mask)
#             for j in range(num_features):
#                 index = j + 1
#                 m = (labeled_array == index)
#                 m_dilate1 = cv2.dilate(1.0 * m, FULL_KERNEL_7)
#                 m_dilate2 = cv2.dilate(1.0 * m, FULL_KERNEL_13)
#                 m_diff = m_dilate2 - m_dilate1
#                 v = np.mean(depth_map[m_diff > 0])
#                 depth_map = np.ma.array(depth_map, mask=m_dilate1, fill_value=v)
#                 depth_map = depth_map.filled()
#                 depth_map = np.array(depth_map)

#         depth_tensor_np[i] = depth_map

#     depth_tensor_filled = torch.from_numpy(depth_tensor_np).unsqueeze(1).to(depth_tensor.device)

#     return depth_tensor_filled




def on_load_checkpoint(checkpoint):
    keys_list = list(checkpoint['network_state_dict'].keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint['network_state_dict'][deal_key] = checkpoint['network_state_dict'][key]
            del checkpoint['network_state_dict'][key]
    return checkpoint

def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for AbsRel_depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        action="store",
        type=lambda x: Path(x),
        default=r'/data1/Chenbingyuan/Depth-Completion/g2_dataset/DIODE',
        help="Path to test dataset",
        required=False
    )
    parser.add_argument(
        "--ReZero",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the ReZero",
        required=False,
    )
    parser.add_argument(
        "--method",
        action="store",
        type=lambda x: Path(x),
        default='',
        help="Path to load models",
        required=False
    )
    parser.add_argument(
        "--swin",
        action="store",
        type=str2bool,
        default=False,
        help="whether to use the Swin Transformer, if false we will use ViT",
        required=False,
    )

    args = parser.parse_args()
    return args





def pred_and_save(network,rgb, point, hole_point, out_path, network_type, desc):
    # total = sum([param.nelement() for param in network.parameters()])
    # print(network_type)
    # print('  + Number of params: %.2fM' % (total / 1e6))
    KITTI_factor = 80.
    if 'JARRN_nosfp_direct_2branch_DIODE_HRWSI' == network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    if 'JARRN' in network_type or 'DIODE' in network_type:
        # SfV2  G2_Mono
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'sfv2' in network_type or 'DIODE' in network_type:
        # SfV2  G2_Mono
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'sf_G2V2' in network_type:
        # SfV2  G2_Mono
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'G2' in network_type or 'g2' in network_type:
        # SfV2  G2_Mono
        gen_depth, = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'CFormer_KITTI' in network_type and 'DIODE' not in network_type:
        # CFormer，记得更改其他CFormer相关
        point = point * KITTI_factor #  KITTI
        # point = point * 25. #  nyu
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth['pred'].detach()
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        # gen_depth = (gen_depth / 25.)  # nyu
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'PEnet' in network_type and 'DIODE_HRWSI' not in network_type:           
        # PENET
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        point = point * KITTI_factor #  KITTI
        gen_depth = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'Penet_DIODE_HRWSI' in network_type:
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'BPnet' in network_type and 'DIODE_HRWSI' not in network_type:           
        # PENET
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        point = point * KITTI_factor #  KITTI
        gen_depth = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth[-1]
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'ReDC' in network_type and 'DIODE_HRWSI' not in network_type:          
        # ReDC
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        point = point * KITTI_factor #  KITTI
        gen_depth = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'ReDC_DIODE_HRWSI' in network_type:          
        # ReDC
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'SemAttNet' in network_type:    
        # SemAttNet
        point = point * KITTI_factor #  KITTI
        seg_path = rgb_path.replace('_rgb/', '_seg/pred/')
        seg_img = torch.from_numpy(np.expand_dims(np.array(Image.open(seg_path), dtype=float) / 255., axis=0))
        seg_img = seg_img.unsqueeze(0).to(torch.float32)
        seg_img = torch.cat((seg_img, seg_img, seg_img), dim=1)
        # 减少缓存
        seg_img = seg_img.to(torch.float32)
        rgb = rgb.to(torch.float32)
        point = point.to(torch.float32)
        _, _, _, _, _, _,_, gen_depth = network(rgb.cuda(), point.cuda(), seg_img.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'ACMNet' in network_type:               
        # ACMNet
        point = point * KITTI_factor * 255. #  KITTI
        if desc == 'nyu':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'DIODE':
            fx, fy, cx, cy = 886.81, 927.06, 512, 384
            # fx, fy, cx, cy = 1, 1, 0, 0
        elif desc == 'redweb':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'HRWSI':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'ETH3D':
            fx, fy, cx, cy = 3429.76, 3429.06, 3117.98, 2061.68
        elif desc == 'Ibims':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'KITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'VKITTI':
            fx, fy, cx, cy = 984.2439, 980.8141, 690.0, 233.1966
        elif desc == 'Matterport3D':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        elif desc == 'UnrealCV':
            fx, fy, cx, cy = 582.6244, 582.6910, 313.0447, 238.4438
        K = torch.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = K.unsqueeze(0)
        gen_depth = network.netDC(point.cuda(),rgb.cuda(), K) 
        gen_depth = gen_depth[0]
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'TWISE' in network_type and 'DIODE_HRWSI' not in network_type:          
        # TWISE
        from src.baselines.TWISE.utils import smooth2chandep
        gen_depth,_,_ = network(point.cuda(),rgb.cuda()) 
        gen_depth = smooth2chandep(gen_depth, params={'depth_maxrange': 1.0,}, device=None)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'TWISE_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'GuideNet' in network_type and 'DIODE_HRWSI' not in network_type:         
        # GuideNet
        point = point * KITTI_factor
        gen_depth, = network(rgb.cuda(), point.cuda())
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'GuideNet_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'NLSPN_KITTI' in network_type and 'DIODE' not in network_type:   
        # NLSPN
        point = point * KITTI_factor #  KITTI
        # point = point * 25. #  NYU
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        # gen_depth = (gen_depth / 25.)  # NYU
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'SDCM' in network_type and 'DIODE_HRWSI' not in network_type:      
        # SDCM
        point = point * KITTI_factor #  KITTI
        import torchvision.transforms as transforms
        grayscale_transform = transforms.Grayscale(num_output_channels=1)  # 创建一个灰度转换器
        rgb = grayscale_transform(rgb)  # 使用转换器将RGB图像转换为灰度图像
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'SDCM_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'MDAnet' in network_type and 'DIODE_HRWSI' not in network_type:            
        # MDAnet
        point = point * KITTI_factor  # 网络需要
        gen_depth = network(point.cuda(),rgb.cuda())
        gen_depth = gen_depth[0]  
        gen_depth = (gen_depth / KITTI_factor)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'MDAnet_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'EMDC' in network_type and 'DIODE_HRWSI' not in network_type:            
        # EMDC
        point = point * 25.  # 网络需要
        gen_depth = network(rgb.cuda(),point.cuda())
        gen_depth = gen_depth[0]
        gen_depth = (gen_depth / 25.)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'EMDC_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'LRRU' in network_type and 'DIODE_HRWSI' not in network_type:            
        # LRRU
        from LRRU_utilis import fill_in_fast_tensor, outlier_removal
        point = point * KITTI_factor #  KITTI
        fill_depth = fill_in_fast_tensor(point, max_depth=KITTI_factor)
        
        dep_np = point.numpy().squeeze(0)
        dep_clear, _ = outlier_removal(dep_np)
        dep_clear = np.expand_dims(dep_clear, 0)
        dep_clear_torch = torch.from_numpy(dep_clear)
        
        sample = {'dep':point.cuda(), 'rgb':rgb.cuda(), 'ip':fill_depth.cuda(), 'dep_clear':dep_clear_torch.cuda()}
        gen_depth = network(sample)  
        gen_depth = gen_depth['results'][-1]
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / KITTI_factor)  # KITTI
        
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'LRRU_DIODE_HRWSI' in network_type:
        gen_depth,_,_,_ = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
                      
    # # save img
    if not os.path.exists(str(Path(out_path).parent)):
        os.makedirs(str(Path(out_path).parent))
    depth_pil = Image.fromarray(depth.astype('uint8'))
    depth_pil.save(out_path)

def demo(args, network, pro, mode, network_type):
    base_path = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/'
    dataset_dict = {base_path+'nyu/val':'nyu', base_path+'DIODE/val':'DIODE', base_path+'HRWSI/val':'HRWSI',
                    base_path+'ETH3D/val':'ETH3D', base_path+'Ibims/val':'Ibims', base_path+'redweb/val':'redweb',
                    base_path+'KITTI/val':'KITTI', base_path+'VKITTI/val':'VKITTI', base_path+'Matterport3D/val':'Matterport3D',
                    base_path+'UnrealCV/val':'UnrealCV',}
    network.eval()
    with torch.no_grad():
        print(f'args.rgbd_dir={args.rgbd_dir}')
        
        jpg_list = ['HRWSI','DIODE','nyu','redweb']
        desc = dataset_dict[args.rgbd_dir]
        print(f'desc = {desc}')
        if desc in jpg_list:
            jpg_files = glob.glob(args.rgbd_dir + '/**/*.jpg', recursive=True)
            for file in tqdm(jpg_files, desc=desc):
                str_file = file
                if '_rgb' in str_file:
                    if 'dis.jpg' not in str_file: continue
                    rgb_path = str_file
                    if mode == 'result':
                        point_path = rgb_path.replace(
                            'val', 'point/point_' + str(pro) + '/', 1)
                    elif mode == 'result_nb':
                        point_path = rgb_path.replace(
                            'val', 'point/point_nb_' + str (pro) + '/', 1)
                    elif mode == 'lines_result':
                        point_path = rgb_path.replace(
                            'val', 'lines/line_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse_same_seg':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_same_seg_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse_differ_seg':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_differ_seg_' + str(pro) + '/', 1)
                    point_path = point_path.replace('dis', '')
                    point_path = point_path.replace('rgb', 'gt')
                    point_path = point_path.replace('jpg', 'png')
                    
                    out_dir = os.path.join(mode, args.method + '_' + str(pro))
                    out_path = rgb_path.replace('val',  out_dir , 1)
                    out_path = out_path.replace('dis', '')
                    out_path = out_path.replace('jpg', 'png')
                    out_path = out_path.replace('rgb', 'gt')
                    
                    rgbd_reader = RGBPReader()
                    rgb, point, hole_point = rgbd_reader.read_rgbp(
                        rgb_path, point_path)
                    pred_and_save(network,rgb,point,hole_point,out_path,network_type,desc)
        else:
            png_files = glob.glob(args.rgbd_dir + '/**/*.png', recursive=True)
            for file in tqdm(png_files, desc=desc):
                # print(file)
                str_file = file
                if '_rgb' in str_file:
                    if 'dis.png' not in str_file: continue
                    rgb_path = str_file
                    if mode == 'result':
                        point_path = rgb_path.replace(
                            'val', 'point/point_' + str(pro) + '/', 1)
                    elif mode == 'result_nb':
                        point_path = rgb_path.replace(
                            'val', 'point/point_nb_' + str(pro) + '/', 1)
                    elif mode == 'lines_result':
                        point_path = rgb_path.replace(
                            'val', 'lines/line_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse_same_seg':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_same_seg_' + str(pro) + '/', 1)
                    elif mode == 'result_very_sparse_differ_seg':
                        point_path = rgb_path.replace(
                            'val', 'point/very_sparse_point_differ_seg_' + str(pro) + '/', 1)
                    point_path = point_path.replace('dis', '')
                    point_path = point_path.replace('rgb', 'gt')
                    
                    out_dir = os.path.join(mode, args.method + '_' + str(pro))
                    out_path = rgb_path.replace('val',  out_dir , 1)
                    out_path = out_path.replace('dis', '')
                    out_path = out_path.replace('rgb', 'gt')
                    
                    rgbd_reader = RGBPReader()
                    rgb, point, hole_point = rgbd_reader.read_rgbp(
                        rgb_path, point_path)    
                    pred_and_save(network,rgb,point,hole_point,out_path,network_type,desc)
        
        
            
def depth_inference():
    print('Start depth inference')
    args = parse_arguments()
    for every_rgbd_dir in rgbd_dir:
        args.rgbd_dir = every_rgbd_dir
        for method in method_list:
            print(f'Now dealing with {method}')
            for epoch in epoch_list:
                epoch = str(epoch) 
                args.method = method
                print(method)
                # load parameters
                if 'rz' not in method:
                    args.ReZero = False
                if method == 'rz_sb_mar_JARRN_G2V2_full':
                    from sfv2_networks import JARRN_G2V2
                    network = JARRN_G2V2()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_JARRN_full_G2V2/models/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_JARRN_100LiDAR':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_100LiDAR/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_JARRN_70LiDAR':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_70LiDAR/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
                elif method == 'rz_sb_mar_JARRN_60LiDAR':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_60LiDAR/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI':
                    from sfv2_networks import JARRN_nosfp_direct_2branch
                    network = JARRN_nosfp_direct_2branch(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2':
                    from sfv2_networks import JARRN_nosfp_direct_2branch_2
                    network = JARRN_nosfp_direct_2branch_2(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_JARRN_full_05line_05point':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_full_05line_05point/models/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                
                elif method == 'rz_sb_mar_JARRN_nosoftmax':
                    from sfv2_networks import JARRN_nosoftmax
                    network = JARRN_nosoftmax(rezero=args.ReZero)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar/JARRN_nosoftmax/models/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN_noSoftmax
                elif method == 'rz_sb_mar_JARRN_mixed_07point_03line':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_0.3line_0.7point/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
                elif method == 'rz_sb_mar_JARRN_mixed_1point_1line':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_1line_1point/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
                elif method == 'rz_sb_mar_JARRN_mixed_line_point':
                    # 只是用采点方式，不改变GT
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_line_point/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) #  JARRN
                elif method == 'rz_sb_mar_JARRN_mixed_05point_05line':
                    # 只是用采点方式，不改变GT
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_0.5line_0.5point_fixed/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) #  JARRN
                
                
                elif method == 'rz_sb_mar_JARRN':
                    from sfv2_networks import JARRN
                    network = JARRN(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_3/models/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
                elif method == 'rz_sb_mar_g2_all_retrain':
                    from sfv2_networks import G2_Mono
                    network = G2_Mono(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_G2Mono/models/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
                elif method == 'rz_sb_mar_sfv2_KITTI_2':
                    from sfv2_networks import sfv2_UNet_KITTI
                    network = sfv2_UNet_KITTI(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_KITTI_2/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI':
                    from sfv2_networks import sfv2_UNet
                    network = sfv2_UNet(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_LSM':
                    from sfv2_networks import sfv2_UNet_LSM
                    network = sfv2_UNet_LSM(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f22':
                    from sfv2_networks import sfv2_UNet_f22
                    network = sfv2_UNet_f22(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f22/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f11':
                    from sfv2_networks import sfv2_UNet_f11
                    network = sfv2_UNet_f11(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f11/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f0505':
                    from sfv2_networks import sfv2_UNet_f0505
                    network = sfv2_UNet_f0505(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f0505/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_s005':
                    from sfv2_networks import sfv2_UNet_s005
                    network = sfv2_UNet_s005(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_s005/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_s05':
                    from sfv2_networks import sfv2_UNet_s05
                    network = sfv2_UNet_s05(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_s05/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_relative_loss':
                    from sfv2_networks import sfv2_UNet
                    network = sfv2_UNet(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_relative/models/epoch_24.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_tiny':
                    from sfv2_networks import sfv2_UNet_tiny
                    network = sfv2_UNet_tiny(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Tiny/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_small':
                    from sfv2_networks import sfv2_UNet_small
                    network = sfv2_UNet_small(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Small/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_large':
                    from sfv2_networks import sfv2_UNet_large
                    network = sfv2_UNet_large(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Large/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_unscale':
                    from sfv2_networks import sfv2_UNet_unscale
                    network = sfv2_UNet_unscale(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'bn_sb_mar_sfv2_DIODE_HRWSI':
                    from sfv2_networks import sfv2_UNet_bn
                    network = sfv2_UNet_bn(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_bn_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_bn':
                    from sfv2_networks import sfv2_UNet_bn
                    network = sfv2_UNet_bn(rezero=False)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_bn_retrain0402/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_f':
                    from sfv2_networks import sfv2_UNet_no_f
                    network = sfv2_UNet_no_f(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_f/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_s':
                    from sfv2_networks import sfv2_UNet_no_s
                    network = sfv2_UNet_no_s(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_s/models/epoch_58.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_p':
                    from sfv2_networks import sfv2_UNet_no_p
                    network = sfv2_UNet_no_p(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_p/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_f':
                    from sfv2_networks import sfv2_UNet_only_f
                    network = sfv2_UNet_only_f(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_f/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_s':
                    from sfv2_networks import sfv2_UNet_only_s
                    network = sfv2_UNet_only_s(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_s/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_p':
                    from sfv2_networks import sfv2_UNet_only_p
                    network = sfv2_UNet_only_p(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_p/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
                elif method == 'rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI':
                    from sfv2_networks import sfv2_UNet
                    network = sfv2_UNet(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_blur':
                    from sfv2_networks import sfv2_UNet
                    network = sfv2_UNet(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_blur/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
                elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_2':
                    from sfv2_networks import sfv2_UNet
                    network = sfv2_UNet(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_2/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_CFormer_DIODE_HRWSI_2' or method == 'rz_sb_mar_CFormer_DIODE_HRWSI':
                    from sfv2_networks import CFormer_DIODE_HRWSI
                    network = CFormer_DIODE_HRWSI()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_CFormer_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_NLSPN_DIODE_HRWSI_60' or method == 'rz_sb_mar_NLSPN_DIODE_HRWSI':
                    from sfv2_networks import NLSPN_DIODE_HRWSI
                    network = NLSPN_DIODE_HRWSI()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_NLSPN_DIODE_HRWSI_60/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_g2_DIODE_HRWSI':
                    from sfv2_networks import g2_UNet
                    network = g2_UNet()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_g2_DIODE_HRWSI/models/epoch_60.pth'
                    network = network.cuda()
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                if method == 'rz_sb_mar_G2_Mono':
                    # G2_Mono
                    from sfv2_networks import G2_Mono
                    network = G2_Mono(rezero=args.ReZero)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/G2_Mono/epoch_100.pth'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['network'],strict=True) 
                elif method == 'rz_sb_mar_CFormer_KITTI':
                    # CFormer
                    from src.baselines.CFormer.model.completionformer import CompletionFormer, check_args
                    from src.baselines.CFormer.model.config import args as args_cformer
                    network = CompletionFormer(args_cformer)  
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/CFormer/model/KITTIDC_L1L2.pt'
                    # model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/CFormer/model/NYUv2.pt'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'],strict=True)  # CFormer
                elif method == 'rz_sb_mar_PEnet':
                    # PEnet
                    from src.baselines.PEnet.model.model import PENet_C2
                    from src.baselines.PEnet.model.config import args as args_penet
                    network = PENet_C2(args_penet) 
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/PEnet/model/pe.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'],strict=False)  # PEnet
                    
                elif method == 'rz_sb_mar_PEnet_DIODE_HRWSI':
                    from sfv2_networks import PEnet_retrain
                    network = PEnet_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_PEnet_DIODE_HRWSI/models/redc_epoch_60.pth'
                elif method == 'rz_sb_mar_ReDC':
                    # PEnet
                    from src.baselines.ReDC.redc import ReDC
                    from src.baselines.ReDC.config import args as args_ReDC
                    network = ReDC(args_ReDC) 
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/ReDC/refinement_model_best.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'],strict=False)  # PEnet
                elif method == 'rz_sb_mar_ReDC_DIODE_HRWSI':
                    from sfv2_networks import ReDC_retrain
                    network = ReDC_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_redc_DIODE_HRWSI/models/redc_epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_SemAttNet':
                    from src.baselines.SemAttNet.model import A_CSPN_plus_plus
                    from src.baselines.SemAttNet.config import args as  args_SemAttNet
                    network = A_CSPN_plus_plus(args_SemAttNet) 
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/SemAttNet/model_best_backup.pth.tar'
                elif method == 'rz_sb_mar_ACMNet':

                    from src.baselines.ACMNet.options.test_options import TestOptions
                    from src.baselines.ACMNet.models.test_model import TESTModel
                    network = TESTModel() 
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/ACMNet/model_64.pth'
                    opt = TestOptions().parse()
                    network.initialize(opt)
                    network.setup(opt)
                elif method == 'rz_sb_mar_TWISE':
                    # TWISE
                    from src.baselines.TWISE.model import MultiRes_network_avgpool_diffspatialsizes
                    from src.baselines.TWISE.utils import smooth2chandep
                    sys.path.append('/data1/Chenbingyuan/Depth-Completion/src/baselines/TWISE')
                    import src.baselines.TWISE.metrics
                    
                    network = MultiRes_network_avgpool_diffspatialsizes()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/TWISE/TWISE_gamma2.5/model_best.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'])  # TWISE
                elif method == 'rz_sb_mar_TWISE_DIODE_HRWSI':
                    from sfv2_networks import TWISE_retrain
                    network = TWISE_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_TWISE_DIODE_HRWSI/models/epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_GuideNet':
                    # GuideNet
                    from src.baselines.GuideNet.utils import init_net, resume_state
                    import yaml
                    from easydict import EasyDict as edict
                    with open('/data1/Chenbingyuan/Depth-Completion/src/baselines/GuideNet/configs/GNS.yaml', 'r') as file:
                        config_data = yaml.load(file, Loader=yaml.FullLoader)
                    GuideNetconfig = edict(config_data)
                    key, params = GuideNetconfig.data_config.popitem()

                    network = init_net(GuideNetconfig)
                    network = torch.nn.DataParallel(network,device_ids=[0])
                    network = resume_state(GuideNetconfig, network)
                elif method == 'rz_sb_mar_GuideNet_DIODE_HRWSI':
                    from sfv2_networks import GuideNet_retrain
                    network = GuideNet_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_GuideNet_DIODE_HRWSI/models/epoch_60.pth'
                elif method == 'rz_sb_mar_NLSPN_KITTI':
                    # NLSPN
                    from src.baselines.NLSPN.src.model.nlspnmodel import NLSPNModel
                    from src.baselines.NLSPN.src.config import args as args_NLSPN                    
                    network = NLSPNModel(args_NLSPN)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/NLSPN/NLSPN_KITTI_DC.pt'
                    # # model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/NLSPN/NLSPN_NYU.pt'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'],strict=True)  # NLSPN
                elif method == 'rz_sb_mar_SDCM':
                    # SDCM
                    from src.baselines.SDCM.model import DepthCompletionNet
                    from src.baselines.SDCM.config import args as args_SDCM

                    network = DepthCompletionNet(args_SDCM)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/SDCM/model_best.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'])  # SDCM
                elif method == 'rz_sb_mar_SDCM_DIODE_HRWSI':
                    from sfv2_networks import SDCM_retrain
                    network = SDCM_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_SDCM_DIODE_HRWSI/models/epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_MDAnet':
                    # MDAnet
                    from src.baselines.MDANet.modules.net import network as MDAnet
                    network = MDAnet()
                    network = torch.nn.DataParallel(network)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/MDANet/results/quickstart/checkpoints/net-best.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'])  # MDAnet
                elif method == 'rz_sb_mar_MDAnet_DIODE_HRWSI':
                    from sfv2_networks import MDAnet_retrain
                    network = MDAnet_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_MDAnet_DIODE_HRWSI/models/epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_EMDC':
                    # EMDC
                    from src.baselines.EMDC.models.EMDC import emdc
                    network = emdc(depth_norm=False)
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/EMDC/checkpoints/milestone.pth.tar'
                    network = network.cuda()
                    network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['state_dict'])  # EMDC
                elif method == 'rz_sb_mar_EMDC_DIODE_HRWSI':
                    # EMDC
                    from sfv2_networks import EMDC_retrain
                    network = EMDC_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_emdc_DIODE_HRWSI/models/epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
                elif method == 'rz_sb_mar_LRRU':
                    #  LRRU
                    from src.baselines.LRRU.model.model_dcnv2 import Model as LRRUModel
                    import argparse

                    arg = argparse.ArgumentParser(description='depth completion')
                    arg.add_argument('-p', '--project_name', type=str, default='inference')
                    arg.add_argument('-c', '--configuration', type=str, default='/data1/Chenbingyuan/Depth-Completion/src/baselines/LRRU/configs/val_lrru_base_kitti.yml')
                    arg = arg.parse_args()
                    from src.baselines.LRRU.configs import get as get_cfg
                    args_LRRU = get_cfg(arg)
                    network = LRRUModel(args_LRRU)
                    network = network.cuda()
                elif method == 'rz_sb_mar_LRRU_DIODE_HRWSI':
                    #  LRRU
                    from sfv2_networks import LRRU_retrain
                    network = LRRU_retrain(rezero=True)
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_LRRU_DIODE_HRWSI/models/epoch_60.pth'
                    network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 

                elif method == 'rz_sb_mar_BPnet':
                    from src.baselines.BPnet.models.BPNet import Net as BPnetModel
                    network = BPnetModel()
                    network = network.cuda()
                    model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/BPnet/BP_KITTI/result_ema.pth'
                    cp = torch.load(model_dir, map_location='cuda:0')
                    network.load_state_dict(cp['net'], strict=True)

                for mode in mode_list:
                    # 0-100
                    for pro in pro_dict[mode]:
                        print(str(args.rgbd_dir) + method + '_'  + mode + '_' + str(pro))
                        demo(args, network, pro, mode, network_type=method)


def eva():
    # 构建命令
    commands = """
    conda activate completionformer
    python /data1/Chenbingyuan/Depth-Completion/application/evaluate.py
    """

    # 启动一个 shell 进程，并捕获标准输出和标准错误
    process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 向 shell 进程发送命令并获取输出
    stdout, stderr = process.communicate(commands)

    # 打印输出和错误
    print(stdout)
    if stderr:
        print("Errors:\n", stderr)

if __name__ == "__main__":
    # turn fast mode on
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    rgbd_dir = ['KITTI','nyu', 'redweb','ETH3D','Ibims', 'VKITTI','Matterport3D', 'UnrealCV']
    # rgbd_dir = ['ETH3D','Ibims', 'VKITTI','Matterport3D', 'UnrealCV']
    # rgbd_dir = ['KITTI']
    dataset_list = copy.deepcopy(rgbd_dir)
    for i,dir in enumerate(rgbd_dir):
        rgbd_dir[i] = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/'+dir+'/val'

    mode_list = [ 'result']

    # method_list = ['rz_sb_mar_JARRN_full_05line_05point']
    # method_list = ['rz_sb_mar_BPnet']
    # method_list = ['rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI']
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_LSM']
    # method_list = ['rz_sb_mar_SDCM','rz_sb_mar_PEnet','rz_sb_mar_ReDC','rz_sb_mar_CFormer_KITTI', 'rz_sb_mar_EMDC', 
    #                'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_TWISE',] # completionformer 一个环境就可以解决
    # method_list = ['rz_sb_mar_MDAnet'] # torch1.7
    method_list = ['rz_sb_mar_LRRU'] # LRRU_new
    # method_list = ['rz_sb_mar_GuideNet'] # cuda121
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_large'] 
    # method_list = ['rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2']
    # method_list = ['rz_sb_mar_ReDC']
    # method_list = ['rz_sb_mar_JARRN_100LiDAR']
    # method_list = ['rz_sb_mar_JARRN_70LiDAR']
    # method_list = ['rz_sb_mar_JARRN_60LiDAR']
    
    # 

    epoch_list = [60]
    crop = False

    pro_dict = {'result':[0.01,0.1,0.2,0.5,0.7, 1.04, 1.016, 1.064,1.08,1.032, 1.0128]}
    # pro_dict = {'result': [1.04]}

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Started Time:", formatted_time)
    depth_inference()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Val-Ended Time:", formatted_time)
    print('Done!')
