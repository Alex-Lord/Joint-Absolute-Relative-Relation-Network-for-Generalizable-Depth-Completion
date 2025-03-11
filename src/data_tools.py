from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
from torchvision.transforms import ToPILImage
# from torchvision.transforms import InterpolationMode
import os
import math
import pickle
import cv2
from tqdm import tqdm
import random
import re
from copy import deepcopy
import glob
target_size = 320
# target_size = 256
# DistributedSampler=RandomSampler  # no DDP
def rgb_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    rgb = (np.array(data) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename: Path) -> Tensor:
    # Read Depth and converts to 0-1
    str_file = str(filename)
    if '.png' in str_file:
        data = Image.open(filename)
        
        if 'DIODE' in str_file:
            if '_indoors_' in str_file:
                depth = (np.array(data) / 20.).astype(np.float32)
            elif '_outdoor_' in str_file:
                depth = (np.array(data) / 100.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        # elif 'HRWSI' in str_file:
        #     depth = (np.array(data) / 255.).astype(np.float32)
        #     depth = np.clip(depth, 0, 1)
        elif 'redweb' in str_file:
            depth = (np.array(data) / 255.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        elif 'NYU' in str_file:
            depth = (np.array(data) / 255.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        else:
            depth = (np.array(data)/ 65535.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        data.close()
        # print(f'depth.shape={depth.shape}')
    return depth

def hole_read(filename: Path) -> Image:
    data = Image.open(filename)
    hole = (np.array(data) / 255.).astype(np.float32)
    hole = torch.from_numpy(hole).unsqueeze(0)
    data.close()
    return hole


def rgbd_transform(
        rgb: Tensor,
        gt: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    aug_together_transform = trans.Compose(
        [
            trans.RandomResizedCrop(size=(target_size, target_size),
                                          scale=(0.64, 1.0),
                                          ratio=(3.0/4.0, 4.0/3.0)),
            trans.RandomHorizontalFlip(0.5),
        ]
    )
    together_transform = trans.Compose(
        [
            trans.RandomCrop((target_size, target_size)),
            trans.RandomHorizontalFlip(0.5),
        ]
    )
    rgb_transform = trans.Compose(
        [
            trans.ColorJitter(0.2, 0.2, 0.2),
        ]
    )
    
    # together transform
    rgbgt = torch.cat((rgb, gt), dim=0)
    rgbgt = together_transform(rgbgt)  # 不用Aug
    
    rgb_aug = rgbgt[:3, :, :]
    gt_aug = rgbgt[3, :, :].unsqueeze(0)
    min_val, max_val = torch.min(gt_aug), torch.max(gt_aug)
    if max_val != 0 and min_val!=max_val:
        gt_aug.sub_(min_val).div_(max_val - min_val)
    else:
        rgbgt = torch.cat((rgb, gt), dim=0)
        rgbgt = together_transform(rgbgt)
        rgb_aug = rgbgt[:3, :, :]
        gt_aug = rgbgt[3, :, :].unsqueeze(0)
    rgb_aug = rgb_transform(rgb_aug)
    hole_gt = torch.ones_like(gt_aug)
    hole_gt[gt_aug == 0] = 0.
    return rgb_aug, gt_aug, hole_gt

def rgbd_transform_eval(
        rgb: Tensor,
        gt: Tensor,
        **kwargs,
) -> Tuple[Tensor, Tensor, Tensor]:
    together_transform = trans.Compose(
        [
            trans.RandomCrop((target_size, target_size)),
            trans.RandomHorizontalFlip(0.5),
        ]
    )
    rgb_transform = trans.ColorJitter(0.2, 0.2, 0.2)
    if 'crop' in kwargs and kwargs['crop'] != True:
        together_transform = trans.Compose(
            [
                trans.RandomHorizontalFlip(0.5),
            ]
        )
    # together transform
    rgbgt = torch.cat((rgb, gt), dim=0)
    rgbgt = together_transform(rgbgt)
    rgb = rgbgt[:3, :, :]
    gt = rgbgt[3, :, :].unsqueeze(0)

    rgb = rgb_transform(rgb)

    hole_gt = torch.ones_like(gt)
    hole_gt[gt == 0] = 0.

    return rgb, gt, hole_gt

def hole_transform(hole: Tensor) -> Tensor:
    transform = trans.Compose(
        [
            trans.RandomCrop((target_size, target_size)),
            # trans.RandomAffine(degrees=180, translate=(0.5, 0.5),
            #                    scale=(0.5, 4.0), shear=60, fill=1.),
            trans.RandomAffine(degrees=180, translate=(0.5, 0.5),
                               scale=(0.5, 4.0), shear=60,),
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
        ]
    )
    hole = transform(hole)
    hole[hole > 0.] = 1.

    return hole

def get_point_cby(pro_dict, rgbd_dir_list: list, **kwargs):
        
    def sample_point(gt: Tensor, sample_rate: float):
        gt = np.squeeze(gt.numpy())
        dim1, dim2 = gt.shape
        sampled_gt = gt.copy()
        # 生成一个随机掩码，用于选择要保留的元素
        mask = np.random.choice([True, False], size=(dim1, dim2), p=[sample_rate, 1-sample_rate])
        # 将掩码为False的元素置为0
        sampled_gt[~mask] = 0
        sampled_gt = torch.from_numpy(sampled_gt)
        sampled_gt = sampled_gt.unsqueeze(0)
        return sampled_gt


    def sample_metric(data_shape, zero_rate) -> Tensor:
        if zero_rate == 0.0:
            random_point = torch.ones(data_shape)
        elif zero_rate == 1.0:
            random_point = torch.zeros(data_shape)
        else:
            random_point = torch.ones(data_shape).uniform_(0.0, 1.0)
            random_point[random_point <= zero_rate] = 0.
            random_point[random_point > zero_rate] = 1.
        return random_point

    def randomlyadddistortion(distorted_gt: Tensor, hole_gt: Tensor, p_noise: float = 0.5, p_blur: float = 0.5) -> Tensor:
        distorted_depth_shape = distorted_gt.shape

        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:
            gaussian_noise = torch.ones(distorted_depth_shape).normal_(
                0, np.random.uniform(0.01, 0.1))
            random_point = sample_metric(
                distorted_depth_shape, np.random.uniform(0.0, 1.0)) * hole_gt
            distorted_gt = distorted_gt + gaussian_noise * random_point

        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:
            sample_factor = 2 ** (np.random.randint(1, 5))
            # depth_trans = trans.Compose([
            #     trans.Resize(
            #         (int(distorted_depth_shape[1] * 1.0 / sample_factor),
            #          int(distorted_depth_shape[2] * 1.0 / sample_factor)),
            #         interpolation=InterpolationMode.NEAREST
            #     ),
            #     trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
            #                  interpolation=InterpolationMode.NEAREST),
            # ])
            
            depth_trans = trans.Compose([
                trans.Resize(
                    (int(distorted_depth_shape[1] * 1.0 / sample_factor),
                    int(distorted_depth_shape[2] * 1.0 / sample_factor)),
                    interpolation=0  # 替换为对应于PIL.Image.NEAREST的整数值
                ),
                trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
                            interpolation=0),  # 同样替换为0
            ])
            distorted_gt = depth_trans(distorted_gt)

        distorted_gt = torch.clamp(distorted_gt, 0.0, 1.0)
        return distorted_gt
    for rgbd_dir in rgbd_dir_list:
        print(f'Dir:{rgbd_dir}')
        # 获取测试用数据point
        # 处理npy类depth
        unloader = ToPILImage()
        png_files = glob.glob(rgbd_dir + '/**/*.png', recursive=True)
        # 为了线数测试专门采样
        if 'line' in kwargs:line = kwargs['line']
        else:line = False
        crop =True
        if 'crop' in kwargs:
            crop = kwargs['crop']
        if crop == True:print('We DO crop img')
        else:print('We do NOT crop img')
        if line == True:
            def rgbd_transform_for_line(rgb: Tensor, gt: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor]:
                # concatenate the two tensors along the first dimension
                concatenated = torch.cat((gt, rgb), dim=0)
                # select the first 320 rows and columns
                if 'crop' in kwargs and kwargs['crop'] is True:
                    cropped = concatenated[:, :320, :320]
                
                # split the cropped tensor into two tensors along the first dimension
                gt_cropped, rgb_cropped = torch.split(cropped, [1, 3], dim=0)
                hole_gt = torch.ones_like(gt_cropped)
                hole_gt[gt_cropped == 0] = 0.

                return rgb_cropped, gt_cropped, hole_gt
            rgbd_trans = rgbd_transform_for_line
        else:rgbd_trans = rgbd_transform_eval
        for file in tqdm(png_files, desc='png_files'):
            str_file = str(file)
            flag = 0
            if '/val/' not in str_file or 'crop' in str_file or '_gt/' not in str_file:
                continue
            
            if 'val/NYUv2_gt' in str_file or 'val/DIODE_gt' in str_file or 'val/HRWSI_gt' in str_file or 'val/redweb_gt' in str_file:
                flag = 1
                rgb_file = str_file.replace('gt', 'rgb')
                rgb_file = rgb_file.replace('.png', '.jpg')
                rgb = rgb_read(Path(rgb_file))
                rgb_file = rgb_file.replace('.jpg', 'dis.jpg')
                gt = depth_read(file)
                rgb, gt, hole_gt = rgbd_trans(rgb, gt,crop=crop)
                gt = torch.as_tensor((gt * 255.), dtype=torch.uint8)
                gt_img = torch.as_tensor(gt).cpu().clone()
                gt_img = gt_img.squeeze(0)
                gt_img = unloader(gt_img)
                rgb = (rgb.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                im = Image.fromarray(rgb)  # numpy 转 image类
                im.save(rgb_file)
                gt_file = str_file.replace('.png', 'crop.png')
                gt_img.save(gt_file)
            else:
                flag = 1
                rgb_file = str_file.replace('gt', 'rgb')
                rgb = rgb_read(Path(rgb_file))
                rgb_file = rgb_file.replace('.png', 'dis.png')
                gt = depth_read(file)
                rgb, gt, hole_gt = rgbd_trans(rgb, gt,crop=crop)
                gt = torch.as_tensor((gt * 255.), dtype=torch.uint8)
                gt_img = torch.as_tensor(gt).cpu().clone()
                gt_img = gt_img.squeeze(0)
                gt_img = unloader(gt_img)
                rgb = (rgb.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
                im = Image.fromarray(rgb)  # numpy 转 image类
                im.save(rgb_file)
                gt_file = str_file.replace('.png', 'crop.png')
                gt_img.save(gt_file)
            if flag == 1:
                for key in pro_dict:
                    org_point_path = str_file.replace('/val/', '/point/')
                    if key == 'result':
                        for pro in pro_dict[key]:
                            point_path = org_point_path
                            point_path = point_path.replace(
                                'point', 'point/point_'+str(pro))
                            if not os.path.exists(str(Path(point_path).parent)):
                                os.makedirs(str(Path(point_path).parent))
                            sampled_gt = sample_point(gt, pro)
                            sampled_gt_img = sampled_gt.cpu().clone()
                            sampled_gt_img = sampled_gt_img.squeeze(0)
                            sampled_gt_img = unloader(sampled_gt_img)
                            sampled_gt_img.save(point_path)
                    else:
                        for pro in pro_dict[key]:
                            point_path = org_point_path
                            point_path = point_path.replace(
                                'point', 'point/point_nb_'+str(pro))
                            if not os.path.exists(str(Path(point_path).parent)):
                                os.makedirs(str(Path(point_path).parent))
                            sampled_gt = sample_point(gt, pro)
                            distorted_gt = randomlyadddistortion(sampled_gt, hole_gt)
                            distorted_gt_im = distorted_gt.cpu().clone()
                            distorted_gt_im = distorted_gt_im.squeeze(0)
                            distorted_gt_im = unloader(distorted_gt_im)
                            distorted_gt_im.save(point_path)
def get_lines_cby(pro_dict, rgbd_dir: Path, lines:list):
    def generate_degree_map(rgbd_dir):
        depths=np.load("/media/10T/cby/dataset/NYUv2/nyuv2_depth.npy")
        depths = torch.as_tensor(depths)
        # 内参
        fx_d = 5.8262448167737955e+02
        fy_d = 5.8269103270988637e+02
        cx_d = 3.1304475870804731e+02
        cy_d = 2.3844389626620386e+02
        for file_2 in rgbd_dir.rglob('*'):
            COUNT = 0
            for file in file_2.rglob('*.png'): 
                COUNT += 1
            for file in tqdm(file_2.rglob('*.png'), desc='png_files', total=COUNT):
                if 'crop' in str(file):
                    str_file = str(file)
                    numbers = re.findall('\d+', str_file)
                    if numbers:
                        number = int(numbers[-1])
                    else:raise ValueError("No number in str_file.")
                    gt = depth_read(file)
                    gt = torch.as_tensor((gt * 255.), dtype=torch.uint8)
                    # 获取深度图的尺寸
                    h, w = gt.shape[1:]
                    theta_map = gt.clone().type(torch.float32)
                    xyz_map = depths[number,:320,:320]
                    
                    # 遍历深度图的每个像素，并读取其深度值及对应的横坐标x和纵坐标y
                    xyz_map = xyz_map.squeeze().numpy()
                    # 计算每个像素点的三维坐标
                    h, w = xyz_map.shape
                    y, x = np.indices((h, w))
                    x3 = (x - cx_d) * xyz_map / fx_d
                    y3 = (y - cy_d) * xyz_map / fy_d
                    z3 = xyz_map

                    # 将三维坐标拼接成点云
                    xyz_map = np.stack((x3, y3, z3), axis=-1)
                    for i in range(h):
                        for j in range(w):
                            x = xyz_map[i][j][0]
                            y = xyz_map[i][j][1]
                            z = xyz_map[i][j][2]
                            theta = np.rad2deg(np.arcsin(y/np.sqrt(x*x + y*y + z*z)))
                            theta_map[0][i][j] = theta  # 小数点后两位
                    theta_map = torch.as_tensor(theta_map)
                    theta_map = torch.cat((gt,theta_map),dim = 0)
                    theta_map_path = str_file.replace('/NYUv2_gt','_gt_degree')
                    theta_map_path = theta_map_path.replace('.png','.npy')
                    np.save(theta_map_path, theta_map.numpy())
                    # 使用with语句打开文件
                    with open('/data1/Chenbingyuan/Depth-Completion/g2_dataset/nyu/val_gt_degree/flag.txt', "w") as file:
                        # 写入文本内容
                        file.write("Done with time")  
    def get_lines(lines:list, file_path:str, line_gap=0.4, single_line_gap=0.08):
        """
        获取不同线数的深度图   
        :param lines: 线数列表
        :param file_path: 深度图路径
        :param single_line_gap: 单条线之间的间隔
        """
        # 获取深度图
        mapping = np.load(file_path)
        mapping_mask = np.zeros_like(mapping[1], dtype=bool)
        
        ran = [np.min(mapping[1])+5, np.max(mapping[1])-5]
        line_gap = (ran[1] - ran[0]) / 64
        line_64 = np.flip(np.arange(ran[0], ran[1], line_gap+single_line_gap))
        for line in lines:
            if line != 0:
                mapping_mask = np.zeros_like(mapping[1], dtype=bool)
                line_list = np.flip(line_64[::int(64/line)])  # 求得每条线的坐标list
                for every_line in line_list:
                    mapping_mask[np.logical_and(every_line <= mapping[1], mapping[1] <= every_line+single_line_gap)] = True
                mapping_copy = deepcopy(mapping)
                mapping_copy[0][mapping_mask != True] = 0
            else:
                mapping_copy = np.zeros_like(mapping)
            save_path = file_path.replace('val_gt_degree', f'lines/line_{line}' + '/NYUv2_gt')
            save_path = save_path.replace('npy', 'png')
            save_path = save_path.replace('crop', '')
            depth_map = mapping_copy[0].astype('uint8')
            cv2.imwrite(save_path, depth_map)
    
    if not os.path.exists('/data1/Chenbingyuan/Depth-Completion/g2_dataset/nyu/val_gt_degree/flag.txt'):
        generate_degree_map(rgbd_dir)
    else:print('degree_map file already exits!')
    
    # 判断degree_map文件是否存在
    if not os.path.exists('/data1/Chenbingyuan/Depth-Completion/g2_dataset/nyu/val_gt_degree/flag.txt'):
        generate_degree_map(rgbd_dir)
    else:print('degree_map file already exits!')
    
    # 生成lines文件
    for line in lines:
        line_path = Path(str(rgbd_dir).replace('val', f'lines/line_{line}'+ '/NYUv2_gt'))
        if not line_path.exists():line_path.mkdir(parents=True, exist_ok=True)
    
    # 生成不同的线数深度图
    degree_path = Path(str(rgbd_dir).replace('val','val_gt_degree'))
    
    COUNT = 0
    for file in degree_path.rglob('*.npy'): 
        if 'crop' in str(file):COUNT += 1
    for file in tqdm(degree_path.rglob('*.npy'), desc='npy_files', total=COUNT):
        if 'crop' in str(file):get_lines(lines, str(file))

def get_very_sparse_point_cby(pro_dict, rgbd_list: list):
    # 获取测试用数据point
    def sample_point(gt: torch.Tensor, n: int) -> torch.Tensor:
        if n != 0:
            # 找到所有非0点的位置坐标
            nonzero_points = torch.nonzero(gt[0]).unbind(1)
            num_nonzero = nonzero_points[0].size(0)
            
            # 如果非0点的数量小于n，则无法进行随机采样，抛出异常
            if num_nonzero < n:
                print(f"There are less than {n} non-zero elements in the matrix")
                return gt
            
            # 随机选择n个非0点，并记录它们的坐标和对应的值
            chosen_indices = torch.randperm(num_nonzero)[:n]
            chosen_points = tuple(nonzero_points[i][chosen_indices] for i in range(len(nonzero_points)))
            chosen_values = gt[0, chosen_points[0], chosen_points[1]]
            
            # 生成一个全零的遮罩，将选中的点的位置设为1，其余位置仍为0
            mask = torch.zeros_like(gt)
            mask[0, chosen_points[0], chosen_points[1]] = 1
            
            # 使用遮罩将选中的点的值设为0，其余点的值仍保留
            gt = torch.mul(gt, mask)
            
            # 将选中的点的原始值赋回到对应位置
            gt[0, chosen_points[0], chosen_points[1]] = chosen_values
        else:
            gt = torch.zeros_like(gt)
        
        return gt
    
    unloader = ToPILImage()
    for rgbd_dir in rgbd_list:
        print(rgbd_dir)
        png_files = glob.glob(rgbd_dir+ '/**/*.png',recursive=True)
        for file in tqdm(png_files, desc='png_files'):
            str_file = file
            if 'crop' in str_file:
                gt = depth_read(file)
                gt = torch.as_tensor((gt * 255.), dtype=torch.uint8)
                hole_gt = torch.ones_like(gt)
                hole_gt[gt == 0] = 0.
                for key in pro_dict:
                    orgpoint_path = str_file.replace('/val/', '/point/')
                    for pro in pro_dict[key]:
                        point_path = orgpoint_path
                        point_path = point_path.replace(
                            '/point/', '/point/very_sparse_point_'+str(pro)+'/')
                        point_path = point_path.replace('crop.png', '.png')
                        if not os.path.exists(str(Path(point_path).parent)):
                            os.makedirs(str(Path(point_path).parent))
                        sampled_gt = sample_point(gt, pro)
                        sampled_gt_img = sampled_gt.cpu().clone()
                        sampled_gt_img = sampled_gt_img.squeeze(0)
                        sampled_gt_img = unloader(sampled_gt_img)
                        sampled_gt_img.save(point_path)
                        
def get_seg_point_cby(pro_dict, rgbd_dir: Path):
    def sample_differ_points(segment_mapping:np.ndarray, gt:Tensor, n:int):
        gt = gt.numpy()
        nonzero_points = np.argwhere(segment_mapping != 0)
        if len(nonzero_points) < n:
            raise ValueError(f"There are less than {n} non-zero elements in the matrix")
        seg_list = []
        chosen_indices = []
        index_list = []
        index = random.randint(0, len(nonzero_points)-1)
        chosen_indices.append(index)
        seg_list.append(segment_mapping[nonzero_points[index][0], nonzero_points[index][1]])
        index_list.append(index)
        # 加一个点
        
        if n >= 2:
        # 加一个不同物体上的点
            for i in range(10):
                index = random.randint(0, len(nonzero_points)-1)
                if segment_mapping[nonzero_points[index][0], nonzero_points[index][1]] not in seg_list:
                    chosen_indices.append(index)
                    seg_list.append(segment_mapping[nonzero_points[index][0], nonzero_points[index][1]])
                    index_list.append(index)
                    break
                else:
                    if i >=  9:
                        index = random.randint(0, len(nonzero_points)-1)
                        chosen_indices.append(index)
                        seg_list.append(segment_mapping[nonzero_points[index][0], nonzero_points[index][1]])
                        index_list.append(index)
        
        if n >= 3:
            # 加上剩余的点
            for i in range(n-2):
                index = random.randint(0, len(nonzero_points)-1)
                chosen_indices.append(index)
                seg_list.append(segment_mapping[nonzero_points[index][0], nonzero_points[index][1]])
        
        chosen_points = nonzero_points[chosen_indices]
        chosen_values = gt[0, chosen_points[:, 0], chosen_points[:, 1]]
        mask = np.zeros_like(gt)
        mask[0, chosen_points[:, 0], chosen_points[:, 1]] = 1
        gt = np.multiply(gt, mask)
        gt[0, chosen_points[:, 0], chosen_points[:, 1]] = chosen_values
        gt = torch.as_tensor(gt, dtype=torch.uint8)
        return gt
    
    def sample_same_points(segment_mapping:np.ndarray, gt:Tensor, n:int):
        gt = gt.numpy()
        nonzero_points = np.argwhere(segment_mapping != 0)
        if len(nonzero_points) < n:
            raise ValueError(f"There are less than {n} non-zero elements in the matrix") 
        
        # 随机选取一个物体，然后将不在这个物体上的点赋值为0
        index = random.randint(0, len(nonzero_points)-1)
        while segment_mapping[nonzero_points[index][0], nonzero_points[index][1]] == 0:
            index = random.randint(0, len(nonzero_points)-1)
        segment_mapping[segment_mapping != segment_mapping[nonzero_points[index][0], nonzero_points[index][1]]] = 0
        
        # 重新记录非0值的index
        nonzero_points = np.argwhere(segment_mapping != 0)
        if len(nonzero_points) < n:
            raise ValueError(f"There are less than {n} non-zero elements in the matrix")
        
        # 从非0值中选取n个点
        chosen_indices = random.sample(range(len(nonzero_points)), n)
        chosen_points = nonzero_points[chosen_indices]
        chosen_values = gt[0, chosen_points[:, 0], chosen_points[:, 1]]
        mask = np.zeros_like(gt)
        mask[0, chosen_points[:, 0], chosen_points[:, 1]] = 1
        gt = np.multiply(gt, mask)
        gt[0, chosen_points[:, 0], chosen_points[:, 1]] = chosen_values
        gt = torch.as_tensor(gt, dtype=torch.uint8)
        return gt
        
    segment = np.load(r'/data/4TSSD/cby/dataset/NYUv2/nyuv2_labels.npy')
    unloader = ToPILImage()
    for file_2 in rgbd_dir.rglob('*'):
        COUNT = 0
        for file in file_2.rglob('*.png'): COUNT += 1
        for file in tqdm(file_2.rglob('*.png'), desc='png_files', total=COUNT):
            str_file = str(file)
            if 'val/NYUv2_gt' in str_file and 'crop' in str_file:
                
                # 生成语义图
                str_file = str(file)
                numbers = re.findall('\d+', str_file)
                if numbers:
                    number = int(numbers[-1])
                else:raise ValueError("No number in str_file.")
                segment_mapping = segment[number,:320,:320]  # (1,320,320)

                gt = depth_read(file)  #(1,320,320)
                # 将深度图转换为张量，并将像素值缩放到[0,255]之间
                gt = torch.as_tensor((gt * 255.), dtype=torch.uint8)
                for key in pro_dict:
                    orgpoint_path = str_file.replace('val', 'point')
                    if str(key) == 'same_seg_':
                        sample_point = sample_same_points
                    else:
                        sample_point = sample_differ_points
                    orgpoint_path = orgpoint_path.replace('point','point/very_sparse_point_'+str(key))
                    for pro in pro_dict[key]:
                        point_path = orgpoint_path
                        point_path = point_path.replace(
                            str(key), str(key)+str(pro))
                        point_path = point_path.replace('crop', '')
                        if pro == 0:
                            sampled_gt = torch.zeros_like(gt)
                        else:
                            sampled_gt = sample_point(segment_mapping, gt, pro)
                        sampled_gt_img = sampled_gt.cpu().clone()
                        sampled_gt_img = sampled_gt_img.squeeze(0)
                        sampled_gt_img = unloader(sampled_gt_img)
                        point_path = point_path.replace('crop','')
                        if not os.path.exists(str(Path(point_path).parent)):
                            os.makedirs(str(Path(point_path).parent))
                        sampled_gt_img.save(point_path)

class RGBDDataset(Dataset):
    def __init__(self, data_dir: list,
                 ) -> None:
        print('start RGBD init')
        super(RGBDDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = rgbd_transform
        datalist = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/data_list/'
        if os.path.exists(datalist + 'rgb_ls.pkl'):
            print('datalist already exists')
            rgb_list = open(datalist + 'rgb_ls.pkl', 'rb')
            self.rgb_ls = pickle.load(rgb_list)
            rgb_list.close()
            depth_list = open(datalist + 'depth_ls.pkl', 'rb')
            self.depth_ls = pickle.load(depth_list)
            depth_list.close()
        else:
            print('datalist do not exists')
            self.rgb_ls, self.depth_ls = self.__getrgbd__(self.data_dir)
            os.makedirs(datalist, exist_ok=True)
            rgb_list = open(datalist + 'rgb_ls.pkl', 'wb')
            pickle.dump(self.rgb_ls, rgb_list)
            rgb_list.close()
            depth_list = open(datalist + 'depth_ls.pkl', 'wb')
            pickle.dump(self.depth_ls, depth_list)
            depth_list.close()

    @ staticmethod
    def __sample_metric__(data_shape, zero_rate) -> Tensor:
        if zero_rate == 0.0:
            random_point = torch.ones(data_shape)
        elif zero_rate == 1.0:
            random_point = torch.zeros(data_shape)
        else:
            random_point = torch.ones(data_shape).uniform_(0.0, 1.0)
            random_point[random_point <= zero_rate] = 0.
            random_point[random_point > zero_rate] = 1.
        return random_point

    def __randomlyadddistortion__(
            self,
            distorted_gt: Tensor,
            hole_gt: Tensor,
            p_noise: float = 0.5,
            p_blur: float = 0.5
    ) -> Tensor:
        distorted_depth_shape = distorted_gt.shape

        # add noise
        if np.random.uniform(0.0, 1.0) < p_noise:
            gaussian_noise = torch.ones(distorted_depth_shape).normal_(
                0, np.random.uniform(0.01, 0.1))
            random_point = self.__sample_metric__(
                distorted_depth_shape, np.random.uniform(0.0, 1.0)) * hole_gt
            distorted_gt = distorted_gt + gaussian_noise * random_point

        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:
            sample_factor = 2 ** (np.random.randint(1, 5))
            # depth_trans = trans.Compose([
            #     trans.Resize(
            #         (int(distorted_depth_shape[1] * 1.0 / sample_factor),
            #          int(distorted_depth_shape[2] * 1.0 / sample_factor)),
            #         interpolation=InterpolationMode.NEAREST
            #     ),
            #     trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
            #                  interpolation=InterpolationMode.NEAREST),
            # ])
            depth_trans = trans.Compose([
    trans.Resize(
        (int(distorted_depth_shape[1] * 1.0 / sample_factor),
         int(distorted_depth_shape[2] * 1.0 / sample_factor)),
        interpolation=0  # 替换为对应于PIL.Image.NEAREST的整数值
    ),
    trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
                 interpolation=0),  # 同样替换为0
])
            distorted_gt = depth_trans(distorted_gt)

        distorted_gt = torch.clamp(distorted_gt, 0.0, 1.0)
        return distorted_gt

    @ staticmethod

    def __getrgbd__(data_path_all: list) -> Tuple[List[Path], List[Path]]:
        rgb_ls = []
        depth_ls = []
        print('start to get rgb_d from png and jpg')
        # print(f'data_path_all = {data_path_all}')
        for data_path in data_path_all:
            jpg_data_file_all = glob.glob(str(data_path)+'/**/*.jpg', recursive=True)
            png_data_file_all = glob.glob(str(data_path)+'/**/*.png', recursive=True)
            for file in tqdm(jpg_data_file_all, desc='jpg_files'):
                if 'nyu' in file and 'rgb' in file and 'crop' not in file:
                    rgb_ls.append(Path(file))
                    depth_file = file.replace('rgb', 'gt')
                    depth_file = depth_file.replace('jpg', 'png')
                    depth_ls.append(Path(depth_file))
                if 'HRWSI_rgb' in file and 'crop' not in file:
                    rgb_ls.append(Path(file))
                    depth_file = file.replace('HRWSI_rgb', 'HRWSI_gt')
                    depth_file = depth_file.replace('jpg', 'png')
                    depth_ls.append(Path(depth_file))
                
            for file in tqdm(png_data_file_all, desc='png_files'):
                if 'DIODE_rgb' in file and 'crop' not in file:
                    rgb_ls.append(Path(file))
                    depth_file = file.replace('rgb', 'gt')
                    depth_ls.append(Path(depth_file))
                if 'HRWSI_2_rgb' in file and 'crop' not in file:
                    rgb_ls.append(Path(file))
                    depth_file = file.replace('HRWSI_2_rgb', 'HRWSI_2_gt')
                    depth_file = depth_file.replace('jpg', 'png')
                    depth_ls.append(Path(depth_file))
        return rgb_ls, depth_ls

    def __getpoint__(self, gt: Tensor, hole_gt: Tensor) -> Tensor:
        distorted_gt = gt.clone()

        random_factor = np.random.uniform(0.0, 1.0)
        if random_factor < 0.2:
            # depth recovery
            zero_rate = 0.0
        elif random_factor < 0.4:
            # not very sparse depth completion
            zero_rate = np.random.uniform(0.0, 0.9)
        elif random_factor < 0.6:
            # very sparse depth completion
            zero_rate = np.random.uniform(0.9, 1.0)
        else:
            # depth estimation
            zero_rate = 1.0

        if zero_rate == 0:
            distorted_gt = self.__randomlyadddistortion__(
                distorted_gt, hole_gt, p_blur=1.0)
        elif zero_rate < 1:
            distorted_gt = self.__randomlyadddistortion__(distorted_gt, hole_gt, 0.3, 0.3)

        # Random select p_noise points
        point_map = self.__sample_metric__(gt.shape, zero_rate) * distorted_gt
        # point_map = self.__sample_metric__(gt.shape, zero_rate)
        return point_map
    def __getline__(self, gt: Tensor, hole_gt: Tensor) -> Tensor:
        npy_folder = '/data1/Chenbingyuan/Depth-Completion/application/lines'

        # 获取所有npy文件路径
        npy_files = [os.path.join(npy_folder, f) for f in os.listdir(npy_folder) if f.endswith('.npy')]

        # 随机选择一个mask npy文件
        mask_npy_path = random.choice(npy_files)

        # 读取mask numpy数组
        mask = np.load(mask_npy_path)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # 转换为 PyTorch Tensor
        mask = torch.from_numpy(mask).to(gt.dtype)
        point_map = gt * mask
        return point_map

    def __len__(self) -> int:
        assert (len(self.rgb_ls) == len(self.depth_ls)
                ), f"The number of RGB and gen_depth is unpaired"
        return len(self.rgb_ls)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # names of RGB and sta_point_depth should be paired
        rgb_path = self.rgb_ls[item]
        depth_path = self.depth_ls[item]
        if str(depth_path)[-3:] != 'npy':
            assert (rgb_path.name[:-4] == depth_path.name[:-4]), \
                f"The RGB {str(self.rgb_ls[item])} and gen_depth {str(self.depth_ls[item])} is unpaired"

        # names of RGB and sta_point_depth should be paired
        rgb = rgb_read(rgb_path)
        gt = depth_read(depth_path)
        rgb, gt, hole_gt = self.transform(rgb, gt)
        
        gt_cp = torch.clone(gt)
        hole_gt_cp = torch.clone(hole_gt)
        
        
        random_factor = np.random.uniform(0.0, 1.0)
        
        # 0%概率变成线数
        line_factor = 0
        if random_factor < line_factor:
            point_map = self.__getline__(gt, hole_gt)
        else:
            point_map = self.__getpoint__(gt, hole_gt)
        return rgb, gt_cp, point_map, hole_gt_cp


class HoleDataset(Dataset):
    def __init__(
            self,
            data_dir: list,
    ) -> None:
        super(HoleDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = hole_transform
        datalist = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/data_list/'
        if os.path.exists(datalist + 'hole_ls.pkl'):
            hole_data_list = open(datalist + 'hole_ls.pkl', 'rb')
            self.hole_ls = pickle.load(hole_data_list)
            hole_data_list.close()
        else:
            self.hole_ls = self.__gethole__(self.data_dir)
            hole_data_list = open(datalist + 'hole_ls.pkl', 'wb')
            pickle.dump(self.hole_ls, hole_data_list)
            hole_data_list.close()

    @ staticmethod
    # def __gethole__(path: Path) -> List[Path]:
    #     hole = []
    #     for file in path.rglob('*.png'):
    #         hole.append(file)
    #     return hole
    def __gethole__(path_all: list) -> List[Path]:
        for path in path_all:
            hole = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
        return hole

    def __len__(self) -> int:
        return len(self.hole_ls)

    def __getitem__(self, item: int) -> Tensor:
        if np.random.uniform(0.0, 1.0) <= 0.5:
            hole = hole_read(self.hole_ls[item])
            hole = self.transform(hole)
        else:
            hole = torch.ones((1, target_size, target_size))

        return hole
def get_dataloader(
        rgbd_dirs: list,
        hole_dirs: list,
        batch_size: int,
        rank: torch.device,
        num_workers: int = 0,
        factor: int = 1,
) -> Tuple[
    DataLoader,
    DataLoader,
    DistributedSampler,
    DistributedSampler,
]:
    # initialize test_datasets
    rgbd_dataset = RGBDDataset(rgbd_dirs)
    hole_dataset = HoleDataset(hole_dirs)
    if rank == 0:
        print(
            f"Loaded the RGBD dataset with: {len(rgbd_dataset)} images...\n"
            f"Loaded the Hole dataset with: {len(hole_dataset)} images...\n"
        )
    # hole imgs should be sufficient enough
    ratio_factor = math.ceil(factor * len(rgbd_dataset) / len(hole_dataset))
    hole_dataset.hole_ls *= ratio_factor

    # initialize dataloaders
    rgbgph_sampler = DistributedSampler(rgbd_dataset)
    hole_sampler = DistributedSampler(hole_dataset)

    rgbgph_data = DataLoader(
        rgbd_dataset, batch_size=batch_size, drop_last=True, sampler=rgbgph_sampler, num_workers=num_workers,
        pin_memory=False, persistent_workers=True)
    hole_data = DataLoader(
        hole_dataset, batch_size=factor * batch_size, drop_last=True, sampler=hole_sampler, num_workers=num_workers,
        pin_memory=False, persistent_workers=True)

    return rgbgph_data, hole_data, rgbgph_sampler, hole_sampler

