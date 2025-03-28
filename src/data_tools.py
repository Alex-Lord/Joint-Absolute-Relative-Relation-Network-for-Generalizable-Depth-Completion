from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
from torchvision.transforms import ToPILImage

import os
import math
import pickle
import cv2
from tqdm import tqdm
import random
import glob
target_size = 320


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
    rgbgt = together_transform(rgbgt) 
    
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
class RGBDDataset(Dataset):
    def __init__(self, data_dir: list,
                 ) -> None:
        print('start RGBD init')
        super(RGBDDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = rgbd_transform
        datalist = '/data1/name/JARRN/g2_dataset/data_list/'
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
        npy_folder = '/data1/name/JARRN/application/lines'

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
        datalist = '/data1/name/JARRN/g2_dataset/data_list/'
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

