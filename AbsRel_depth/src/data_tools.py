from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as trans
from torchvision.transforms import InterpolationMode
import os
import math
import pickle

target_size = 320


def rgb_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    rgb = (np.array(data) / 255.).astype(np.float32)
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1)))
    data.close()
    return rgb


def depth_read(filename: Path) -> Tensor:
    data = Image.open(filename)
    # make sure we have a proper 16bit sta_point_depth map here.. not 8bit!
    depth = (np.array(data) / 65535.).astype(np.float32)
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
    together_transform = trans.Compose(
        [
            trans.RandomCrop((target_size, target_size)),
            trans.RandomHorizontalFlip(0.5),
        ]
    )
    rgb_transform = trans.ColorJitter(0.2, 0.2, 0.2)

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
            trans.RandomAffine(degrees=180, translate=(0.5, 0.5), scale=(0.5, 4.0), shear=60, fill=1.),
            trans.RandomHorizontalFlip(0.5),
            trans.RandomVerticalFlip(0.5),
        ]
    )
    hole = transform(hole)
    hole[hole > 0.] = 1.

    return hole


class RGBDDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
    ) -> None:
        super(RGBDDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = rgbd_transform
        if os.path.exists('data_list/rgb_ls.pkl'):
            rgb_list = open('data_list/rgb_ls.pkl', 'rb')
            self.rgb_ls = pickle.load(rgb_list)
            rgb_list.close()
            depth_list = open('data_list/depth_ls.pkl', 'rb')
            self.depth_ls = pickle.load(depth_list)
            depth_list.close()
        else:
            self.rgb_ls, self.depth_ls = self.__getrgbd__(self.data_dir)
            rgb_list = open('data_list/rgb_ls.pkl', 'wb')
            pickle.dump(self.rgb_ls, rgb_list)
            rgb_list.close()
            depth_list = open('data_list/depth_ls.pkl', 'wb')
            pickle.dump(self.depth_ls, depth_list)
            depth_list.close()

    @staticmethod
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
            gaussian_noise = torch.ones(distorted_depth_shape).normal_(0, np.random.uniform(0.01, 0.1))
            random_point = self.__sample_metric__(distorted_depth_shape, np.random.uniform(0.0, 1.0)) * hole_gt
            distorted_gt = distorted_gt + gaussian_noise * random_point

        # add blur
        if np.random.uniform(0.0, 1.0) < p_blur:
            sample_factor = 2 ** (np.random.randint(1, 5))
            depth_trans = trans.Compose([
                trans.Resize(
                    (int(distorted_depth_shape[1] * 1.0 / sample_factor),
                     int(distorted_depth_shape[2] * 1.0 / sample_factor)),
                    interpolation=InterpolationMode.NEAREST
                ),
                trans.Resize((distorted_depth_shape[1], distorted_depth_shape[2]),
                             interpolation=InterpolationMode.NEAREST),
            ])
            distorted_gt = depth_trans(distorted_gt)

        distorted_gt = torch.clamp(distorted_gt, 0.0, 1.0)
        return distorted_gt

    @staticmethod
    def __getrgbd__(data_path: Path) -> Tuple[List[Path], List[Path]]:
        rgb_ls = []
        depth_ls = []

        for file in data_path.rglob('*.png'):
            str_file = str(file)
            if '/rgb/' in str_file:
                rgb_ls.append(file)
                depth_file = str_file.replace('/rgb/', '/depth/', 1)
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
            distorted_gt = self.__randomlyadddistortion__(distorted_gt, hole_gt, p_blur=1.0)
        elif zero_rate < 1:
            distorted_gt = self.__randomlyadddistortion__(distorted_gt, hole_gt, 0.3, 0.3)

        # Random select p_noise points
        point_map = self.__sample_metric__(gt.shape, zero_rate) * distorted_gt

        return point_map

    def __len__(self) -> int:
        assert (len(self.rgb_ls) == len(self.depth_ls)), f"The number of RGB and gen_depth is unpaired"
        return len(self.rgb_ls)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # names of RGB and sta_point_depth should be paired
        rgb_path = self.rgb_ls[item]
        depth_path = self.depth_ls[item]
        assert (rgb_path.name[:-4] == depth_path.name[:-4]), \
            f"The RGB {str(self.rgb_ls[item])} and gen_depth {str(self.depth_ls[item])} is unpaired"

        # names of RGB and sta_point_depth should be paired
        rgb = rgb_read(rgb_path)
        gt = depth_read(depth_path)
        rgb, gt, hole_gt = self.transform(rgb, gt)
        point_map = self.__getpoint__(gt, hole_gt)
        return rgb, gt, point_map, hole_gt


class HoleDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
    ) -> None:
        super(HoleDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = hole_transform
        if os.path.exists('data_list/hole_ls.pkl'):
            hole_data_list = open('data_list/hole_ls.pkl', 'rb')
            self.hole_ls = pickle.load(hole_data_list)
            hole_data_list.close()
        else:
            self.hole_ls = self.__gethole__(self.data_dir)
            hole_data_list = open('data_list/hole_ls.pkl', 'wb')
            pickle.dump(self.hole_ls, hole_data_list)
            hole_data_list.close()

    @staticmethod
    def __gethole__(path: Path) -> List[Path]:
        hole = []
        for file in path.rglob('*.png'):
            hole.append(file)
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
        rgbd_dirs: Path,
        hole_dirs: Path,
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
        pin_memory=True, persistent_workers=True)
    hole_data = DataLoader(
        hole_dataset, batch_size=factor * batch_size, drop_last=True, sampler=hole_sampler, num_workers=num_workers,
        pin_memory=True, persistent_workers=True)

    return rgbgph_data, hole_data, rgbgph_sampler, hole_sampler
