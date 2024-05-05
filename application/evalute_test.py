import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from application.application_utils import DepthEvaluation
import datetime
import sys
import os
import glob

from typing import List, Tuple
import numpy as np
from PIL import Image
from torch import Tensor
import torch
print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(sys.path)

def depth_read(filename: Path) -> Tensor:
    str_file = str(filename)
    if '.png' in str_file:
        data = Image.open(filename)
        if 'DIODE' in str_file:
            if '_indoors_' in str_file:
                depth = (np.array(data) / 20.).astype(np.float32)
            elif '_outdoor_' in str_file:
                depth = (np.array(data) / 100.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        else:
            depth = (np.array(data) / 255.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        data.close()
    return depth

def evaluate_depth_file(args):
    depth_file, depth_path, gt_path, pro, crop = args
    depth_array = np.array(Image.open(depth_file)).astype(np.float32)
    depth = np.clip(depth_array, 1.0, 255.0)
    gt_files = depth_file.replace(depth_path, gt_path).replace('.png', 'crop.png')
    gt = np.clip(np.array(Image.open(gt_files)).astype(np.float32), 0., 255.)
    if crop:
        depth, gt = depth[:256, :256], gt[:256, :256]
    else:
        imae = DepthEvaluation.imae(depth, gt)
        rmse = 0
        mae = DepthEvaluation.mae(depth, gt)
        irmse = DepthEvaluation.irmse(depth, gt)
        rel = 0
    return depth_file, rmse, rel, imae, mae,irmse


class DepthDatasetEvaluation(object):
    @staticmethod
    def evaluate_depth_dataset(depth_path, gt_path, pro, crop, log_path):
        png_files = glob.glob(depth_path + '/**/*.png', recursive=True)
        args = [(depth_file, depth_path, gt_path, pro, crop) for depth_file in png_files]
        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(evaluate_depth_file, args))

        # Aggregate and calculate mean values
        aggregated_results = np.mean(np.array([result[1:] for result in results]), axis=0)
        with open(log_path, 'a') as log_file:
            log_file.write(f"Mean Results: RMSE={aggregated_results[0]}, REL={aggregated_results[1]}, IMAE={aggregated_results[2]}, MAE={aggregated_results[3]}, IRMSE={aggregated_results[4]}\n")

    def evaluate_all_datasets(self, dataset_list, mode_list, method_list, pro_dict, crop):
        for method in method_list:
            for dataset in dataset_list:
                print(f'Now dealing with dataset: {dataset}')
                for mode in mode_list:
                    gt_path = os.path.join('/data/8T/cby/g2_dataset/', dataset, 'val')
                    for pro in pro_dict[mode]:
                        log_path = os.path.join('/data/8T/cby/g2_dataset/', dataset, mode, 'logs', f'logs_{method}_{pro}.txt')
                        check_parent_path(log_path)
                        depth_path = os.path.join('/data/8T/cby/g2_dataset/', dataset, mode, method + '_' + str(pro))
                        self.evaluate_depth_dataset(depth_path, gt_path, pro, crop, log_path)
def check_parent_path(file_path):
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"父路径 '{parent_dir}' 不存在，已创建。")
   
        
        
def check_path(file_path):
    parent_dir = file_path
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"路径 '{parent_dir}' 不存在，已创建。")
        
def delete_file(file_path):
    if os.path.exists(file_path):  # 检查文件是否存在
        os.remove(file_path)        # 删除文件
        print(f"文件 '{file_path}' 已删除。")
    else:
        print(f"文件 '{file_path}' 不存在。")

if __name__ == '__main__':
    current_time = datetime.datetime.now()
    print("Evaluate Started Time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
    mode_list = ['result']
    dataset_list = ['nyu', 'redweb', 'ETH3D', 'Ibims', 'VKITTI', 'Matterport3D', 'UnrealCV', 'KITTI']
    # method_list = [
    #     'rz_sb_mar_GuideNet_DIODE_HRWSI', 
    #     'rz_sb_mar_MDAnet_DIODE_HRWSI', 
    #     'rz_sb_mar_LRRU_DIODE_HRWSI', 
    #     'rz_sb_mar_EMDC_DIODE_HRWSI', 
    #     'rz_sb_mar_TWISE_DIODE_HRWSI', 
    #     'rz_sb_mar_SDCM_DIODE_HRWSI', 
    #     'rz_sb_mar_PEnet_DIODE_HRWSI', 
    #     'rz_sb_mar_ReDC_DIODE_HRWSI',
    #     'rz_sb_mar_CFormer_DIODE_HRWSI',
    #     'rz_sb_mar_NLSPN_DIODE_HRWSI_60',
    #     'rz_sb_mar_g2_DIODE_HRWSI',
    #     'rz_sb_mar_sfv2_DIODE_HRWSI',  
    #     'rz_sb_mar_CFormer_KITTI',
    #     'rz_sb_mar_EMDC',
    #     'rz_sb_mar_LRRU',
    #     'rz_sb_mar_GuideNet','rz_sb_mar_ReDC', 'rz_sb_mar_MDAnet',
    #     'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_PEnet',
    #     'rz_sb_mar_SDCM','rz_sb_mar_TWISE'] 
    method_list = ['rz_sb_mar_g2_DIODE_HRWSI','rz_sb_mar_sfv2_DIODE_HRWSI']
    pro_dict = {'result': [0.01, 0.1, 0.2, 0.5, 0.7, 1.01, 1.04, 1.08, 1.016, 1.032, 1.064, 1.0128]}
    crop = False
    
    datasets = ['nyu', 'redweb', 'ETH3D', 'Ibims', 'VKITTI', 'Matterport3D', 'UnrealCV', 'KITTI']
    for dataset in datasets:
        log_dir = os.path.join('/data/8T/cby/g2_dataset/', dataset ,'result','logs')
        check_path(log_dir)
        for file in os.listdir(log_dir):
            if file.startswith('logs_rz') and file.endswith('.txt'):
                delete_file(os.path.join('/data/8T/cby/g2_dataset/', dataset,'result',file))
                print(file)
    evaluator = DepthDatasetEvaluation()
    evaluator.evaluate_all_datasets(dataset_list, mode_list, method_list, pro_dict, crop)
