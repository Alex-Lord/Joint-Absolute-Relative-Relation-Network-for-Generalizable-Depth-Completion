import sys
import os
import glob
# append module path
# sys.path.append('/data/4TSSD/cby/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime
from PIL import Image
import numpy as np
from application.application_utils import DepthEvaluation

from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
from torch import Tensor
import torch
print(sys.path)

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
        else:
            depth = (np.array(data) / 255.).astype(np.float32)
            depth = np.clip(depth, 0, 1)
        depth = torch.from_numpy(depth).unsqueeze(0)
        data.close()
    return depth
from tqdm import tqdm


class DepthDatasetEvaluation(object):
    def __init__(self):
        super(DepthDatasetEvaluation, self).__init__()

    @staticmethod
    def evaluate_depth_dataset(depth_path, gt_path, pro):
        starmse = 0.0
        ord_error = 0.0
        rmse = 0.0
        irmse = 0.0
        imae = 0.0
        mae = 0.0
        rel = 0.0
        count = 0.0
        png_files = glob.glob(depth_path + '/**/*.png',recursive=True)
        for depth_files in tqdm(png_files, desc='eva'):
            count += 1.0
            # print(f'depth_files={depth_files}')
            # depth should be nonzero
            depth_array = np.array(Image.open(depth_files)).astype(np.float32)
            depth = np.clip(depth_array.astype(np.float32), 1.0, 255.0)
            if 'DIODE_gt' in str(depth_files):
                gt_files = str(depth_files).replace(depth_path, gt_path, 1)
                gt_files = gt_files.replace('.png', 'crop.png')
                gt = ((np.array(depth_read(gt_files)) * 255.).squeeze()).astype(np.float32)
                if (gt == np.zeros_like(gt)).all(): 
                    count -= 1.0
                    continue
   
            else:
                gt_files = str(depth_files).replace(depth_path, gt_path, 1)
                gt_files = gt_files.replace('.png', 'crop.png')
                gt = np.clip(np.array(Image.open(gt_files)).astype(np.float32), 0., 255.)
                if (gt == np.zeros_like(gt)).all():
                    count -= 1.0
                    continue
            if crop:
                depth, gt = depth[:256, :256], gt[:256, :256]
            if pro == 0.0:
                # print(f'depth={depth_files},gt={gt_files}')
                # print(f'depth.shape={depth.shape}, gt.shape={gt.shape}')
                starmse += DepthEvaluation.starmse(depth, gt)
                # ord_error += DepthEvaluation.oe(depth, gt)
                ord_error = 0
            else:
                rmse += DepthEvaluation.rmse(depth, gt)
                rel += DepthEvaluation.absRel(depth, gt)
                irmse += DepthEvaluation.irmse(depth, gt)
                imae += DepthEvaluation.imae(depth, gt)
                mae += DepthEvaluation.mae(depth, gt)
                # rel = 0
                # irmse = 0
                # mae = 0

        if pro == 0.0:
            starmse /= count
            # ord_error /= count
            ord_error = 0
            return starmse, ord_error
        else:
            rmse /= count
            rel /= count
            irmse /= count
            imae /= count
            mae /= count
            return rmse, rel, irmse, imae, mae

    def evaluate_all_datasets(self):
        for method in method_list:
            for dataset in dataset_list:
                print(f'Now dealing with dataset:{dataset}')
                for mode in mode_list:
                    gt_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, 'val')  # nyu需要加入val后缀
                    log_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, mode, 'logs_ablation.txt')
                    # 0-100
                    for pro in pro_dict[mode]:
                        print(method + '_' + dataset + '_' + str(pro))
                        depth_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, mode, method + '_' + str(pro))
                        if pro == 0.0:
                            # print(f'depth_path={depth_path},gt_path={gt_path}')
                            starmse, ord_error = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ' : OE= ' + str(ord_error) +
                                    ', SRMSE= {:.4f}'.format(starmse) + '\n'
                                )
                        else:
                            rmse, rel, irmse, imae, mae = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ': AbsRel= ' + str(rel) +
                                    ', RMSE= {:.4f}'.format(rmse) + 
                                    ', iRMSE= {:.6f}'.format(irmse) + 
                                    ', iMAE= {:.6f}'.format(imae) + 
                                    ', MAE= {:.4f}'.format(mae) +
                                    '\n'
                                )

    def evaluate_baseline(self):
        est_dict = {'mode_list': ['result'], 'pro_list': [0.0]}
        comp_dict = {'mode_list': ['result', 'result_nb'], 'pro_list': [0.001, 0.01, 0.1]}
        sr_dict = {'mode_list': ['result', 'result_nb'], 'pro_list': [1.0]}
        baseline = {
            'SGRL': est_dict, 'MiDas': est_dict, 'Mega': est_dict, 'DMF': est_dict,  # relative depth estimation
            'TWISE': comp_dict, 'NLSPN': comp_dict, 'GuideNet': comp_dict, 'SDCM': comp_dict,  # depth completion
            'PMBAN': sr_dict, 'PDSR': sr_dict, 'JIFF': sr_dict, 'DKN': sr_dict  # depth super-resolution
        }
        # dataset_list = ['Sintel','DIODE', 'ETH3D', 'Ibims', 'NYUV2', 'KITTI']
        dataset_list = ['Ibims', 'NYUV2', 'KITTI']
        for key, value in baseline.items():
            method = str(key)
            for dataset in dataset_list:
                for mode in value['mode_list']:
                    gt_path = os.path.join('../Test_datasets', dataset, 'gt')
                    log_path = os.path.join('../Test_datasets', dataset, mode, 'logs_baseline.txt')
                    for pro in value['pro_list']:
                        print(method + '_' + dataset + '_' + str(pro))
                        depth_path = os.path.join('../Test_datasets', dataset, mode, method + '_' + str(pro))
                        if pro == 0.0:
                            starmse, ord_error = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ' : OE= ' + str(ord_error) +
                                    ', SRMSE= {:.4f}'.format(starmse) + '\n'
                                )
                        else:
                            rmse, rel, irmse, imae, mae = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ': AbsRel= ' + str(rel) +
                                    ', RMSE= ' + str(rmse) + 
                                    ', iRMSE= ' + str(irmse) + 
                                    ', iMAE= {:.6f}'.format(imae) + 
                                    ', MAE= ' + str(mae) + 
                                    '\n'
                                )
                        

    @staticmethod
    def inverse_evaluate_depth_dataset(depth_path, gt_path, pro):
        starmse = 0.0
        rmse = 0.0
        irmse = 0.0
        imae = 0.0
        mae = 0.0
        count = 0.0
        png_files = glob.glob(depth_path + '/**/*.png',recursive=True)
        for depth_files in tqdm(png_files, desc='inverse_eva'):
            count += 1.0
            # depth should be nonzero
            depth = np.clip(np.array(Image.open(depth_files)), 1.0, 255.0).astype(np.float32)
            gt_files = str(depth_files).replace(depth_path, gt_path, 1)
            gt_files = gt_files.replace('.png', 'crop.png')
            gt = np.clip(np.array(Image.open(gt_files)).astype(np.float32), 0., 255.)
            if (gt == np.zeros_like(gt)).all():
                count -= 1.0

            if crop:
                depth, gt = depth[:256, :256], gt[:256, :256]
            if pro != 0.0:
                starmse += DepthEvaluation.starmse(depth, gt)
            else:
                rmse += DepthEvaluation.rmse(depth, gt)
                irmse += DepthEvaluation.irmse(depth, gt)
                imae += DepthEvaluation.imae(depth, gt)
                mae += DepthEvaluation.mae(depth, gt)
                # rmse = 0
        
        if pro != 0.0:
            starmse /= count
            return starmse
        else:
            rmse /= count
            irmse /= count
            imae /= count
            mae /= count
            return rmse, irmse, imae, mae

    def inverse_evaluate_all_datasets(self):

        for method in method_list:
            for dataset in dataset_list:
                print(f'Now dealing with dataset:{dataset}')
                for mode in mode_list:
                    gt_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, 'val')  # nyu需要加入val
                    log_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, mode, 'logs_absrel.txt')

                    # 0-100
                    for pro in pro_dict[mode]:
                        depth_path = os.path.join('/data1/Chenbingyuan/Depth-Completion/g2_dataset/', dataset, mode, method + '_' + str(pro))
                        # print(f'depth_path = {depth_path}')
                        if pro != 0.0:
                            starmse = self.inverse_evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ' : SRMSE= ' + str(starmse) + '\n'
                                )
                        else:
                            rmse, irmse, imae,mae = self.inverse_evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ': RMSE= {:.4f}'.format(rmse) + 
                                    ': iRMSE= {:.6f}'.format(irmse) + 
                                    ': iMAE= {:.6f}'.format(imae) + 
                                    ': MAE= {:.4f}'.format(mae) + 
                                    '\n'
                                    
                                )
                        print(method + '_' + dataset + '_' + str(pro))

def delete_txt_file(dataset_list):
    for data_set in dataset_list:
        txt_file_1 = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/'+data_set+'/result/'+ 'logs_ablation.txt'
        txt_file_2 = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/'+data_set+'/result/'+ 'logs_absrel.txt'
        # 检查文件是否存在
        # print(txt_file_1)
        if os.path.exists(txt_file_1):
            # 删除文件
            os.remove(txt_file_1)
            print(f'removed {txt_file_1}')
        if os.path.exists(txt_file_2):
            # 删除文件
            os.remove(txt_file_2)
            print(f'removed {txt_file_2}')
    

if __name__ == '__main__':
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Evaluate Started Time:", formatted_time)

    # mode_list = ['result', 'result_nb']
    mode_list = [ 'result']
    # mode_list = ['lines_result']
    # mode_list = ['result_very_sparse']
    # mode_list = ['result_very_sparse_same_seg', 'result_very_sparse_differ_seg']
    # dataset_list = ['KITTI','nyu', 'redweb','ETH3D','Ibims', 'VKITTI']
    # dataset_list = ['Matterport3D', 'UnrealCV']
    dataset_list = ['KITTI','nyu','redweb','ETH3D','Ibims', 'VKITTI', 'Matterport3D', 'UnrealCV']
    # dataset_list = ['KITTI']
    # dataset_list = ['nyu','redweb','ETH3D','Ibims', 'VKITTI', 'Matterport3D', 'UnrealCV']

    # method_list = ['rz_sb_mar_CFormer_DIODE_HRWSI','rz_sb_mar_CFormer_KITTI','rz_sb_mar_EMDC',
    #                'rz_sb_mar_LRRU','rz_sb_mar_NLSPN_DIODE_HRWSI_60','rz_sb_mar_SDCM',
    #                'rz_sb_mar_GuideNet','rz_sb_mar_ReDC', 'rz_sb_mar_MDAnet',
    #                'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_PEnet',
    #                'rz_sb_mar_SDCM','rz_sb_mar_TWISE','rz_sb_mar_g2_DIODE_HRWSI','rz_sb_mar_sfv2_DIODE_HRWSI',
    #                'rz_sb_mar_sfv2_DIODE_HRWSI_no_f', 'rz_sb_mar_sfv2_DIODE_HRWSI_no_s',
    #                'rz_sb_mar_sfv2_DIODE_HRWSI_no_p','rz_sb_mar_sfv2_DIODE_HRWSI_only_p', 'rz_sb_mar_sfv2_DIODE_HRWSI_only_s',
    #                'rz_sb_mar_sfv2_DIODE_HRWSI_only_f','rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI']
    # method_list = ['rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI']
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_bn']
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
    #     'rz_sb_mar_sfv2_DIODE_HRWSI'
    # ]
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI','rz_sb_mar_sfv2_KITTI_2','rz_sb_mar_CFormer_DIODE_HRWSI','rz_sb_mar_CFormer_KITTI']
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI','rz_sb_mar_sfv2_DIODE_HRWSI_f0505','rz_sb_mar_sfv2_DIODE_HRWSI_f22','rz_sb_mar_sfv2_DIODE_HRWSI_s005','rz_sb_mar_sfv2_DIODE_HRWSI_s05']
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_f0505','rz_sb_mar_sfv2_DIODE_HRWSI','rz_sb_mar_sfv2_DIODE_HRWSI_f22','rz_sb_mar_sfv2_DIODE_HRWSI_s005','rz_sb_mar_sfv2_DIODE_HRWSI_s05']
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
    #     'rz_sb_mar_sfv2_DIODE_HRWSI', '  
    # ]
    
    # 全部
    method_list = [
        'rz_sb_mar_GuideNet_DIODE_HRWSI', 
        'rz_sb_mar_MDAnet_DIODE_HRWSI', 
        'rz_sb_mar_LRRU_DIODE_HRWSI', 
        'rz_sb_mar_EMDC_DIODE_HRWSI', 
        'rz_sb_mar_TWISE_DIODE_HRWSI', 
        'rz_sb_mar_SDCM_DIODE_HRWSI', 
        'rz_sb_mar_PEnet_DIODE_HRWSI', 
        'rz_sb_mar_ReDC_DIODE_HRWSI',
        'rz_sb_mar_g2_DIODE_HRWSI',
        'rz_sb_mar_sfv2_DIODE_HRWSI',
        'rz_sb_mar_CFormer_DIODE_HRWSI','rz_sb_mar_CFormer_KITTI','rz_sb_mar_EMDC', 'rz_sb_mar_BPnet',
        'rz_sb_mar_LRRU','rz_sb_mar_NLSPN_DIODE_HRWSI_60',
    
        'rz_sb_mar_GuideNet','rz_sb_mar_ReDC', 'rz_sb_mar_MDAnet',
        'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_PEnet','rz_sb_mar_G2_Mono', 'rz_sb_mar_JARRN_full_05line_05point', 'rz_sb_mar_JARRN_mixed_05point_05line',
        'rz_sb_mar_SDCM','rz_sb_mar_TWISE',
        'rz_sb_mar_sfv2_DIODE_HRWSI_no_f', 'rz_sb_mar_sfv2_DIODE_HRWSI_no_s',
        'rz_sb_mar_sfv2_DIODE_HRWSI_no_p','rz_sb_mar_sfv2_DIODE_HRWSI_only_p', 'rz_sb_mar_sfv2_DIODE_HRWSI_only_s',
        'rz_sb_mar_sfv2_DIODE_HRWSI_only_f'
    ]
    # method_list = ['rz_sb_mar_BPnet', 'rz_sb_mar_sfv2_DIODE_HRWSI', 'rz_sb_mar_JARRN_mixed_05point_05line']
    # method_list = ['rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI']
    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_LSM']
    # method_list = ['rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2', 'rz_sb_mar_SDCM','rz_sb_mar_PEnet','rz_sb_mar_ReDC','rz_sb_mar_CFormer_KITTI', 'rz_sb_mar_EMDC', 
    #                'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_TWISE','rz_sb_mar_MDAnet', 'rz_sb_mar_LRRU', 'rz_sb_mar_GuideNet']
    # method_list = ['rz_sb_mar_JARRN_60LiDAR']
    # method_list = ['rz_sb_mar_CFormer_KITTI',
    #     'rz_sb_mar_EMDC',
    #     'rz_sb_mar_LRRU',
    #     'rz_sb_mar_GuideNet','rz_sb_mar_ReDC', 'rz_sb_mar_MDAnet',
    #     'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_PEnet',
    #     'rz_sb_mar_SDCM','rz_sb_mar_TWISE']
    
    
    # method_list = ['rz_sb_mar_G2_Mono','rz_sb_mar_JARRN']
    # method_list = ['rz_sb_mar_JARRN_mixed_line_point','rz_sb_mar_JARRN_mixed_1line_1point','rz_sb_mar_JARRN_mixed_07point_03line']
    # method_list = ['rz_sb_mar_JARRN_mixed_05point_05line']
    # method_list = ['rz_sb_mar_G2_Mono','rz_sb_mar_JARRN_nosoftmax']
    # pro_dict = {'lines_result': [0,1,4,16,64]}  # 线数
    # pro_dict = {'result': [0, 283/102400, 1129/102400, 3708/102400, 14691/102400]}
    # pro_dict = {'result_very_sparse':[0,1,2,3,10,20,50,100]}
    # pro_dict = {'result': [0.001,0.01,0.1,0.2,0.5,0.7]}
    # pro_dict = {'result': [1.0]}
    pro_dict = {'result': [0.01,0.1,0.2,0.5,0.7, 1.04,1.08,1.016, 1.032,1.064, 1.0128]}
    # pro_dict = {'result': [0.2]}
    # pro_dict = {'result': [0.1,0.2,0.7]}
    # pro_dict = {'result': [0.01,0.1,0.2,0.5,0.7]}
    # pro_dict = {'result_very_sparse_same_seg':[0,1,2,3,10,20,50,100], 'result_very_sparse_differ_seg':[0,1,2,3,10,20,50,100],}
    # from demo import pro_dict, crop
    # from demo import dataset_list as dataset_list
    crop = False
    # delete_txt_file(dataset_list)
    if crop:
        print('进行256*256尺寸的评估')
    else:
        print('进行320*320尺寸的评估')
    eva = DepthDatasetEvaluation()
    eva.evaluate_all_datasets()
    # eva.evaluate_baseline()
    # eva.inverse_evaluate_all_datasets()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Evaluate Ended Time:", formatted_time)
