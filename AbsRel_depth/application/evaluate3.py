import sys
import os

# append module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from PIL import Image
import numpy as np
from pathlib import Path
from application.application_utils import DepthEvaluation


class DepthDatasetEvaluation(object):
    def __init__(self):
        super(DepthDatasetEvaluation, self).__init__()

    @staticmethod
    def evaluate_depth_dataset(depth_path, gt_path, pro):
        starmse = 0.0
        ord_error = 0.0
        rmse = 0.0
        rel = 0.0
        count = 0.0

        for depth_files in Path(depth_path).rglob('*.png'):
            count += 1.0

            # depth should be nonzero
            depth = np.clip(np.array(Image.open(depth_files)).astype(np.float32), 1., 65535.)

            gt_files = str(depth_files).replace(depth_path, gt_path, 1)
            gt = np.array(Image.open(gt_files)).astype(np.float32)

            starmse += DepthEvaluation.starmse(depth, gt)
            if pro == 0.0:
                ord_error += DepthEvaluation.oe(depth, gt)
            else:
                rmse += DepthEvaluation.rmse(depth, gt)
                rel += DepthEvaluation.absRel(depth, gt)

        starmse /= count
        if pro == 0.0:
            ord_error /= count
            return starmse, ord_error
        else:
            rmse /= count
            rel /= count
            return rmse, rel

    def evaluate_all_datasets(self):
        mode_list = ['result', 'result_nb']
        # dataset_list = ['Sintel','DIODE', 'ETH3D', 'Ibims', 'NYUV2', 'KITTI']
        dataset_list = ['Ibims', 'NYUV2', 'KITTI']
        method_list = ['bn_mar']
        pro_dict = {'result': [0.0, 0.1, 0.01, 0.001, 1.0], 'result_nb': [0.1, 0.01, 0.001, 1.0]}

        for method in method_list:
            for dataset in dataset_list:
                for mode in mode_list:
                    gt_path = os.path.join('../Test_datasets', dataset, 'gt')
                    log_path = os.path.join('../Test_datasets', dataset, mode, 'logs_ablation.txt')

                    # 0-100
                    for pro in pro_dict[mode]:
                        depth_path = os.path.join('../Test_datasets', dataset, mode, method + '_' + str(pro))
                        if pro == 0.0:
                            starmse, ord_error = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ' : OE=' + str(ord_error) +
                                    ', SRMSE=' + str(starmse) + '\n'
                                )
                        else:
                            rmse, rel = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ': AbsRel=' + str(rel) +
                                    ', RMSE=' + str(rmse) + '\n'
                                )
                        print(method + '_' + dataset + '_' + str(pro))

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
                        depth_path = os.path.join('../Test_datasets', dataset, mode, method + '_' + str(pro))
                        if pro == 0.0:
                            starmse, ord_error = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ' : OE=' + str(ord_error) +
                                    ', SRMSE=' + str(starmse) + '\n'
                                )
                        else:
                            rmse, rel = self.evaluate_depth_dataset(depth_path, gt_path, pro)
                            with open(log_path, 'a') as log_file:
                                log_file.write(
                                    method + '_' + str(pro) +
                                    ': AbsRel=' + str(rel) +
                                    ', RMSE=' + str(rmse) + '\n'
                                )
                        print(method + '_' + dataset + '_' + str(pro))


if __name__ == '__main__':
    eva = DepthDatasetEvaluation()
    eva.evaluate_all_datasets()
    # eva.evaluate_baseline()
