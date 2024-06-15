import time
import torch
import torch.nn as nn
import torch.optim as optim
# append module path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from tqdm import tqdm
import torch
import sys
import contextlib
import torch


def get_model(method):
    if 'rz' not in method:
        args_rezero = False
    else :
        args_rezero = True
    if method == 'rz_sb_mar_JARRN_G2V2_full':
        from sfv2_networks import JARRN_G2V2
        network = JARRN_G2V2()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_JARRN_full_G2V2/models/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
    if method == 'rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI':
        from sfv2_networks import JARRN_nosfp_direct_2branch
        network = JARRN_nosfp_direct_2branch(rezero=args_rezero)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
    if method == 'rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2':
        from sfv2_networks import JARRN_nosfp_direct_2branch_2
        network = JARRN_nosfp_direct_2branch_2(rezero=args_rezero)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_nosfp_direct_2branch_DIODE_HRWSI_2/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
    if method == 'rz_sb_mar_JARRN_full_05line_05point':
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_full_05line_05point/models/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  

    if method == 'rz_sb_mar_JARRN_nosoftmax':
        from sfv2_networks import JARRN_nosoftmax
        network = JARRN_nosoftmax(rezero=args_rezero)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar/JARRN_nosoftmax/models/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN_noSoftmax
    if method == 'rz_sb_mar_JARRN_mixed_07point_03line':
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_0.3line_0.7point/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
    if method == 'rz_sb_mar_JARRN_mixed_1point_1line':
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_1line_1point/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
    if method == 'rz_sb_mar_JARRN_mixed_line_point':
        # 只是用采点方式，不改变GT
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_line_point/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) #  JARRN
    if method == 'rz_sb_mar_JARRN_mixed_05point_05line':
        # 只是用采点方式，不改变GT
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/result_JARRN/_rz_sb_mar_JARRN_mixed_0.5line_0.5point_fixed/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) #  JARRN


    if method == 'rz_sb_mar_JARRN':
        from sfv2_networks import JARRN
        network = JARRN(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_3/models/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  JARRN
    if method == 'rz_sb_mar_g2_all_retrain':
        from sfv2_networks import G2_Mono
        network = G2_Mono(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/Abs_Rel_train_logs/train_logs_rz_sb_mar_mar_G2Mono/models/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
    if method == 'rz_sb_mar_sfv2_KITTI_2':
        from sfv2_networks import sfv2_UNet_KITTI
        network = sfv2_UNet_KITTI(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_KITTI_2/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI':
        from sfv2_networks import sfv2_UNet
        network = sfv2_UNet(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_LSM':
        from sfv2_networks import sfv2_UNet_LSM
        network = sfv2_UNet_LSM(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f22':
        from sfv2_networks import sfv2_UNet_f22
        network = sfv2_UNet_f22(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f22/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2

    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f11':
        from sfv2_networks import sfv2_UNet_f11
        network = sfv2_UNet_f11(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f11/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_f0505':
        from sfv2_networks import sfv2_UNet_f0505
        network = sfv2_UNet_f0505(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_f0505/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_s005':
        from sfv2_networks import sfv2_UNet_s005
        network = sfv2_UNet_s005(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_s005/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_s05':
        from sfv2_networks import sfv2_UNet_s05
        network = sfv2_UNet_s05(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_s05/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2

    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_relative_loss':
        from sfv2_networks import sfv2_UNet
        network = sfv2_UNet(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_relative/models/epoch_24.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_tiny':
        from sfv2_networks import sfv2_UNet_tiny
        network = sfv2_UNet_tiny(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Tiny/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_small':
        from sfv2_networks import sfv2_UNet_small
        network = sfv2_UNet_small(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Small/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    if method == 'rz_sb_mar_sfv2_DIODE_HRWSI_large':
        from sfv2_networks import sfv2_UNet_large
        network = sfv2_UNet_large(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_Large/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_unscale':
        from sfv2_networks import sfv2_UNet_unscale
        network = sfv2_UNet_unscale(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    elif method == 'bn_sb_mar_sfv2_DIODE_HRWSI':
        from sfv2_networks import sfv2_UNet_bn
        network = sfv2_UNet_bn(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_bn_sb_mar_sfv2_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_bn':
        from sfv2_networks import sfv2_UNet_bn
        network = sfv2_UNet_bn(rezero=False)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_bn_retrain0402/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  #  Sfv2
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_f':
        from sfv2_networks import sfv2_UNet_no_f
        network = sfv2_UNet_no_f(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_f/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_s':
        from sfv2_networks import sfv2_UNet_no_s
        network = sfv2_UNet_no_s(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_s/models/epoch_58.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_p':
        from sfv2_networks import sfv2_UNet_no_p
        network = sfv2_UNet_no_p(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_p/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_f':
        from sfv2_networks import sfv2_UNet_only_f
        network = sfv2_UNet_only_f(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_f/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_s':
        from sfv2_networks import sfv2_UNet_only_s
        network = sfv2_UNet_only_s(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_s/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_only_p':
        from sfv2_networks import sfv2_UNet_only_p
        network = sfv2_UNet_only_p(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_only_p/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
    elif method == 'rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI':
        from sfv2_networks import sfv2_UNet
        network = sfv2_UNet(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_L1L2_loss_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)  
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_no_blur':
        from sfv2_networks import sfv2_UNet
        network = sfv2_UNet(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_no_blur/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True)
    elif method == 'rz_sb_mar_sfv2_DIODE_HRWSI_2':
        from sfv2_networks import sfv2_UNet
        network = sfv2_UNet(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_sfv2_DIODE_HRWSI_2/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_CFormer_DIODE_HRWSI_2' or method == 'rz_sb_mar_CFormer_DIODE_HRWSI':
        from sfv2_networks import CFormer_DIODE_HRWSI
        network = CFormer_DIODE_HRWSI()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_CFormer_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_NLSPN_DIODE_HRWSI_60' or method == 'rz_sb_mar_NLSPN_DIODE_HRWSI':
        from sfv2_networks import NLSPN_DIODE_HRWSI
        network = NLSPN_DIODE_HRWSI()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_NLSPN_DIODE_HRWSI_60/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_g2_DIODE_HRWSI':
        from sfv2_networks import g2_UNet
        network = g2_UNet()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_g2_DIODE_HRWSI/models/epoch_60.pth'
        network = network.cuda()
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    if method == 'rz_sb_mar_G2_Mono':
        # G2_Mono
        from sfv2_networks import G2_Mono
        network = G2_Mono(rezero=args_rezero)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/G2_Mono/epoch_100.pth'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['network'],strict=True) 
    elif method == 'rz_sb_mar_CFormer_KITTI':
        # CFormer
        from src.baselines.CFormer.model.completionformer import CompletionFormer, check_args
        from src.baselines.CFormer.model.config import args as args_cformer
        network = CompletionFormer(args_cformer)  
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/CFormer/model/KITTIDC_L1L2.pt'
        # model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/CFormer/model/NYUv2.pt'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'],strict=True)  # CFormer
    elif method == 'rz_sb_mar_PEnet':
        # PEnet
        from src.baselines.PEnet.model.model import PENet_C2
        from src.baselines.PEnet.model.config import args as args_penet
        network = PENet_C2(args_penet) 
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/PEnet/model/pe.pth.tar'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'],strict=False)  # PEnet
        
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
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'],strict=False)  # PEnet
    elif method == 'rz_sb_mar_ReDC_DIODE_HRWSI':
        from sfv2_networks import ReDC_retrain
        network = ReDC_retrain(rezero=True)
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_redc_DIODE_HRWSI/models/redc_epoch_60.pth'
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
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
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'])  # TWISE
    elif method == 'rz_sb_mar_TWISE_DIODE_HRWSI':
        from sfv2_networks import TWISE_retrain
        network = TWISE_retrain(rezero=True)
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_TWISE_DIODE_HRWSI/models/epoch_60.pth'
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
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
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'],strict=True)  # NLSPN
    elif method == 'rz_sb_mar_SDCM':
        # SDCM
        from src.baselines.SDCM.model import DepthCompletionNet
        from src.baselines.SDCM.config import args as args_SDCM

        network = DepthCompletionNet(args_SDCM)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/SDCM/model_best.pth.tar'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['model'])  # SDCM
    elif method == 'rz_sb_mar_SDCM_DIODE_HRWSI':
        from sfv2_networks import SDCM_retrain
        network = SDCM_retrain(rezero=True)
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_SDCM_DIODE_HRWSI/models/epoch_60.pth'
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_MDAnet':
        # MDAnet
        from src.baselines.MDANet.modules.net import network as MDAnet
        network = MDAnet()
        network = torch.nn.DataParallel(network)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/MDANet/results/quickstart/checkpoints/net-best.pth.tar'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['net'])  # MDAnet
    elif method == 'rz_sb_mar_MDAnet_DIODE_HRWSI':
        from sfv2_networks import MDAnet_retrain
        network = MDAnet_retrain(rezero=True)
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_MDAnet_DIODE_HRWSI/models/epoch_60.pth'
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
    elif method == 'rz_sb_mar_EMDC':
        # EMDC
        from src.baselines.EMDC.models.EMDC import emdc
        network = emdc(depth_norm=False)
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/EMDC/checkpoints/milestone.pth.tar'
        network = network.cuda()
        # network.load_state_dict(torch.load(model_dir, map_location='cuda:0')['state_dict'])  # EMDC
    elif method == 'rz_sb_mar_EMDC_DIODE_HRWSI':
        # EMDC
        from sfv2_networks import EMDC_retrain
        network = EMDC_retrain(rezero=True)
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/results/_rz_sb_mar_emdc_DIODE_HRWSI/models/epoch_60.pth'
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 
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
        # network.load_state_dict(on_load_checkpoint(torch.load(model_dir, map_location='cuda:0'))['network_state_dict'],strict=True) 

    elif method == 'rz_sb_mar_BPnet':
        from src.baselines.BPnet.models.BPNet import Net as BPnetModel
        network = BPnetModel()
        network = network.cuda()
        model_dir = '/data1/Chenbingyuan/Depth-Completion/src/baselines/BPnet/BP_KITTI/result_ema.pth'
        cp = torch.load(model_dir, map_location='cuda:0')
        # network.load_state_dict(cp['net'], strict=True)
        
    return network

if __name__ == "__main__":
    mode = 'train' # or train infer

    # method_list = ['rz_sb_mar_sfv2_DIODE_HRWSI_tiny','rz_sb_mar_sfv2_DIODE_HRWSI_small','rz_sb_mar_sfv2_DIODE_HRWSI', 'rz_sb_mar_sfv2_DIODE_HRWSI_large']
    method_list = ['rz_sb_mar_SDCM','rz_sb_mar_CFormer_KITTI', 'rz_sb_mar_EMDC', 
                   'rz_sb_mar_NLSPN_KITTI','rz_sb_mar_TWISE','rz_sb_mar_PEnet','rz_sb_mar_ReDC',] # completionformer 一个环境就可以解决
    # method_list = ['rz_sb_mar_MDAnet'] # torch1.7
    method_list = ['rz_sb_mar_LRRU'] # LRRU_new
    # method_list = ['rz_sb_mar_GuideNet'] # cuda121
    method_list = ['rz_sb_mar_G2_Mono']
    
    for method in method_list:
        
        Model = get_model(method).cuda()
        if mode == 'train':
            Model.train()  # 设置模型为训练模式
            criterion = nn.MSELoss()  # 假设损失函数是MSE
            optimizer = optim.Adam(Model.parameters(), lr=0.001)
        else:
            Model.eval()  # 设置模型为评估模式

        # 生成一些示例数据
        batch_size = 10
        channels = 3
        height = 320
        width = 320

        rgb = torch.randn(batch_size, channels, height, width).cuda()
        point = torch.randn(batch_size, 1, height, width).cuda()
        hole_point = torch.randn(batch_size, 1, height, width).cuda()
        target = torch.randn(batch_size, 1, height, width).cuda()
        
        if method == 'rz_sb_mar_SDCM':
            rgb = torch.randn(batch_size, 1, height, width).cuda()
        if method == 'rz_sb_mar_PEnet' or method == 'rz_sb_mar_ReDC':
            K = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 1]]]).cuda()

        # 预热GPU，运行几个批次
        with torch.no_grad() if mode == 'infer' else contextlib.nullcontext():
            for _ in tqdm(range(10)):
                if method == 'rz_sb_mar_LRRU':
                    output = Model(rgb, point, hole_point)
                    output = output['results'][-1]
                elif method == 'rz_sb_mar_GuideNet':
                    output, = Model(rgb, point)
                elif method == 'rz_sb_mar_MDAnet':
                    output = Model(point, rgb)[0]
                elif method == 'rz_sb_mar_SDCM':
                    output = Model(point, rgb)
                elif method == 'rz_sb_mar_PEnet'or method == 'rz_sb_mar_ReDC':
                    output = Model(rgb, point, K)
                elif method == 'rz_sb_mar_CFormer_KITTI':
                    output = Model(rgb, point)['pred']
                elif method == 'rz_sb_mar_EMDC':
                    output = Model(rgb, point)[0]
                elif method == 'rz_sb_mar_NLSPN_KITTI':
                    output = Model(rgb, point)
                elif method == 'rz_sb_mar_TWISE':
                    output, _, _ = Model(point, rgb)
                elif 'sfv2' in method:
                    output, _, _, _ = Model(rgb, point, hole_point)
                else:
                    output = Model(rgb, point, hole_point)
                
                if mode == 'train':
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # 正式计时
        num_batches = 100
        total_time = 0

        with torch.no_grad() if mode == 'infer' else contextlib.nullcontext():
            for _ in range(num_batches):
                start_time = time.time()
                
                if method == 'rz_sb_mar_LRRU':
                    output = Model(rgb, point, hole_point)
                    output = output['results'][-1]
                elif method == 'rz_sb_mar_GuideNet':
                    output, = Model(rgb, point)
                elif method == 'rz_sb_mar_MDAnet':
                    output = Model(point, rgb)[0]
                elif method == 'rz_sb_mar_SDCM':
                    output = Model(point, rgb)
                elif method == 'rz_sb_mar_PEnet'or method == 'rz_sb_mar_ReDC':
                    output = Model(rgb, point, K)
                elif method == 'rz_sb_mar_CFormer_KITTI':
                    output = Model(rgb, point)['pred']
                elif method == 'rz_sb_mar_EMDC':
                    output = Model(rgb, point)[0]
                elif method == 'rz_sb_mar_NLSPN_KITTI':
                    output = Model(rgb, point)
                elif method == 'rz_sb_mar_TWISE':
                    output, _, _ = Model(point, rgb)
                elif 'sfv2' in method:
                    output, _, _, _ = Model(rgb, point, hole_point)
                else:
                    output = Model(rgb, point, hole_point)
                
                if mode == 'train':
                    loss = criterion(output, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                end_time = time.time()
                total_time += (end_time - start_time)

        average_time_per_batch = total_time / num_batches
        samples_per_second = batch_size / average_time_per_batch
        print("*" * 10 + method + "*" * 10)
        print(f'Average Time per Batch: {average_time_per_batch:.4f} seconds')
        print(f'Samples per Second: {samples_per_second:.2f}')
        print("*" * 30)