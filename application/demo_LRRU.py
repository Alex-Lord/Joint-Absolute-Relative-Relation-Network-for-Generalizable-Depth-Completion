# append module path
import sys
import io
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import copy
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse
from PIL import Image
from pathlib import Path
import setproctitle
setproctitle.setproctitle("PyThon")


from src import str2bool
from application.application_utils import RGBPReader
import torch
from torch.backends import cudnn
from tqdm import tqdm
import numpy as np
import glob
import sys


# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def on_load_checkpoint(checkpoint):
    keys_list = list(checkpoint['network_state_dict'].keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('module.', '')
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
        default=r'/data1/Chenbingyuan/Trans_G2/g2_dataset/DIODE',
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
    if 'sfv2' in network_type or 'G2_Mono' in network_type or 'DIODE' in network_type:
        # SfV2  G2_Mono
        gen_depth, s, f, prob = network(rgb.cuda(), point.cuda(), hole_point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy()
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'CFormer_KITTI' in network_type and 'DIODE' not in network_type:
        # CFormer，记得更改其他CFormer相关
        point = point * 95. #  KITTI
        # point = point * 25. #  nyu
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth['pred'].detach()
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        # gen_depth = (gen_depth / 25.)  # nyu
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'PEnet' in network_type:          
        # PENET
        if desc == 'NYU':
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
        point = point * 95. #  KITTI
        gen_depth = network(rgb.cuda(), point.cuda(), K.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'SemAttNet' in network_type:    
        # SemAttNet
        point = point * 95. #  KITTI
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
        gen_depth = (gen_depth / 95.)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'ACMNet' in network_type:               
        # ACMNet
        point = point * 95. * 255. #  KITTI
        if desc == 'NYU':
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
        gen_depth = (gen_depth / 95.)  # KITTI
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'TWISE' in network_type:            
        # TWISE
        from src.baselines.TWISE.utils import smooth2chandep
        gen_depth,_,_ = network(point.cuda(),rgb.cuda()) 
        gen_depth = smooth2chandep(gen_depth, params={'depth_maxrange': 1.0,}, device=None)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'GuideNet' in network_type:           
        # GuideNet
        point = point * 95.
        gen_depth, = network(rgb.cuda(), point.cuda())
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        gen_depth = np.clip(gen_depth, 0, 1)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'NLSPN_KITTI' in network_type and 'DIODE' not in network_type:            
        # NLSPN
        point = point * 95. #  KITTI
        # point = point * 25. #  NYU
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        # gen_depth = (gen_depth / 25.)  # NYU
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'SDCM' in network_type:        
        # SDCM
        point = point * 95. #  KITTI
        import torchvision.transforms as transforms
        grayscale_transform = transforms.Grayscale(num_output_channels=1)  # 创建一个灰度转换器
        rgb = grayscale_transform(rgb)  # 使用转换器将RGB图像转换为灰度图像
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'MDAnet' in network_type:            
        # MDAnet
        point = point * 95.  # 网络需要
        gen_depth = network(point.cuda(),rgb.cuda())
        gen_depth = gen_depth[0]  
        gen_depth = (gen_depth / 95.)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'EMDC' in network_type:            
        # EMDC
        point = point * 25.  # 网络需要
        gen_depth = network(rgb.cuda(),point.cuda())
        gen_depth = gen_depth[0]
        gen_depth = (gen_depth / 25.)
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
    elif 'LRRU' in network_type:            
        # LRRU
        point = point * 95. #  KITTI
        gen_depth = network(rgb.cuda(), point.cuda())  
        gen_depth = gen_depth['results'][-1]
        gen_depth = gen_depth.squeeze().to('cpu').numpy().astype(np.float32)
        gen_depth = (gen_depth / 95.)  # KITTI
        depth = np.clip(gen_depth * 255., 0, 255).astype(np.int8)
                      
    # # save img
    if not os.path.exists(str(Path(out_path).parent)):
        os.makedirs(str(Path(out_path).parent))
    depth_pil = Image.fromarray(depth.astype('uint8'))
    depth_pil.save(out_path)

def demo(args, network, pro, mode, network_type):
    base_path = '/data1/Chenbingyuan/Trans_G2/g2_dataset/'
    dataset_dict = {base_path+'nyu/val':'nyu', base_path+'DIODE/val':'DIODE', base_path+'HRWSI/val':'HRWSI',
                    base_path+'ETH3D/val':'ETH3D', base_path+'Ibims/val':'Ibims', base_path+'redweb/val':'redweb',
                    base_path+'KITTI/val':'KITTI', base_path+'VKITTI/val':'VKITTI', base_path+'Matterport3D/val':'Matterport3D',
                    base_path+'UnrealCV/val':'UnrealCV',}
    network.eval()
    with torch.no_grad():
        print(f'args.rgbd_dir={args.rgbd_dir}')
        jpg_list = ['HRWSI','DIODE','nyu','redweb']
        desc = dataset_dict[args.rgbd_dir]
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
                if method == 'rz_sb_mar_LRRU':
                    #  LRRU
                    from src.baselines.LRRU.model.model_dcnv2 import Model as LRRUModel
                    import argparse

                    arg = argparse.ArgumentParser(description='depth completion')
                    arg.add_argument('-p', '--project_name', type=str, default='inference')
                    arg.add_argument('-c', '--configuration', type=str, default='/data1/Chenbingyuan/Trans_G2/src/baselines/LRRU/configs/val_lrru_base_kitti.yml')
                    arg = arg.parse_args()
                    from src.baselines.LRRU.configs import get as get_cfg
                    args_LRRU = get_cfg(arg)
                    network = LRRUModel(args_LRRU)
                    network = network.cuda()

                for mode in mode_list:
                    # 0-100
                    for pro in pro_dict[mode]:
                        print(str(args.rgbd_dir) + method + '_'  + mode + '_' + str(pro))
                        demo(args, network, pro, mode, network_type=method)
                sys.path.clear()
                sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def eva():
    # 构建命令
    commands = """
    source /home/PanLingzhi/anaconda3/etc/profile.d/conda.sh
    conda activate completionformer
    python /data1/Chenbingyuan/Trans_G2/application/evaluate.py
    """

    # 启动一个 shell 进程，并捕获标准输出和标准错误
    process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 向 shell 进程发送命令并获取输出
    stdout, stderr = process.communicate(commands)

    # 打印输出和错误
    print(stdout)
    if stderr:
        print("Errors:\n", stderr)
rgbd_dir = ['HRWSI','DIODE', 'nyu', 'redweb','ETH3D','Ibims', 'KITTI', 'VKITTI', 'Matterport3D', 'UnrealCV']

dataset_list = copy.deepcopy(rgbd_dir)
for i,dir in enumerate(rgbd_dir):
    rgbd_dir[i] = '/data1/Chenbingyuan/Trans_G2/g2_dataset/'+dir+'/val'

mode_list = [ 'result']

# method_list = ['rz_sb_mar_NLSPN_DIODE_HRWSI_60' ,'rz_sb_mar_CFormer_KITTI', 'rz_sb_mar_EMDC', 'rz_sb_mar_g2_DIODE_HRWSI',  
#                'rz_sb_mar_NLSPN_KITTI', 'rz_sb_mar_SDCM', 'rz_sb_mar_sfv2_DIODE_HRWSI', 'rz_sb_mar_sfv2_DIODE_HRWSI_no_f',
#                'rz_sb_mar_sfv2_DIODE_HRWSI_no_p', 'rz_sb_mar_sfv2_DIODE_HRWSI_only_f', 'rz_sb_mar_sfv2_DIODE_HRWSI_only_p',
#                'rz_sb_mar_sfv2_DIODE_HRWSI_only_s', 'rz_sb_mar_TWISE']

method_list = ['rz_sb_mar_LRRU']

# 'rz_sb_mar_LRRU' 'rz_sb_mar_PEnet'  需要切换环境
# method_list = [ 'rz_sb_mar_TWISE'] conda activate completionformer

epoch_list = [60]
crop = False

# pro_dict = {'result': [0.001,0.01,0.1,0.2,0.5,0.7,1.01, 1.04, 1.016, 1.064]}
pro_dict = {'result': [1.08, 1.016, 1.032, 1.0128]}
if __name__ == "__main__":
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Started Time:", formatted_time)
    depth_inference()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Val-Ended Time:", formatted_time)
    print('Done!')
