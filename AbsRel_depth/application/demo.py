import sys
import os

# append module path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
from PIL import Image
from pathlib import Path
from src.networks import UNet
from src.utils import str2bool
from application.application_utils import RGBPReader
import torch
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# turn fast mode on
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        "options for AbsRel_depth  estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rgbd_dir",
        action="store",
        type=lambda x: Path(x),
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

    args = parser.parse_args()
    return args


def demo(args, network, pro, mode):
    network.eval()
    with torch.no_grad():
        for file in args.rgbd_dir.rglob('*.png'):
            str_file = str(file)
            if '/rgb/' in str_file:
                rgb_path = str_file
                if mode == 'result':
                    point_path = rgb_path.replace('/rgb/', '/point_' + str(pro) + '/', 1)
                else:
                    point_path = rgb_path.replace('/rgb/', '/point_nb_' + str(pro) + '/', 1)

                out_dir = os.path.join(mode, args.method + '_' + str(pro))
                out_path = rgb_path.replace('/rgb/', '/' + out_dir + '/', 1)
                rgbd_reader = RGBPReader()
                # processing
                rgb, point, hole_point = rgbd_reader.read_rgbp(rgb_path, point_path)
                gen_depth = network(rgb.cuda(), point.cuda(), hole_point.cuda())
                depth = rgbd_reader.adjust_domain(gen_depth)

                # # save img
                if not os.path.exists(str(Path(out_path).parent)):
                    os.makedirs(str(Path(out_path).parent))
                depth_pil = Image.fromarray(depth)
                depth_pil.save(out_path)


def depth_inference():
    mode_list = ['result', 'result_nb']
    # dataset_list = ['Sintel','DIODE', 'ETH3D', 'Ibims', 'NYUV2', 'KITTI']
    dataset_list = ['Ibims', 'NYUV2', 'KITTI']
    method_list = ['bn_mar']
    epoch_list = [60]
    pro_dict = {'result': [0.0, 0.1, 0.01, 0.001, 1.0], 'result_nb': [0.1, 0.01, 0.001, 1.0]}
    args = parse_arguments()
    for method in method_list:
        for epoch in epoch_list:
            epoch = str(epoch)
            args.method = method
            model_dir = ('../train_logs_' + method + '/models/epoch_' + epoch + '.pth')

            # load parameters
            if 'rz' not in method:
                args.ReZero = False

            network = UNet(rezero=args.ReZero)
            network = network.cuda()
            network.load_state_dict(torch.load(model_dir)['network_state_dict'])
            for dataset in dataset_list:
                args.rgbd_dir = Path('../Test_datasets/' + dataset)
                for mode in mode_list:
                    # 0-100
                    for pro in pro_dict[mode]:
                        demo(args, network, pro, mode)
                        print(method + '_' + dataset + '_' + mode + '_' + str(pro))


if __name__ == "__main__":
    depth_inference()
