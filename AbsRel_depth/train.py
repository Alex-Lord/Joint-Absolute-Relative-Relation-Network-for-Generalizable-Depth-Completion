import argparse
from pathlib import Path
from src.src_main import AbsRel_depth
from src.networks import UNet
from src.utils import str2bool, DDPutils
import os
import torch
from torch.backends import cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'

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
        default=r'/data1/Chenbingyuan/Trans_G2/AbsRel_depth/RGBD_datasets/',
        help="Path to RGB-depth folder",
        required=False
    )
    parser.add_argument(
        "--hole_dir",
        action="store",
        type=lambda x: Path(x),
        default=r'Hole_datasets',
        help="Path to Hole folder",
        required=False
    )
    parser.add_argument(
        "--save_dir",
        action="store",
        type=str,
        default=r'train_logs',
        help="Path to the directory for saving the logs and models",
        required=False
    )
    parser.add_argument(
        "--rezero",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the ReZero",
        required=False,
    )
    parser.add_argument(
        "--sobel",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the sobel operator",
        required=False,
    )
    parser.add_argument(
        "--msgrad",
        action="store",
        type=str2bool,
        default=True,
        help="whether to use the multi-scale gradient",
        required=False,
    )
    parser.add_argument(
        "--mode",
        action="store",
        type=str,
        default='mar',
        help="standardization methods: ['mean_robust(mar)','mean(ma)','median(md)']",
        required=False,
    )
    parser.add_argument(
        "--epochs",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=60,
        help="epochs numbers",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        type=int,
        required=False,
        nargs="+",
        default=16,
        help="batch sizes",
    )
    parser.add_argument(
        "--resume_train",
        action="store",
        type=str2bool,
        default=False,
        help="resume train or not",
        required=False,
    )
    parser.add_argument(
        "--model_dir",
        action="store",
        type=lambda x: Path(x),
        default=r'/data1/Chenbingyuan/Trans_G2/AbsRel_depth/train_logs_rz_sb_mar/models/epoch_16.pth',
        help="Path to load models",
        required=False
    )
    parser.add_argument(
        "--port",
        action="store",
        type=int,
        default=6005,
        help="DDP port",
    )

    args = parser.parse_args()
    return args


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def DDP_main(rank, world_size):
    args = parse_arguments()
    if args.rezero:
        args.save_dir += '_rz'
    else:
        args.save_dir += '_bn'
    if args.msgrad:
        if args.sobel:
            args.save_dir += '_sb'
        else:
            args.save_dir += '_gd'
    args.save_dir += '_' + args.mode
    args.save_dir += '_' + args.mode + '_2'
    args.save_dir = Path(args.save_dir)
    args.model_dir = args.save_dir / 'models' / 'epoch_18.pth'

    # DDP components
    DDPutils.setup(rank, world_size, args.port)

    if rank == 0:
        print(f"Selected arguments: {args}")

    network = UNet(rezero=args.rezero)
    print_model_parm_nums(network)

    semigan = AbsRel_depth(
        network,
        rank,
    )

    # resume train
    if args.resume_train:
        if rank == 0:
            print('resume training...')
            # load everything
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.model_dir, map_location=map_location)
    else:
        checkpoint = None

    semigan.train(
        args=args,
        rank=rank,
        learning_rate=0.0002,
        feedback_factor=1000,
        checkpoint_factor=2,
        num_workers=2,
        checkpoint=checkpoint,
    )

    DDPutils.cleanup()


def Non_DDP_main(rank=0, world_size=1):
    args = parse_arguments()
    if args.rezero:
        args.save_dir += '_rz'
    else:
        args.save_dir += '_bn'
    if args.msgrad:
        if args.sobel:
            args.save_dir += '_sb'
        else:
            args.save_dir += '_gd'
    args.save_dir += '_' + args.mode
    args.save_dir = Path(args.save_dir)
    args.model_dir = args.save_dir / 'models' / 'epoch_16.pth'

    if rank == 0:
        print(f"Selected arguments: {args}")

    network = UNet(rezero=args.rezero)
    print_model_parm_nums(network)

    semigan = AbsRel_depth(
        network,
        rank,
    )

    # resume train
    if args.resume_train:
        if rank == 0:
            print('resume training...')
            # load everything
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.model_dir, map_location=map_location)
    else:
        checkpoint = None

    semigan.train(
        args=args,
        rank=rank,
        learning_rate=0.0002,
        feedback_factor=1000,
        checkpoint_factor=2,
        num_workers=3,
        checkpoint=checkpoint,
    )



if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        DDPutils.run_demo(DDP_main, n_gpus)
        # Non_DDP_main()
