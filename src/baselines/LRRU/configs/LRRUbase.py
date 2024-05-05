
import argparse

def get_args():
    args = argparse.ArgumentParser(description='NLSPN')
    
    # Hardware
    args.add_argument('--seed', type=int, default=1128)
    args.add_argument('--gpus', type=int, nargs='+', default=[0])
    args.add_argument('--port', type=int, default=29000)
    args.add_argument('--num_threads', type=int, default=1)
    args.add_argument('--no_multiprocessing', action='store_true')
    args.add_argument('--cudnn_deterministic', action='store_false')
    args.add_argument('--cudnn_benchmark', action='store_true')

    # Dataset
    args.add_argument('--data_folder', type=str, default='/home/temp_user/kitti_depth')
    args.add_argument('--dataset', type=str, nargs='+', default=['dep', 'gt', 'rgb'])
    args.add_argument('--val', type=str, default='select')
    args.add_argument('--grid_spot', action='store_true')
    args.add_argument('--num_sample', type=int, default=1000)
    args.add_argument('--cut_mask', action='store_false')
    args.add_argument('--max_depth', type=float, default=80.0)
    args.add_argument('--rgb_noise', type=float, default=0.0)
    args.add_argument('--noise', type=float, default=0.0)

    args.add_argument('--hflip', action='store_true')
    args.add_argument('--colorjitter', action='store_true')
    args.add_argument('--rotation', action='store_true')
    args.add_argument('--resize', action='store_false')
    args.add_argument('--normalize', action='store_true')
    args.add_argument('--scale_depth', action='store_false')

    args.add_argument('--val_h', type=int, default=352)
    args.add_argument('--val_w', type=int, default=1216)
    args.add_argument('--random_crop_height', type=int, default=256)
    args.add_argument('--random_crop_width', type=int, default=1216)
    args.add_argument('--train_bottom_crop', action='store_true')
    args.add_argument('--train_random_crop', action='store_true')
    args.add_argument('--val_bottom_crop', action='store_true')
    args.add_argument('--val_random_crop', action='store_true')
    args.add_argument('--test_bottom_crop', action='store_true')
    args.add_argument('--test_random_crop', action='store_true')

    # Network
    args.add_argument('--depth_norm', action='store_false')
    args.add_argument('--dkn_residual', action='store_true')
    args.add_argument('--summary_name', type=str, default='summary')

    # Test
    args.add_argument('--test', action='store_true')
    args.add_argument('--test_option', type=str, default='val')
    args.add_argument('--test_name', type=str, default='ben_depth')
    args.add_argument('--tta', action='store_false')
    args.add_argument('--test_not_random_crop', action='store_false')
    args.add_argument('--wandb_id_test', type=str, default='')

    args.add_argument('--prob', type=float, default=0.5)
    args.add_argument('--bc', type=int, default=16)
    args.add_argument('--model', type=str, default='model_dcnv2')
    args.add_argument('--test_dir', type=str, default='./pretrained/LRRU_Base')
    args.add_argument('--test_model', type=str, default='./pretrained/LRRU_Base/LRRU_Base.pt')

    # Summary
    args.add_argument('--num_summary', type=int, default=6)
    args.add_argument('--save_test_image', action='store_false')

    # Logs
    args.add_argument('--vis_step', type=int, default=1000)
    args.add_argument('--record_by_wandb_online', action='store_false')
    args.add_argument('--test_record_by_wandb_online', action='store_false')
    args.add_argument('--save_result_only', action='store_false')

    return args.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(args)
