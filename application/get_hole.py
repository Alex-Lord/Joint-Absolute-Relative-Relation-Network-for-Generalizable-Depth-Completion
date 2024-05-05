import os
import traceback
import application_utils as ap

from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    holedataset = ap.RGBPReader()
    rgb_path_list = ['/data/4TSSD/cby/dataset/g2_dataset/DIODE/train', 
                     '/data/4TSSD/cby/dataset/g2_dataset/HRWSI/train', '/data/4TSSD/cby/dataset/g2_dataset/vkitti/vkitti_rgb',
                     '/data/4TSSD/cby/dataset/g2_dataset/DIODE/val', '/data/4TSSD/cby/dataset/g2_dataset/HRWSI/val']
    # rgb_path_list = ['/data/4TSSD/cby/dataset/g2_dataset/kitti/kitti_rgb', '/data/4TSSD/cby/dataset/NYUv2/NYUv2_img', '/data/4TSSD/cby/dataset/g2_dataset/HRWSI/train',
    #                  '/data/4TSSD/cby/dataset/g2_dataset/DIODE/train', '/data/4TSSD/cby/dataset/g2_dataset/vkitti/vkitti_rgb']
    COUNTER = 0
    
    # png_files

    try:
        for rgb_path in rgb_path_list:
            print(f'Now dealing with {rgb_path}')
            rgb_path = Path(rgb_path)
            num_files = 0
            for file in rgb_path.rglob('*.png'): num_files += 1
            for file in tqdm(rgb_path.rglob('*.png'), desc='png_files', total=num_files):
                str_file = str(file)
                if 'kitti' in str_file and 'image_02' in str_file and 'vkitti' not in str_file:
                    point_path = str_file.replace(
                        'image_02/data', 'proj_depth/groundtruth/image_02')
                    point_path = point_path.replace('kitti_rgb', 'kitti_gt')
                    point_path = point_path[:49] + point_path[60:]
                    if os.path.exists(point_path):
                        save_dir = str_file.replace('kitti/kitti_rgb','hole_dataset/kitti_hole')
                        if not os.path.exists(save_dir): 
                            holedataset.get_hole(str_file, point_path,save_dir)
                            COUNTER += 1 
                elif 'DIODE' in str_file and 'dis' not in str_file:
                    point_path = str_file.replace('.png', '_depth.npy')
                    save_dir = str_file.replace('train', 'hole')
                    save_dir = save_dir.replace('DIODE',
                                                'hole_dataset/DIODE_hole')
                    if not os.path.exists(save_dir):
                        holedataset.get_hole(str_file, point_path, save_dir)
                        COUNTER += 1 
            
            # jpg files
            num_files = 0
            for file in rgb_path.rglob('*.jpg'): num_files += 1
            for file in tqdm(rgb_path.rglob('*.jpg'), desc='png_files', total=num_files):
                str_file = str(file)
                if 'vkitti' in str_file and '15-deg-left' in str_file and 'Camera_0' in str_file:
                    point_path = str_file.replace('rgb', 'depth')
                    point_path = point_path.replace('jpg', 'png')
                    point_path = point_path.replace(r'/rgb/', r'/depth/')
                    save_dir = str_file.replace(r'vkitti/vkitti_rgb',
                                                r'hole_dataset/vkitti_hole')
                    sava_dir = save_dir.replace('jpg', 'png')
                    if not os.path.exists(save_dir):
                        holedataset.get_hole(str_file, point_path, save_dir)
                        COUNTER += 1 
                elif 'NYUv2_rgb' in str_file:
                    point_path = str_file.replace('rgb', 'gt')
                    point_path = point_path.replace('jpg', 'png')
                    save_dir = point_path.replace(r'nyu/NYUv2_gt',
                                                  r'hole_dataset/nyu_hole')
                    if not os.path.exists(save_dir):
                        holedataset.get_hole(str_file, point_path, save_dir)
                        COUNTER += 1 
                elif 'HRWSI' in str_file and 'imgs' in str_file:
                    point_path = str_file.replace('imgs', 'gts')
                    point_path = point_path.replace('jpg', 'png')
                    save_dir = str_file.replace('HRWSI',
                                                'hole_dataset/HRWSI_hole')
                    save_dir = save_dir.replace('jpg', 'png')
                    if not os.path.exists(save_dir):
                        holedataset.get_hole(str_file, point_path, save_dir)
                        COUNTER += 1 
            print(f'Done Dealing {rgb_path}')
        print('Done!')
    except Exception:
        traceback.print_exc()
