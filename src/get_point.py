#%%
from data_tools import *
from multiprocessing import Pool

# pro_dict = {'result': [0.0, 0.1, 0.01, 0.001, 1.0],
#             'result_nb': [0.1, 0.01, 0.001, 1.0]}
# pro_dict = {'result': [0.001,0.01,0.1,0.2,0.5,0.7,1.0]}
# pro_dict = {'result': [1.0]}
# pro_dict = {'result': [0, 283/102400, 1129/102400, 3708/102400, 14691/102400]}
# pro_dict = {'lines_result': [0,1,4,16,64]}
# pro_dict = {'result_very_sparse':[0,1,2,3,10,20,50,100]}
# pro_dict = {'same_seg_':[0,1,2,3,10,20,50,100], 'differ_seg_':[0,1,2,3,10,20,50,100],}

# pro_dict = {'differ_seg_':[0,1,2,3,10,20,50,100]}
# rgbd_dir_list=['nyu', 'redweb','ETH3D','Ibims', 'KITTI', 'VKITTI', 'Matterport3D', 'UnrealCV']
rgbd_dir_list=['KITTI_validation']
for i,dir in enumerate(rgbd_dir_list):
    rgbd_dir_list[i] = '/data1/Chenbingyuan/Depth-Completion/g2_dataset/'+dir+'/val'
# for file in rgbd_dir.rglob('*'):
#     for file_2 in file.rglob('*'):
#         if 'dis' in str(file_2) or 'npy' in str(file_2) or 'crop' in str(file_2): os.remove(file_2)
get_point_cby(pro_dict, rgbd_dir_list, crop=True)
# get_very_sparse_point_cby(pro_dict,rgbd_dir_list)
# get_lines_cby(pro_dict, rgbd_dir, [0,1, 4, 16, 64])
# get_seg_point_cby(pro_dict,rgbd_dir)
print('Done!')
