import torch
from .base_model import BaseModel
from . import networks
import numpy as np

def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

class TESTModel(BaseModel):
    def name(self):
        return 'TESTModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        
        self.visual_names = ['sparse', 'pred', 'img']
        
        self.model_names = ['DC']
  
        self.netDC = networks.DCOMPNet(channels=opt.channels, knn=opt.knn, nsamples=opt.nsamples, scale=opt.scale)
        self.netDC = networks.init_net(self.netDC, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, need_init=False)
        self.model_dir = '/data/4TSSD/cby/Depth-Completion/src/ACMNet/model_64.pth'

        cam2cam = read_calib_file('/data/4TSSD/cby/dataset/KITTI/calib_cam_to_cam.txt')
        P2_rect = cam2cam['P_rect_02'].reshape(3,4)
        K = P2_rect[:, :3].astype(np.float32)
        self.K = K
    def set_input(self, input):

        self.img = input['img'].to(self.device)
        self.sparse = input['sparse'].to(self.device)
        self.K = input['K'].to(self.device)

    def forward(self, input_s, input_i, input_K):
        self.sparse = input_s
        self.img = input_i
        self.K = input_K
        sparse = self.sparse
        if self.opt.flip_input:
            # according to https://github.com/kakaxi314/GuideNet,
            # this operation might be helpful to reduce the error greatly.
            input_s = torch.cat([self.sparse, self.sparse.flip(3)], 0)
            input_i = torch.cat([self.img, self.img.flip(3)], 0)
            input_K = torch.cat([self.K, self.K], 0)
        else:
            input_s = self.sparse
            input_i = self.img
            input_K = self.K
        
        with torch.no_grad():
            out = self.netDC(input_s, input_i, input_K)
            self.pred = out[0]

        return self.pred