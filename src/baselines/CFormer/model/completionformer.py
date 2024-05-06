"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

from .nlspn_module import NLSPN
from .backbone import Backbone
import torch
import torch.nn as nn
import os
from torch.cuda.amp import autocast

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args

class CompletionFormer(nn.Module):
    def __init__(self, args):
        super(CompletionFormer, self).__init__()

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        self.backbone = Backbone(args, mode='rgbd')
        with autocast(enabled=False):
            if self.prop_time > 0:
                self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                        self.args.prop_kernel)

    def forward(self, rgb, depth):
        # rgb = sample['rgb']
        # dep = sample['dep']
        rgb = rgb
        dep = depth
        pred_init, guide, confidence = self.backbone(rgb, dep)
        pred_init = pred_init + dep

        # Diffusion
        y_inter = [pred_init, ]
        conf_inter = [confidence, ]
        if self.prop_time > 0:
            with autocast(enabled=False):
                y, y_inter, offset, aff, aff_const = \
                    self.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            y = pred_init
            offset, aff, aff_const = torch.zeros_like(y), torch.zeros_like(y), torch.zeros_like(y).mean()

        # Remove negative depth
        y = torch.clamp(y, min=0)
        # best at first
        y_inter.reverse()
        conf_inter.reverse()
        if not self.args.conf_prop:
            conf_inter = None

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': conf_inter}

        return output
