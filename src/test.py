import sys
sys.path.extend(['/data1/Chenbingyuan/Depth-Completion/src'])
import torch
import torch.nn.functional as F
import torch.nn as nn


from torch.cuda.amp import autocast

# from baselines.midas.dpt_depth import DPTDepthModel
from baselines.CFormer.model.completionformer import CompletionFormer
from baselines.CFormer.model.config import args as args_cformer
model = CompletionFormer(args_cformer)
model = model.cuda()
rgb = torch.ones((4,3,320,320))
dep = torch.ones((4,1,320,320))
rgb = rgb.cuda()
dep = dep.cuda()
b = model(rgb, dep)
print(b)