import timm
import torch
model = timm.create_model('swinv2_base_window12to24_192to384', pretrained=True, features_only=True, window_size=12 ,img_size=(192, 384),pretrained_cfg_overlay=dict(file='/data/4TSSD/cby/dataset/g2_dataset/pretrained_models/pytorch_model.bin'))
a = torch.rand((1,3,192,384))
b = model(a)
for bb in b:
    print(bb.shape)

