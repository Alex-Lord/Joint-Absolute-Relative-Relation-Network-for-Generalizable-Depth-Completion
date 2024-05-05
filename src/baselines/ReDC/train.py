def iterate(mode, args, loader, model, optimizer, scheduler, logger, epoch, loss_curve=[], writer=None):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        scheduler.step()
        print("Learning Rate:", scheduler.get_last_lr())
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()
    for i, batch_data in enumerate(loader):
        dstart = time.time()
        del batch_data["path"]
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        start = time.time()
        pred = model((batch_data))
        
        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        if mode == 'train':
            depth_loss = depth_criterion(pred, gt)
            loss = depth_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % args.print_freq == 0:
                print("loss:", loss.data, " epoch:", epoch, " ", i, "/", len(loader))
            if writer is not None:
                writer.add_scalar('Loss/train', loss.data, epoch*len(loader)+i)
            loss_curve.append(loss.data)
        if mode == "test_completion":
            str_i = str(i)
            path_i = str_i.zfill(10) + '.png'
            path = os.path.join(args.data_folder_save, path_i)
            vis_utils.save_depth_as_uint16png_upload(pred, path)

        if(not args.evaluate):
            gpu_time = time.time() - start
        if mode == "val":
            with torch.no_grad():
                mini_batch_size = next(iter(batch_data.values())).size(0)
                result = Result()
                if mode != 'test_prediction' and mode != 'test_completion':
                    result.evaluate(pred.data, gt.data, photometric_loss)
                    [
                        m.update(result, gpu_time, data_time, mini_batch_size)
                        for m in meters
                    ]

                    if mode != 'train':
                        logger.conditional_print(mode, i, epoch, lr, len(loader),
                                         block_average_meter, average_meter)
                    logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                       epoch)
                    logger.conditional_save_pred(mode, i, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best, loss_curve

# ************************************************************************************************************

from PIL import ImageFile
import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader_original import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils
import sys
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from backbone import PENet_C2
from redc import ReDC
# ************************************************************************************************************

print(args)
kitti_data_folder = '/data/xs15/Kitti'

args.data_folder = kitti_data_folder
# define loss functions
if (args.criterion == 'l2'):
    depth_criterion = criteria.MaskedMSELoss()
elif (args.criterion == 'l1'):
    depth_criterion = criteria.MaskedL1Loss()
elif (args.criterion == 'both'):
    print("Using a mixed l2 & l1 loss")
    depth_criterion = criteria.MaskedBothLoss()
else:
    print("Unrecognized Type")
    exit()
    
checkpoint = None
is_eval = False

# ************************************************************************************************************
val_dataset = KittiDepth('val', args)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True)  # set batch size to be 1 for validation
print("\t==> val_loader size:{}".format(len(val_loader)))
train_dataset = KittiDepth('train', args)
# train_dataset = KittiDepth('train_val', args)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers,
                                           pin_memory=True)
print("\t==> train_loader size:{}".format(len(train_loader)))
# ************************************************************************************************************

logger = helper.logger(args)
writer = SummaryWriter()
if checkpoint is not None:
    logger.best_result = checkpoint['best_result']
    del checkpoint
print("=> logger created.")

# As mentioned in the paper, we study our deformanle refinement module on top of ENet.
# Here, we load the pretrained ENet model for faster convergence and train our deformable refinement module on top of it from scratch.
# You can also train the whole network from scratch
# Check redc.py on the implementation of our architecture
args.network_model = 'pe'
orig_model = PENet_C2(args)
model = ReDC(args)
pt_path = "/shared/rsaas/common/Kitti/pe.pth.tar"
ckpt = torch.load(pt_path)
orig_model.load_state_dict(ckpt['model'], strict=False)
orig_model.eval()
param_list = dict(model.named_parameters())
orig_dict = dict(orig_model.named_parameters())
for i, (name, module) in enumerate(model.named_parameters()):
    if name in orig_dict:
        param_list[name].data.copy_(orig_dict[name].data)


if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      model = torch.nn.DataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
# ************************************************************************************************************
total_loss = []

start_epoch = args.start_epoch

for epoch in range(start_epoch, args.epochs):
    print("=> starting training epoch {} ..".format(epoch))
    avg, is_best, loss_curve = iterate("train", args, train_loader, model, optimizer, scheduler, logger, epoch, writer=writer)  # train for one epoch
    total_loss += loss_curve
    result, is_best, loss_curve = iterate("val", args, val_loader, model, None, None, logger, epoch)  # evaluate on validation set
    try:
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            "scheduler": scheduler.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)
    except:
        continue
