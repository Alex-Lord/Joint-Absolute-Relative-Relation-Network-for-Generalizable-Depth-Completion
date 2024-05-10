import datetime
import time
import timeit
from pathlib import Path
import argparse

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Tuple

from .utils import save_img, DataGenerator, min_max_norm, StandardizeData, on_load_checkpoint
from .data_tools import get_dataloader
from .losses import (
    WeightedDataLoss,
    WeightedMSGradLoss,
)
from .networks import UNet
from tqdm import tqdm

class AbsRel_depth:
    def __init__(
            self,
            network: UNet,
            rank: torch.device,
    ) -> None:
        self.network = network.to(rank)

        # if rank == 0:
        #     print(f"Network: {self.network}\n")

    def optimize_net(
            self,
            rgb: Tensor,
            gt: Tensor,
            point_map: Tuple,
            hole_tuple: Tuple,
            optimizer: Optimizer,
            scaler: GradScaler,
            args: argparse,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        optimizer.zero_grad()
        with autocast():
            # loss in absolute domain
            reg_function = WeightedDataLoss()
            depth = self.network(rgb, point_map, hole_tuple[1])
            loss_adepth = reg_function(depth, gt, hole_tuple[1])

            # loss in relative domain
            sta_tool = StandardizeData(mode=args.mode)
            sta_depth, sta_gt = sta_tool(depth, gt, hole_tuple[0])
            loss_rdepth = reg_function(sta_depth, sta_gt, hole_tuple[0])

            if args.msgrad:
                grad_function = WeightedMSGradLoss(sobel=args.sobel)
                loss_rgrad = grad_function(sta_depth, sta_gt, hole_tuple[0])
                loss = loss_adepth + loss_rdepth + 0.5 * loss_rgrad
            else:
                loss = loss_adepth + loss_rdepth
                loss_rgrad = torch.tensor(0.)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss_gen.backward()
        # optimizer.step()

        return loss_adepth.item(), loss_rdepth.item(), loss_rgrad.item(), depth

    def feedback_module(
            self,
            elapsed: str,
            i: int,
            iteration_num: int,
            loss_adepth: Tensor,
            loss_rdepth: Tensor,
            loss_rgrad: Tensor,
            summary: SummaryWriter,
            global_step: int,
    ) -> None:
        print(
            'Elapsed:[%s]|batch:[%d/%d]|abs:%.4f|rel:%.4f|grad:%.4f'
            % (
                elapsed, i, iteration_num, float(loss_adepth), float(loss_rdepth), float(loss_rgrad)
            )
        )

        # log loss
        summary.add_scalar(
            "loss/loss_abs", loss_adepth, global_step=global_step
        )
        summary.add_scalar(
            "loss/loss_rel", loss_rdepth, global_step=global_step
        )
        summary.add_scalar(
            "loss/loss_grad", loss_rgrad, global_step=global_step
        )


    @staticmethod
    def save_imgs(
            rgb: Tensor,
            point_map: Tensor,
            depth: Tensor,
            gt: Tensor,
            log_dir: Path,
            epoch: int,
            iter_step: int,
    ) -> None:
        # make data_path
        epoch_dir = log_dir / ('epoch_' + str(epoch))
        epoch_dir.mkdir(exist_ok=True)
        rgb_file = epoch_dir / f"rgb_{epoch}_{iter_step}.png"
        point_file = epoch_dir / f"point_{epoch}_{iter_step}.png"
        depth_file = epoch_dir / f"depth_{epoch}_{iter_step}.png"
        gt_file = epoch_dir / f"gt_{epoch}_{iter_step}.png"
        norm_depth_file = epoch_dir / f"norm_depth_{epoch}_{iter_step}.png"
        norm_gt_file = epoch_dir / f"norm_gt_{epoch}_{iter_step}.png"

        # save the images:
        save_img(rgb[0], rgb_file)
        save_img(point_map[0], point_file)
        save_img(depth[0], depth_file)
        save_img(gt[0], gt_file)
        save_img(min_max_norm(depth[0]), norm_depth_file)
        save_img(min_max_norm(gt[0]), norm_gt_file)

    def train(
            self,
            args: argparse,
            rank: torch.device,
            learning_rate: float = 0.0001,
            feedback_factor: int = 10,
            checkpoint_factor: int = 5,
            num_workers: int = 0,
            checkpoint: Dict = None,
    ):
        # train mode
        self.network.train()
        # create optimizer
        optimizer = torch.optim.Adam(
            [{
                'params': self.network.parameters(),
                'initial_lr': learning_rate,
            }],
            lr=learning_rate,
        )
        # use amp
        scaler = GradScaler()

        start_epoch = 0
        # resume train
        
        if checkpoint is not None:
            checkpoint = on_load_checkpoint(checkpoint)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # release memory of checkpoint
            checkpoint.clear()
            del checkpoint

        # learning rate scheduler
        scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80, 90], gamma=0.5, last_epoch=(start_epoch - 1))
        
        self.network = torch.compile(self.network)
        
        # Use DistributedDataParallel:
        self.network = DDP(self.network, device_ids=[rank], static_graph=True)

        # verbose stuff
        if rank == 0:
            print("Setting up the image saving mechanism")
            model_dir, log_dir = args.save_dir / "models", args.save_dir / "logs"
            model_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)

            # tensorboard summarywriter
            summary = SummaryWriter(str(log_dir / "tensorboard"))
            # create a global time counter
            global_time = time.time()
            print("Starting the training process ... ")

        # get data loaders and samplers
        rgbgph_data, hole_data, rgbgph_sampler, hole_sampler = get_dataloader(
            args.rgbd_dir, args.hole_dir, args.batch_size, rank, num_workers)

        iteration_num = len(rgbgph_data)
        global_step = start_epoch * iteration_num

        # warm-up parameters
        warm_up_cnt = 0.0
        warm_up_max_cnt = iteration_num + 1.0

        # start epoch
        for epoch in range(start_epoch + 1, args.epochs + 1):
            # set epoch in Distributed sampler
            rgbgph_sampler.set_epoch(epoch)
            hole_sampler.set_epoch(epoch + args.epochs)

            if rank == 0:
                start = timeit.default_timer()  # record time at the start of epoch
                print(f"\nEpoch: [{epoch}/{args.epochs}]")
            iterator = tqdm(enumerate(zip(rgbgph_data, hole_data), start=1), desc='Batch', total=iteration_num) if rank == 0 else enumerate(zip(rgbgph_data, hole_data), start=1)
            
            for (i, all_data) in iterator:

                # warm up in the 1st epoch
                if epoch == 1:
                    warm_up_cnt += 1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['initial_lr'] * warm_up_cnt / warm_up_max_cnt

                # preprocessing
                train_data = DataGenerator(args.batch_size)
                rgb, gt, point_map, hole_tuple = train_data.generate_data(all_data)

                # optimizing
                loss_adepth, loss_rdepth, loss_rgrad, depth = self.optimize_net(
                    rgb, gt, point_map, hole_tuple, optimizer, scaler, args)

                global_step += 1
                if rank == 0:
                    with torch.no_grad():
                        # log a loss feedback
                        if (i % feedback_factor == 0) or (i == 1):
                            elapsed = time.time() - global_time
                            elapsed = str(datetime.timedelta(seconds=elapsed))

                            # log and print
                            self.feedback_module(
                                elapsed, i, iteration_num, loss_adepth, loss_rdepth, loss_rgrad, summary, global_step
                            )
                            # save intermediate results
                            self.save_imgs(rgb, point_map, depth, gt, log_dir, epoch, i)

            # learning rate decay
            scheduler.step()

            if rank == 0:
                stop = timeit.default_timer()
                print("Time taken for epoch: %.3f secs" % (stop - start))

                # checkpoint
                if (
                        epoch % checkpoint_factor == 0
                        or epoch == 1
                        or epoch == args.epochs
                ):
                    save_file = model_dir / f"epoch_{epoch}.pth"

                    torch.save({
                        'epoch': epoch,
                        'network_state_dict': self.network.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, save_file)

        if rank == 0:
            print(f"Training completed ... Model saved in {args.save_dir}")
