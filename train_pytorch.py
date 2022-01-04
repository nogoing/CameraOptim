import os
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch3d

from tensorboardX import SummaryWriter

from co3d.dataset.dataset_zoo import dataset_zoo
from co3d.dataset.dataloader_zoo import dataloader_zoo

from network.nerformer import NerFormerArchitecture
from network.feature_network import FeatureNetArchitecture

from positional_embedding import HarmonicEmbedding
from ray_sampling import RaySampler
from rendering import render_rays
from utils import *

from omegaconf import OmegaConf




def mse_loss(preds, labels):
    return F.mse_loss(preds, labels, reduction="sum")


def mask_mse_loss(preds, labels, mask):
    return F.mse_loss(preds[mask!=0], labels[mask!=0], reduction="sum")


def bce_loss(preds, labels):
    preds = torch.clamp(preds, min=0.0001, max=0.9999)
    
    return F.binary_cross_entropy(preds, labels, reduction="sum")



def train(args):
    # CO3D Dataset
    datasets = dataset_zoo(
            category=args.co3d_category,
            assert_single_seq=(args.co3d_task == "singlesequence"),
            dataset_name=f"co3d_{args.co3d_task}",
            test_on_train=False,
            load_point_clouds=False,
            test_restrict_sequence_id=args.co3d_single_sequence_id,
        )

    # CO3D Dataset loader
    dataloaders = dataloader_zoo(
            datasets,
            dataset_name=f"co3d_{args.co3d_task}",
            batch_size=(args.N_src + args.N_src_extra),
            # num_workers=1,
            dataset_len=1000,
            dataset_len_val=10,
            images_per_seq_options=[100],
        )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    args.train_len = len(train_loader)
    args.val_len = len(val_loader)

    # save directory
    out_folder = os.path.join(args.root_dir, 'out', args.exp_name)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)
    
    # save current config
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        OmegaConf.save(config=args, f=file.name)

    # feature dim
    resnet_feature_dim = 32 * 3
    rgb_dim = 3
    segemtation_dim = 1
    pe_dim = args.pe_dim

    d_z = resnet_feature_dim + rgb_dim + segemtation_dim + pe_dim*2*3
    print(f"D_z: {d_z}")

    # Nerforemr Network
    coarse_net = NerFormerArchitecture(d_z=d_z).to(args.device)
    # fine_net = NerFormerArchitecture(d_z=d_z).to(args.device)

    # FeatureNet
    feature_net = FeatureNetArchitecture().to(args.device)

    # Positional Embedding
    PE = HarmonicEmbedding(n_harmonic_functions=pe_dim).to(args.device)

    # tensorboard writer
    tb_dir = os.path.join(args.root_dir, 'logs/', args.exp_name)
    writer = SummaryWriter(tb_dir)
    print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}
    
    # optimizer
    # optimizer = torch.optim.Adam([coarse_net.parameters(), fine_net.parameters()], lr=args.lr)
    optimizer = torch.optim.Adam([
                {'params': coarse_net.parameters()},
                # {'params': fine_net.parameters()},
                ], 
                lr=args.lr)

    n_iters = args.n_iters
    global_step = 0
    epoch = 0

    # train
    while global_step < n_iters + 1:
        for train_data in train_loader:
            time0 = time.time()
            
            ##################### target, source data frame 구성 #####################
            target, srcs = data_to_frame(train_data, args.N_src)
            for k in target:
                if k != "rgb_path":
                    target[k] = target[k].to(args.device)
            for k in srcs:
                if k != "rgb_path":
                    srcs[k] = srcs[k].to(args.device)

            ######################## source feature map 구성 ########################
            with torch.no_grad():
                feature_maps = feature_net(srcs["rgb"], srcs["mask"])

            # ray sampler 생성
            ray_sampler = RaySampler(target, srcs["camera"])
            # 타겟 이미지에서 N_rays개의 ray 샘플링
            ray_batch = ray_sampler.random_sample(args.N_rays, args.ray_sampling_mode, args.center_ratio)

            output = render_rays(ray_batch, coarse_net, coarse_net, feature_maps, PE, args)

            optimizer.zero_grad()

            # coarse loss
            coarse_rgb_loss = mask_mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
            coarse_mask_loss = bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
            coarse_loss = coarse_rgb_loss + coarse_mask_loss
            # fine loss
            fine_rgb_loss = mask_mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
            fine_mask_loss = bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
            fine_loss = fine_rgb_loss + fine_mask_loss

            total_loss = coarse_loss + fine_loss
            total_loss.backward()

            scalars_to_log['train/total_loss'] = total_loss.item()
            
            optimizer.step()

            dt = time.time() - time0

            # loss 출력
            if global_step % args.log_loss_step == 0 or global_step < 10:
                scalars_to_log['train/coarse-rgb-loss'] = coarse_rgb_loss
                scalars_to_log['train/coarse-mask-loss'] = coarse_mask_loss

                if output['outputs_fine'] is not None:
                    scalars_to_log['train/fine-rgb-loss'] = fine_rgb_loss
                    scalars_to_log['train/fine_mask_loss'] = fine_mask_loss

                logstr = '{} Epoch: {}  step: {} '.format(args.exp_name, epoch, global_step)
                for k in scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar(k, scalars_to_log[k], global_step)
                print(logstr)

                print('each iter time {:.05f} seconds'.format(dt))

            # checkpoint 저장
            if global_step % args.log_weight_step == 0:
                print(f"Step[{global_step+1}/{n_iters}]: Checkpoint 저장...")
                save_path = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))

                to_save = {
                            'optimizer': optimizer.state_dict(),
                            'coarse_net': coarse_net.state_dict(),
                            # 'fine_net': fine_net.state_dict(),
                            }
                torch.save(to_save, save_path)

            # 시각화 저장
            if args.log_img and global_step % args.log_img_step == 0:
                ######################## current training data ########################
                print(f"Step[{global_step+1}/{n_iters}]: Training data 결과 저장...")

                # train ray sampler 생성
                train_ray_sampler = RaySampler(target, srcs["camera"])
                H, W = train_ray_sampler.H, train_ray_sampler.W
                train_gt_img = train_ray_sampler.rgb.reshape(H, W, 3)
                train_gt_mask = train_ray_sampler.mask.reshape(H, W, 1)
                gts = {
                        "img":train_gt_img,
                        "mask":train_gt_mask,
                    }
                # log
                log_view_to_tensorboard(args, writer, global_step, coarse_net, coarse_net, train_ray_sampler, feature_maps,
                                    srcs, PE, gts, prefix='train/')

                ######################## random validation data ########################
                print(f"Step[{global_step+1}/{n_iters}]: Validation data 결과 저장...")

                val_data = next(iter(val_loader))
                #  validation target, source data frame 구성 
                val_target, val_srcs = data_to_frame(val_data, args.N_src)
                for k in val_target:
                    if k != "rgb_path":
                        val_target[k] = val_target[k].to(args.device)
                for k in val_srcs:
                    if k != "rgb_path":
                        val_srcs[k] = val_srcs[k].to(args.device)

                with torch.no_grad():
                    val_feature_maps = feature_net(val_srcs["rgb"], val_srcs["mask"])

                # validation ray sampler 생성
                val_ray_sampler = RaySampler(val_target, val_srcs["camera"])
                H, W = val_ray_sampler.H, val_ray_sampler.W
                val_gt_img = val_ray_sampler.rgb.reshape(H, W, 3)
                val_gt_mask = val_ray_sampler.mask.reshape(H, W, 1)
                gts = {
                        "img":val_gt_img,
                        "mask":val_gt_mask,
                    }
                # log
                log_view_to_tensorboard(args, writer, global_step, coarse_net, coarse_net, val_ray_sampler, val_feature_maps,
                                    val_srcs, PE, gts, prefix='val/')

            global_step += 1

            if global_step > n_iters + 1:
                break
            
        epoch += 1


if __name__ == '__main__':
    args = OmegaConf.load("config_file.yaml")
    print(args, "\n\n")

    train(args)