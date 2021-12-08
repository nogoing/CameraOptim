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

from network.model import NerFormer
from network.feature_network import FeatureNet

from positional_embedding import HarmonicEmbedding
from ray_sampling import RaySampler
from rendering import render_rays, render_image

import config

def data_to_frame(co3d_data, device, tgt_idx, src_idxs):
    imgs = co3d_data.image_rgb.to(device)
    img_paths = co3d_data.image_path
    masks = co3d_data.fg_probability.to(device)
    camera = co3d_data.camera.to(device)

    target = {}
    target["rgb"] = imgs[tgt_idx]
    target["rgb_path"] = img_paths[tgt_idx]
    target["mask"] = masks[tgt_idx]
    target["camera"] = camera
    # 현재 타겟 카메라의 위치를 기준으로
    # 샘플링할 ray들의 depth range 계산
    target_camera_position = camera.get_camera_center()[tgt_idx]
    near = torch.norm(target_camera_position) - 8
    far = torch.norm(target_camera_position) + 8
    depth_range =  torch.tensor([near, far], dtype=torch.float32)
    target["depth_range"] = depth_range

    srcs = {}
    srcs["rgb"] = imgs[src_idxs]
    srcs["mask"] = masks[src_idxs]

    return target, srcs


def log_view_to_tensorboard(args, writer, global_step, model, ray_sampler, feature_maps, srcs, src_idxs, gt_img, gt_mask, prefix=''):
    model.eval()

    with torch.no_grad():
        # 이미지의 모든 픽셀에 대해 정의된 ray batch 생성.
        ray_batch = ray_sampler.get_all()

        output = render_image(ray_sampler=ray_sampler,
                            ray_batch=ray_batch,
                            src_idxs=src_idxs,
                            model=model,
                            feature_maps=feature_maps,
                            args=args,
                            )

    srcs = srcs["rgb"].detach().cpu()
    N_src = srcs.shape[0]

    if args.render_stride != 1:
        gt_img = gt_img[::args.render_stride, ::args.render_stride]
        srcs = srcs[:, :, ::args.render_stride, ::args.render_stride]

    img_HWC2CHW = lambda x: x.permute(2, 0, 1)
    
    rgb_gt = img_HWC2CHW(gt_img)
    rgb_pred = img_HWC2CHW(output['outputs_coarse']['rgb'].detach().cpu())

    h = rgb_gt.shape[-2]
    w = rgb_gt.shape[-1]

    srcs_im = torch.zeros(3, h, N_src*w)
    for i in range(0, N_src):
        srcs_im[:, :h, i*w:(i+1)*w] = srcs[i]

    rgb_im = torch.zeros(3, h, 3*w)
    rgb_im[:, :rgb_gt.shape[-2], :rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], w:w+rgb_pred.shape[-1]] = rgb_pred

    # depth_im = output['outputs_coarse']['depth'].detach().cpu()
    # acc_map = torch.sum(output['outputs_coarse']['weights'], dim=-1).detach().cpu()
    mask_gt = img_HWC2CHW(gt_mask).detach().cpu()
    mask_pred = img_HWC2CHW(output["outputs_coarse"]["mask"].detach().cpu().unsqueeze(-1))

    if output['outputs_fine'] is None:
        # depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        # depth_im = img_HWC2CHW(depth_im)
        # acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        mask_im = torch.cat((mask_gt, mask_pred), dim=-1)
    else:
        rgb_fine = img_HWC2CHW(output['outputs_fine']['rgb'].detach().cpu())
        rgb_im[:, :rgb_fine.shape[-2], 2*w:2*w+rgb_fine.shape[-1]] = rgb_fine
        # depth_im = torch.cat((depth_im, output['outputs_fine']['depth'].detach().cpu()), dim=-1)
        # depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        # depth_im = img_HWC2CHW(depth_im)
        # acc_map = torch.cat((acc_map, torch.sum(output['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        # acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
        mask_fine = img_HWC2CHW(output["outputs_fine"]["mask"].detach().cpu().unsqueeze(-1))
        mask_im = torch.cat((mask_gt, mask_pred, mask_fine), dim=-1)

    # 위에서 만든 시각화 이미지를 텐서보드에 기록
    writer.add_image(prefix + 'rgb_sources', srcs_im, global_step)
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    # writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    # writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)
    writer.add_image(prefix + 'mask_gt-coarse-fine', mask_im, global_step)

    # write scalar
    # pred_rgb = output['outputs_fine']['rgb'] if output['outputs_fine'] is not None else output['outputs_coarse']['rgb']
    # psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    # writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.train()


def train():
    args = config.get_args()

    out_folder = os.path.join(args.root_dir, 'out', args.exp_name)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    device = "cuda:0"

    # dataset 생성
    category = "teddybear"
    task = "singlesequence"
    single_sequence_id = 0
    datasets = dataset_zoo(
            category=category,
            assert_single_seq=task == "singlesequence",
            dataset_name=f"co3d_{task}",
            test_on_train=False,
            load_point_clouds=False,
            test_restrict_sequence_id=single_sequence_id,
        )
    # dataloader 생성
    dataloaders = dataloader_zoo(
        datasets,
        dataset_name=f"co3d_{task}",
        batch_size=(args.N_src + 1),
        # num_workers=2,
        dataset_len=1000,
        dataset_len_val=10,
        images_per_seq_options=[100],
    )
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # input tensor 차원 설정
    resnet_feature_dim = 32 * 3
    rgb_dim = 3
    segemtation_dim = 1
    pe_dim = args.pe_dim
    d_z = resnet_feature_dim + rgb_dim + segemtation_dim + pe_dim*2*3
    print(f"NerFormer d_z: {d_z}")

    # model 객체 생성
    net = NerFormer(d_z=d_z).to(device)
    feature_net = FeatureNet().to(device)

    # loss
    mse_loss = nn.MSELoss(reduction="sum")         # target RGB <-> GT RGB
    bce_loss = nn.BCELoss(reduction="sum")         # target Mask <-> GT Mask

    # tensorboard writer
    tb_dir = os.path.join(args.root_dir, 'logs/', args.exp_name)
    writer = SummaryWriter(tb_dir)
    print('saving tensorboard files to {}'.format(tb_dir))

    scalars_to_log = {}

    n_iters = args.n_iters
    global_step = 0
    epoch = 0

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    target_idx = 0
    sources = [src_idx for src_idx in range(1, args.N_src + 1)]

    PE = HarmonicEmbedding(n_harmonic_functions=pe_dim).to(device)
    args["PE"] = PE

    while global_step < n_iters + 1:
        for train_data in train_loader:
            time0 = time.time()
            
            ##################### target, source data frame 구성 #####################
            target, srcs = data_to_frame(train_data, device, target_idx, sources)

            ######################## source feature map 구성 ########################
            with torch.no_grad():
                feature_maps = feature_net(srcs["rgb"], srcs["mask"])

            # ray sampler 생성
            ray_sampler = RaySampler(target, target_idx, device, args)
            # 타겟 이미지에서 N_rays개의 ray 샘플링
            ray_batch = ray_sampler.random_sample(args.N_rays, args.ray_sampling_mode)

            output = render_rays(ray_batch, net, net, feature_maps, sources, args)

            optimizer.zero_grad()

            # coarse loss
            coarse_rgb_loss = mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"])
            # coarse_mask_loss = bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
            coarse_loss = coarse_rgb_loss # + coarse_mask_loss

            # fine loss
            fine_rgb_loss = mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"])
            # fine_mask_loss = bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
            fine_loss = fine_rgb_loss # + fine_mask_loss

            total_loss = coarse_loss + fine_loss
            total_loss.backward()

            scalars_to_log['loss'] = total_loss.item()
            
            optimizer.step()

            dt = time.time() - time0

            # loss 출력
            if global_step % args.log_loss == 0 or global_step < 10:
                scalars_to_log['train/coarse-rgb-loss'] = coarse_rgb_loss
                # scalars_to_log['train/coarse-mask-loss'] = coarse_mask_loss

                if output['outputs_fine'] is not None:
                    scalars_to_log['train/fine-rgb-loss'] = fine_rgb_loss
                    # scalars_to_log['train/fine_mask_loss'] = fine_mask_loss

                logstr = '{} Epoch: {}  step: {} '.format(args.exp_name, epoch, global_step)
                for k in scalars_to_log.keys():
                    logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                    writer.add_scalar(k, scalars_to_log[k], global_step)
                print(logstr)
                print('each iter time {:.05f} seconds'.format(dt))

            # checkpoint 저장
            if global_step % args.log_weight == 0:
                print(f"Step[{global_step+1}/{n_iters}]: Checkpoint 저장...")
                save_path = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))

                to_save = {
                            'optimizer': optimizer.state_dict(),
                            'net': net.state_dict(),
                            }
                torch.save(to_save, save_path)

            # validation 시각화 저장
            if global_step % args.log_img == 0:
                ######################## current training data ########################
                print(f"Step[{global_step+1}/{n_iters}]: Training data 결과 저장...")

                # train ray sampler 생성
                train_ray_sampler = RaySampler(target, target_idx, device, args)
                H, W = train_ray_sampler.H, train_ray_sampler.W
                train_gt_img = train_ray_sampler.rgb.reshape(H, W, 3)
                train_gt_mask = train_ray_sampler.mask.reshape(H, W, 1)
                # # log
                log_view_to_tensorboard(args, writer, global_step, net, train_ray_sampler, feature_maps,
                                    srcs, sources, train_gt_img, train_gt_mask, prefix='train/')

                ######################## random validation data ########################
                print(f"Step[{global_step+1}/{n_iters}]: Validation data 결과 저장...")

                val_data = next(iter(val_loader))
                #  validation target, source data frame 구성 
                val_target, val_srcs = data_to_frame(val_data, device, target_idx, sources)

                with torch.no_grad():
                    val_feature_maps = feature_net(val_srcs["rgb"], val_srcs["mask"])

                # validation ray sampler 생성
                val_ray_sampler = RaySampler(val_target, target_idx, device, args)
                H, W = val_ray_sampler.H, val_ray_sampler.W
                val_gt_img = val_ray_sampler.rgb.reshape(H, W, 3)
                val_gt_mask = train_ray_sampler.mask.reshape(H, W, 1)
                # log
                log_view_to_tensorboard(args, writer, global_step, net, val_ray_sampler, val_feature_maps,
                                    val_srcs, sources, val_gt_img, val_gt_mask, prefix='val/')

            global_step += 1

            if global_step > n_iters + 1:
                break
            
        epoch += 1


if __name__ == '__main__':
    train()