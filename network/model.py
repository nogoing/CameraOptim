import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from network.nerformer import NerFormerArchitecture
from network.feature_network import FeatureNetArchitecture

from positional_embedding import HarmonicEmbedding
from ray_sampling import RaySampler
from rendering import render_rays
from utils import *



class FeatureNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_net = FeatureNetArchitecture()
    

    def forward(self, x):
        feature = self.feature_net(x)

        return feature



class NerFormer(pl.LightningModule):
    def __init__(self, d_z, args):
        super().__init__()

        # NerFormer
        self.nerformer = NerFormerArchitecture(d_z)
        # Image Feature Net
        self.feature_net = FeatureNet()
        self.feature_net.freeze()
        # Positional Embedding
        self.PE = HarmonicEmbedding(n_harmonic_functions=args.pe_dim)
        self.args = args

    def forward(self, x):
        output = self.nerformer(x)

        return output

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerformer.parameters(), lr=0.0005)

        return optimizer


    def training_step(self, train_batch, batch_idx):
        # target_idx = 0
        # sources = [src_idx for src_idx in range(1, args.N_src + 1)]


        # train_batch에서 targe과 source를 정의
        target, srcs = data_to_frame(train_batch, args.device, target_idx, sources)

        # source 이미지로부터 iamge feature 추출
        self.feature_net.eval()
        with torch.no_grad():
            feature_maps = self.feature_net(srcs["rgb"], srcs["mask"])

        # target 이미지에서 학습 rays 샘플링
        ray_sampler = RaySampler(target, target_idx, args.device, args)
        ray_batch = ray_sampler.random_sample(args.N_rays, args.ray_sampling_mode)

        # Inference
        output = render_rays(ray_batch, net, net, feature_maps, sources, args)

        coarse_rgb_loss = mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"])
        # coarse_mask_loss = bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
        self.log('coarse_rgb_loss', coarse_rgb_loss)

        fine_rgb_loss = mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"])
        # fine_mask_loss = bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
        self.log('fine_rgb_loss', fine_rgb_loss)

        total_loss = coarse_rgb_loss + fine_rgb_loss
        self.log('train_loss', total_loss)

        return loss


    def validation_step(self, val_batch, val_idx):
        target_idx = 0
        sources = [src_idx for src_idx in range(1, args.N_src + 1)]


        # dataloader 아이템에서 targe과 source를 정의
        target, srcs = data_to_frame(val_batch, args.device, target_idx, sources)

        # source 이미지로부터 iamge feature 추출
        with torch.no_grad():
            feature_maps = feature_net(srcs["rgb"], srcs["mask"])

        # target 이미지에서 학습 rays 샘플링
        ray_sampler = RaySampler(target, target_idx, args.device, args)
        ray_batch = ray_sampler.random_sample(args.N_rays, args.ray_sampling_mode)

        # Inference
        output = render_rays(ray_batch, net, net, feature_maps, sources, args)

        coarse_rgb_loss = mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"])
        # coarse_mask_loss = bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
        self.log('val_coarse_rgb_loss', coarse_rgb_loss)

        fine_rgb_loss = mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"])
        # fine_mask_loss = bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
        self.log('val_fine_rgb_loss', fine_rgb_loss)

        total_loss = coarse_rgb_loss + fine_rgb_loss
        self.log('val_loss', total_loss)
