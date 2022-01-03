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
    

    def forward(self, rgbs, masks):
        features = self.feature_net(rgbs, masks)

        return features



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
    

    def forward(self, input_tensor):
        output = self.nerformer(input_tensor)

        return output

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerformer.parameters(), lr=0.0005)

        return optimizer


    def mse_loss(self, preds, labels):
        return F.mse_loss(preds, labels, reduction="sum")


    def masked_mse_loss(self, preds, labels, masks):
        return F.mse_loss(preds[masks!=0], labels[masks!=0], reduction="sum")


    def bce_loss(self, preds, labels):
        return F.binary_cross_entropy(preds, labels, reduction="sum")


    def training_step(self, train_batch, batch_idx):
        # train_batch에서 targe과 source를 정의
        target, srcs = data_to_frame(train_batch, self.args.N_src)

        # source 이미지로부터 iamge feature 추출
        self.feature_net.eval()
        with torch.no_grad():
            feature_maps = self.feature_net(srcs["rgb"], srcs["mask"])

        # target 이미지에서 학습 rays 샘플링
        ray_sampler = RaySampler(target, srcs["camera"])
        ray_batch = ray_sampler.random_sample(self.args.N_rays, self.args.ray_sampling_mode, self.args.center_ratio)

        # Inference
        output = render_rays(ray_batch, self.nerformer, self.nerformer, feature_maps, self.PE, self.args)

        # Loss
        coarse_rgb_loss = self.masked_mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
        coarse_mask_loss = self.bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
        coarse_loss = coarse_rgb_loss + coarse_mask_loss

        fine_rgb_loss = self.masked_mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
        fine_mask_loss = self.bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
        fine_loss = fine_rgb_loss + fine_mask_loss

        total_loss = coarse_loss + fine_loss

        ########################## logging ##########################
        # training loss
        if self.global_step % self.args.step_loss == 0 or self.global_step < 10:
            logstr = 'Epoch: {}  step: {} '.format(self.current_epoch, self.global_step)
            logstr += ' {}: {:.6f}'.format("coarse_rgb_loss", coarse_rgb_loss)
            logstr += ' {}: {:.6f}'.format("fine_rgb_loss", fine_rgb_loss)
            print(logstr)

            self.log('train/coarse_rgb_loss', coarse_rgb_loss)
            self.log('train/fine_rgb_loss', fine_rgb_loss)
            self.log('train/total_loss', total_loss)

        # training 데이터 시각화
        if self.global_step % self.args.step_img == 0:
            print(f"Step[{self.global_step+1}]: Training 시각화 결과 저장...")
            H, W = ray_sampler.H, ray_sampler.W
            gt_img = ray_sampler.rgb.reshape(H, W, 3)
            gt_mask = ray_sampler.mask.reshape(H, W, 1)
            gts = {
                    "img":gt_img,
                    "mask":gt_mask,
            }

            tb_imgs = log_view_to_tensorboard_pl(self.args, self.nerformer, self.nerformer, ray_sampler, feature_maps, srcs, self.PE, gts)
            srcs_im = tb_imgs["src"]
            rgb_im = tb_imgs["rgb"]
            depth_im = tb_imgs["depth"]
            mask_im = tb_imgs["mask"]

            self.logger.experiment.add_image("train/[rgb]sources", srcs_im, self.global_step)
            self.logger.experiment.add_image("train/[rgb]GT-coarse-fine", rgb_im, self.global_step)
            self.logger.experiment.add_image("train/[rgb]coarse-fine", depth_im, self.global_step)
            self.logger.experiment.add_image("train/[rgb]GT-coarse-fine", mask_im, self.global_step)

        # # weight
        # if self.global_step % self.args.step_weights == 0:
        #     print(f"Step[{self.global_step+1}/]: Checkpoint 저장...")
        #     save_path = os.path.join(self.dirpath, "model_{:06d}.ckpt".format(self.global_step))

        #     self.trainer.save_checkpoint(save_path)


    def validation_step(self, val_batch, batch_idx):
        # dataloader 아이템에서 targe과 source를 정의
        target, srcs = data_to_frame(val_batch, self.args.N_src)

        # source 이미지로부터 iamge feature 추출
        self.feature_net.eval()
        with torch.no_grad():
            feature_maps = self.feature_net(srcs["rgb"], srcs["mask"])

        # target 이미지에서 학습 rays 샘플링
        ray_sampler = RaySampler(target, srcs["camera"])
        ray_batch = ray_sampler.random_sample(self.args.N_rays, self.args.ray_sampling_mode, self.args.center_ratio)

        # Inference
        output = render_rays(ray_batch, self.nerformer, self.nerformer, feature_maps, self.PE, self.args)

        coarse_rgb_loss = self.mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"])
        # coarse_mask_loss = self.bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"])
        self.log('val/coarse_rgb_loss', coarse_rgb_loss)

        fine_rgb_loss = self.mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"])
        # fine_mask_loss = self.bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"])
        self.log('val/fine_rgb_loss', fine_rgb_loss)

        total_loss = coarse_rgb_loss + fine_rgb_loss
        self.log('val/total_loss', total_loss, prog_bar=True)

        # 마지막 배치의 결과만 logging 함수로 전달
        if batch_idx == self.args.val_len - 1 and self.current_epoch % self.args.epoch_val_img == 0:
            val_step_output = {
                                "srcs":srcs,
                                "ray_sampler":ray_sampler,
                                "feature_maps":feature_maps
                            }

            return val_step_output


    # validation 데이터 시각화 
    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.args.epoch_val_img == 0:
            print(f"Step[{self.global_step+1}]: Validation 시각화 결과 저장...")
            val_outputs = outputs[-1]
            val_srcs = val_outputs["srcs"]
            val_feature_maps = val_outputs["feature_maps"]
            val_ray_sampler = val_outputs["ray_sampler"]

            H, W = val_ray_sampler.H, val_ray_sampler.W

            gt_img = val_ray_sampler.rgb.reshape(H, W, 3)
            gt_mask = val_ray_sampler.mask.reshape(H, W, 1)
            gts = {
                    "img":gt_img,
                    "mask":gt_mask,
            }

            # 전체 이미지 렌더하기...
            tb_imgs = log_view_to_tensorboard_pl(self.args, self.nerformer, self.nerformer, val_ray_sampler, val_feature_maps, val_srcs, self.PE, gts)
            srcs_im = tb_imgs["src"]
            rgb_im = tb_imgs["rgb"]
            depth_im = tb_imgs["depth"]
            mask_im = tb_imgs["mask"]

            self.logger.experiment.add_image("val/[rgb]sources", srcs_im, self.global_step)
            self.logger.experiment.add_image("val/[rgb]GT-coarse-fine", rgb_im, self.global_step)
            self.logger.experiment.add_image("val/[rgb]coarse-fine", depth_im, self.global_step)
            self.logger.experiment.add_image("val/[rgb]GT-coarse-fine", mask_im, self.global_step)