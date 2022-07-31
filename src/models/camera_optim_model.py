import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .modules.nerformer import NerFormerArchitecture
from .modules.feature_network import FeatureNetArchitecture

from .modules.ray_sampling import RaySampler
from src.utils.rendering import render_rays
from src.utils.utils import *
from src.utils.proj_ray_distance import *


class FeatureNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_net = FeatureNetArchitecture()
    

    def forward(self, rgbs, masks):
        features = self.feature_net(rgbs, masks)

        return features



class CameraOptimNerFormer(pl.LightningModule):
    def __init__(self, dataset, pe_dim, N_src, N_rays, ray_sampling_mode, center_ratio, 
                N_samples, N_importance, inv_uniform, det, 
                lr, chunk_size, render_stride,
                lambda_mask, lambda_pose, lambda_ray,
                log_img, log_img_step, log_loss_step, log_weight_step, epoch_val_img, val_len,
                model_type, with_colmap, proj_ray_dist_threshold):
        super().__init__()

        # init 함수의 인자로 들어온 하이퍼 파라미터를 self.hparams로 가지고 있음.
        self.save_hyperparameters(logger=False)

        resnet_feature_dim = 32*3
        rgb_dim = 3
        mask_dim = 1
        d_z = resnet_feature_dim + rgb_dim + mask_dim + self.hparams.pe_dim*2*3

        # Nerformer
        self.nerformer = NerFormerArchitecture(d_z)
        for _, child in self.nerformer.named_children():
            for param in child.parameters():
                param.requires_grad = False

        self.pair_idxs = []
        for i in range(0, N_src):
            for j in range(i+1, N_src):
                self.pair_idxs.append((i, j))
        
        # Image Feature Net
        self.feature_net = FeatureNet()
        self.feature_net.freeze()
        # Correspondences Matcher
        self.matcher = init_superglue(0)
        
        self.camera_optim_step = 200
        

    def forward(self, input_tensor):
        output = self.nerformer(input_tensor)

        return output

    
    def configure_optimizers(self):
        return None


    def mse_loss(self, preds, labels):
        return F.mse_loss(preds, labels, reduction="sum")


    def masked_mse_loss(self, preds, labels, masks):
        return F.mse_loss(preds[masks!=0], labels[masks!=0], reduction="sum")


    def bce_loss(self, preds, labels):
        return F.binary_cross_entropy(preds, labels, reduction="sum")

    
    # batch data로부터 타겟뷰, 소스뷰를 선택하여 한 step의 input tensor를 구성할 N_rays개의 ray batch와
    # 그에 따른 feature map을 리턴
    def get_input_data(self, batch_data):
        # target, source data frame 구성 
        target, srcs = data_to_frame(batch_data, self.hparams.N_src, self.hparams.dataset)

        # source feature map 구성 
        with torch.no_grad():
            self.feature_net.eval()
            feature_maps = self.feature_net(srcs["rgb"], srcs["mask"])

        # ray sampler 생성
        ray_sampler = RaySampler(target, srcs["noise_camera"])
        gt_camera_ray_sampler = RaySampler(target, srcs["camera"])
        
        # 타겟 이미지에서 N_rays개의 ray 샘플링
        ray_batch = ray_sampler.random_sample(self.hparams.N_rays, self.hparams.ray_sampling_mode, self.hparams.center_ratio)

        return ray_sampler, gt_camera_ray_sampler, ray_batch, feature_maps, srcs
    
    
    def training_step(self, train_batch, batch_idx):
        # 배치 데이터로부터 rays와 feature map을 샘플링
        ray_sampler, gt_camera_ray_sampler, ray_batch, feature_maps, srcs = self.get_input_data(train_batch)
        
        initial_src_cams = ray_batch["src_cameras"].clone()
        
        rotation_base = ray_batch["src_cameras"].R.clone()
        translation_base = ray_batch["src_cameras"].T.clone()
        
        H, W = ray_sampler.H, ray_sampler.W
        gt_img = ray_sampler.rgb.reshape(H, W, 3)
        gt_mask = ray_sampler.mask.reshape(H, W, 1)
        gt_data = {"img":gt_img, "mask":gt_mask}
        
        # GT 카메라 렌더링 결과 저장
        print(f"Epoch[{self.current_epoch}]/Step[{self.global_step}]: COLMAP GT 카메라 렌더링 결과 저장...")
        gt_imgs = log_view_to_tensorboard_pl(self.nerformer, self.nerformer, gt_camera_ray_sampler, feature_maps, srcs, self.hparams.pe_dim, gt_data,
                                                self.hparams.chunk_size, self.hparams.render_stride,
                                                self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
                                                self.hparams.model_type, self.hparams.with_colmap, 1)
        self.logger.experiment.add_image("[GT]/RGB_GT-coarse-fine", gt_imgs["rgb"], self.global_step)
        self.logger.experiment.add_image("[GT]/Sources", gt_imgs["src"], self.global_step)
        self.logger.experiment.add_image("[GT]/Depth_coarse-fine", gt_imgs["depth"], self.global_step)
        self.logger.experiment.add_image("[GT]/Mask_GT-coarse-fine", gt_imgs["mask"], self.global_step)
        
        # 최적화 전 렌더링 결과 저장
        print(f"Epoch[{self.current_epoch}]/Step[{self.global_step}]: 최적화 전 렌더링 결과 저장...")
        noisy_imgs = log_view_to_tensorboard_pl(self.nerformer, self.nerformer, ray_sampler, feature_maps, srcs, self.hparams.pe_dim, gt_data,
                                                self.hparams.chunk_size, self.hparams.render_stride,
                                                self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
                                                self.hparams.model_type, self.hparams.with_colmap, 1)
        self.logger.experiment.add_image("[Noisy]/RGB_GT-coarse-fine", noisy_imgs["rgb"], self.global_step)
        self.logger.experiment.add_image("[Noisy]/Sources", noisy_imgs["src"], self.global_step)
        self.logger.experiment.add_image("[Noisy]/Depth_coarse-fine", noisy_imgs["depth"], self.global_step)
        self.logger.experiment.add_image("[Noisy]/Mask_GT-coarse-fine", noisy_imgs["mask"], self.global_step)
        
        N_cam = self.hparams.N_src
        # rotation
        delta_roll = nn.Parameter(torch.zeros(N_cam, device=ray_batch["src_cameras"].device))
        delta_pitch = nn.Parameter(torch.zeros(N_cam, device=ray_batch["src_cameras"].device))
        delta_yaw = nn.Parameter(torch.zeros(N_cam, device=ray_batch["src_cameras"].device))
        # translation
        delta_t = nn.Parameter(torch.zeros((N_cam, 3), device=ray_batch["src_cameras"].device))
        
        optimizer = torch.optim.Adam([delta_roll, delta_pitch, delta_yaw, delta_t], lr=0.005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
        optimized_cameras = []
        with torch.autograd.set_detect_anomaly(True):
            for optim_step in range(self.camera_optim_step):
                optimizer.zero_grad()
                   
                # camera parameters optimization
                delta_R = get_rotation_matrix_from_RPY(delta_roll, delta_pitch, delta_yaw)
                delta_T = delta_t

                R_new = torch.matmul(rotation_base, delta_R)
                T_new = translation_base + delta_T
                ray_batch["src_cameras"].R = R_new
                ray_batch["src_cameras"].T = T_new
                optimized_cameras.append(ray_batch["src_cameras"].clone())
                
                
                logstr = 'Batch: {}   /   Optim Step: {}\n'.format(self.global_step, optim_step)
                ########################## Photometric Consistencty Loss (RGB, Mask) ##########################
                # Pretrained Nerformer Inference
                output = render_rays(ray_batch, self.nerformer, self.nerformer, feature_maps, self.hparams.pe_dim,
                                    self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det,
                                    self.hparams.model_type, self.hparams.with_colmap, 1)
                
                
                # Coarse Loss
                coarse_rgb_loss = self.masked_mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
                coarse_mask_loss = self.bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"]) * self.hparams.lambda_mask
                coarse_loss = (coarse_rgb_loss + coarse_mask_loss)
                # Fine Loss
                fine_rgb_loss = self.masked_mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
                fine_mask_loss = self.bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"]) * self.hparams.lambda_mask
                fine_loss = (fine_rgb_loss + fine_mask_loss)
                
                total_loss = coarse_loss + fine_loss
                
                logstr += ' {}: {:.6f} /'.format("coarse_rgb_loss", coarse_rgb_loss)
                logstr += ' {}: {:.6f}\n'.format("coarse_mask_loss", coarse_mask_loss)
                logstr += ' {}: {:.6f} /'.format("fine_rgb_loss", fine_rgb_loss)
                logstr += ' {}: {:.6f}\n'.format("fine_mask_loss", fine_mask_loss)
                
                # ########################## Geometric Consistency Loss (proj_ray_dist) ##########################
                # projected_ray_results = {}
                # proj_ray_dist = 0.
                # # 각 이미지 페어에 대해서 projected ray distance 측정
                # for pair_idx in self.pair_idxs:
                #     img1 = srcs["rgb"][pair_idx[0]].permute(1, 2, 0)
                #     img2 = srcs["rgb"][pair_idx[1]].permute(1, 2, 0)
                    
                #     with torch.no_grad():
                #         result = runSuperGlueSinglePair(self.matcher, img1, img2, 0)
                #         result = preprocess_match(result)
                                                
                #     # correspondence pairs가 존재하고, 또 그 개수가 꽤 많이 나왔다면... (서로 가까운 위치의 카메라 & correspondences의 신뢰도 높음)
                #     if result[0] != None and len(result[0]) >= 20:
                #         # 전체 optimized step에 대해서 loss 측정...
                #         # Coarse Loss - ray
                #         pair_rays = get_pair_rays(pair_idx, srcs["noise_camera"], 800, 800, result)
                #         loss = proj_ray_dist_loss_single(result[0], result[1], srcs["noise_camera"], pair_idx, pair_rays, 
                #                                         "train", 800, 800, proj_ray_dist_threshold=self.hparams.proj_ray_dist_threshold)[0]
                #         proj_ray_dist += torch.nan_to_num(loss)

                #         # 시각화 스텝이면 시각화 할 결과를 저장
                #         if self.hparams.log_img and self.global_step % self.hparams.log_img_step == 0:
                #             projected_ray_results[str(pair_idx)] = {"result": result, "pair_rays": pair_rays}
                #     else:
                #         projected_ray_results[str(pair_idx)] = None
                # total_loss += (proj_ray_dist * self.hparams.lambda_ray)
                
                # logstr += ' {}: {:.6f}\n'.format("proj_ray_dist", proj_ray_dist)
                
                logstr += ' {}: {:.6f}'.format("total_loss", total_loss)
                print(logstr)
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()

        # 최적화 후 렌더링 결과 저장
        print(f"Epoch[{self.current_epoch}]/Step[{self.global_step}]: 최적화 후 렌더링 결과 저장...")
        optim_imgs = log_view_to_tensorboard_pl(self.nerformer, self.nerformer, ray_sampler, feature_maps, srcs, self.hparams.pe_dim, gt_data,
                                                self.hparams.chunk_size, self.hparams.render_stride,
                                                self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
                                                self.hparams.model_type, self.hparams.with_colmap, 1)
        self.logger.experiment.add_image("[Optim]/RGB_GT-coarse-fine", optim_imgs["rgb"], self.global_step)
        self.logger.experiment.add_image("[Optim]/Sources", optim_imgs["src"], self.global_step)
        self.logger.experiment.add_image("[Optim]/Depth_coarse-fine", optim_imgs["depth"], self.global_step)
        self.logger.experiment.add_image("[Optim]/Mask_GT-coarse-fine", optim_imgs["mask"], self.global_step)
        
        # camera optimizing 시각화
        # 전체 카메라 포즈 변화 시각화
        all_cam_figure = get_noise_camera_figure(srcs["camera"], [initial_src_cams], True)
        self.logger.experiment.add_figure("camera/all_initial_noisy_cams", all_cam_figure, self.global_step)
        all_cam_figure = get_noise_camera_figure(srcs["camera"], [optimized_cameras[-1]], True)
        self.logger.experiment.add_figure("camera/all_optimized_cams", all_cam_figure, self.global_step)
        
        # 카메라 각각 하나씩 시각화
        interval = int(self.camera_optim_step / 5)
        cam_figures = get_noise_camera_figure(srcs["camera"], optimized_cameras[::interval], False)               
        for i, cam_fig in enumerate(cam_figures):
            self.logger.experiment.add_figure("camera/optimized_cam_%d"%(i+1), cam_fig, self.global_step)               
        
        # # correspondences 재투영 결과 비교 시각화
        # # COLMAP GT 카메라 vs Final Optimized 카메라 (from Noise or from COLMAP GT)
        # final_optimized_camera = optimized_cameras[-1]
        # for pair_idx in self.pair_idxs:
        #     if projected_ray_results[str(pair_idx)] == None:
        #         continue
        #     corr_result = projected_ray_results[str(pair_idx)]["result"]
            
        #     # COLMAP 또는 (COLMAP + Noise) 카메라로 정의된 레이와 비교
        #     initial_pair_rays = get_pair_rays(pair_idx, initial_src_cams, 800, 800, corr_result)
        #     optim_pair_rays = projected_ray_results[str(pair_idx)]["pair_rays"]
            
        #     fig = get_correspondences_figure(srcs["rgb"].permute(0, 2, 3, 1), initial_src_cams, initial_pair_rays, final_optimized_camera, optim_pair_rays, pair_idx, corr_result,
        #                                         self.hparams.with_colmap)
        #     self.logger.experiment.add_figure("train/camera/projected_ray_on_(%d, %d)"%(pair_idx[0], pair_idx[1]), fig, self.global_step)
        
        # ############################### logging ###############################
        # # Loss
        # if self.global_step % self.hparams.log_loss_step == 0 or self.global_step < 10:
        #     logstr = 'Epoch: {}  step: {} \n'.format(self.current_epoch, self.global_step)
        #     logstr += ' {}: {:.6f} /'.format("coarse_rgb_loss", coarse_rgb_loss)
        #     logstr += ' {}: {:.6f}\n'.format("coarse_mask_loss", coarse_mask_loss)
        #     logstr += ' {}: {:.6f} /'.format("fine_rgb_loss", fine_rgb_loss)
        #     logstr += ' {}: {:.6f}\n'.format("fine_mask_loss", fine_mask_loss)
            
        #     logstr += ' {}: {:.6f}'.format("proj_ray_dist", proj_ray_dist)
        #     print(logstr)

        #     self.log('train/coarse_rgb_loss', coarse_rgb_loss)
        #     self.log('train/coarse_mask_loss', coarse_mask_loss)
        #     self.log('train/fine_rgb_loss', fine_rgb_loss)
        #     self.log('train/fine_mask_loss', fine_mask_loss)
        #     self.log('train/proj_ray_dist_loss', proj_ray_dist)
        #     self.log('train/total_loss', total_loss)
            
        # # 시각화
        # if self.hparams.log_img and self.global_step % self.hparams.log_img_step == 0:
        #     print(f"Epoch[{self.current_epoch}]/Step[{self.global_step}]: Training 시각화 결과 저장...")
            
        #     H, W = ray_sampler.H, ray_sampler.W
        #     gt_img = ray_sampler.rgb.reshape(H, W, 3)
        #     gt_mask = ray_sampler.mask.reshape(H, W, 1)
        #     gt_data = {"img":gt_img, "mask":gt_mask}

        #     tb_imgs = log_view_to_tensorboard_pl(self.nerformer, self.nerformer, ray_sampler, feature_maps, srcs, self.hparams.pe_dim, gt_data,
        #                                         self.hparams.chunk_size, self.hparams.render_stride,
        #                                         self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
        #                                         self.hparams.model_type, self.hparams.with_colmap, 1)

        #     self.logger.experiment.add_image("train/[rgb]sources", tb_imgs["src"], self.global_step)
        #     self.logger.experiment.add_image("train/[rgb]GT-coarse-fine", tb_imgs["rgb"], self.global_step)
        #     self.logger.experiment.add_image("train/[depth]coarse-fine", tb_imgs["depth"], self.global_step)
        #     self.logger.experiment.add_image("train/[mask]GT-coarse-fine", tb_imgs["mask"], self.global_step)

        #     # camera optimizing 시각화
        #     # 전체 카메라 포즈 변화 시각화
        #     all_cam_figure = get_noise_camera_figure(srcs["camera"], [initial_src_cams], True)
        #     self.logger.experiment.add_figure("train/camera/all_initial_noisy_cams", all_cam_figure, self.global_step)
        #     all_cam_figure = get_noise_camera_figure(srcs["camera"], [optimized_cameras[-1]], True)
        #     self.logger.experiment.add_figure("train/camera/all_optimized_cams", all_cam_figure, self.global_step)
            
        #     # 카메라 각각 하나씩 시각화
        #     interval = int(self.camera_optim_step / 5)
        #     cam_figures = get_noise_camera_figure(srcs["camera"], optimized_cameras[::interval], False)               
        #     for i, cam_fig in enumerate(cam_figures):
        #         self.logger.experiment.add_figure("train/camera/optimized_cam_%d"%(i+1), cam_fig, self.global_step)               
            
        #     # correspondences 재투영 결과 비교 시각화
        #     # COLMAP GT 카메라 vs Final Optimized 카메라 (from Noise or from COLMAP GT)
        #     final_optimized_camera = optimized_cameras[-1]
        #     for pair_idx in self.pair_idxs:
        #         if projected_ray_results[str(pair_idx)] == None:
        #             continue
        #         corr_result = projected_ray_results[str(pair_idx)]["result"]
                
        #         # COLMAP 또는 (COLMAP + Noise) 카메라로 정의된 레이와 비교
        #         initial_pair_rays = get_pair_rays(pair_idx, initial_src_cams, 800, 800, corr_result)
        #         optim_pair_rays = projected_ray_results[str(pair_idx)]["pair_rays"]
                
        #         fig = get_correspondences_figure(srcs["rgb"].permute(0, 2, 3, 1), initial_src_cams, initial_pair_rays, final_optimized_camera, optim_pair_rays, pair_idx, corr_result,
        #                                             self.hparams.with_colmap)
        #         self.logger.experiment.add_figure("train/camera/projected_ray_on_(%d, %d)"%(pair_idx[0], pair_idx[1]), fig, self.global_step)
        
        # return total_loss


    # def validation_step(self, val_batch, batch_idx):
    #     # 배치 데이터로부터 rays와 feature map을 샘플링
    #     ray_sampler, ray_batch, feature_maps, srcs = self.get_input_data(val_batch)
    #     # 나중에 결과 비교할 초기 카메라를 복사해둔다.
    #     if self.hparams.with_colmap:
    #         initial_src_cams = srcs["camera"].clone()
    #     else:
    #         initial_src_cams = srcs["noise_camera"].clone()
            
    #     # Nerformer Inference
    #     output = render_rays(ray_batch, self.nerformer, self.nerformer, feature_maps, self.hparams.pe_dim,
    #                         self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
    #                         self.hparams.model_type, self.hparams.with_colmap, self.training_phase)

    #     ############ Coarse Loss - (rgb, mask)
    #     coarse_rgb_loss = self.masked_mse_loss(output["outputs_coarse"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
    #     coarse_mask_loss = self.bce_loss(output["outputs_coarse"]["mask"][..., 0], ray_batch["mask"]) * self.hparams.lambda_mask
    #     coarse_loss = coarse_rgb_loss + coarse_mask_loss

    #     ############ Fine Loss - (rgb, mask)
    #     fine_rgb_loss = self.masked_mse_loss(output["outputs_fine"]["rgb"], ray_batch["rgb"], ray_batch["mask"])
    #     fine_mask_loss = self.bce_loss(output["outputs_fine"]["mask"][..., 0], ray_batch["mask"]) * self.hparams.lambda_mask
    #     fine_loss = fine_rgb_loss + fine_mask_loss
        
    #     ########################## Camera Optimizing Loss ##########################
    #     if self.hparams.model_type == "BA" and self.training_phase == 1:
    #         ############ Coarse Loss - pose
    #         coarse_pose_reg_loss = self.delta_pose_reg_loss(output["camera_outputs_coarse"]["delta_pose"]) * self.hparams.lambda_pose
    #         coarse_loss += coarse_pose_reg_loss
    #         ############ Fine Loss - pose
    #         fine_pose_reg_loss = self.delta_pose_reg_loss(output["camera_outputs_fine"]["delta_pose"]) * self.hparams.lambda_pose
    #         fine_loss += fine_pose_reg_loss
            
    #         projected_ray_results = {}
    #         coarse_proj_ray_dist = 0
    #         fine_proj_ray_dist = 0
    #         for pair_idx in self.pair_idxs:
    #             img1 = srcs["rgb"][pair_idx[0]].permute(1, 2, 0)
    #             img2 = srcs["rgb"][pair_idx[1]].permute(1, 2, 0)
                
    #             with torch.no_grad():
    #                 result = runSuperGlueSinglePair(self.matcher, img1, img2, 0)
    #                 result = preprocess_match(result)
                
    #             # correspondence pairs가 존재하고, 또 그 개수가 꽤 많이 나왔다면... (서로 가까운 위치의 카메라 & correspondences의 신뢰도 높음)
    #             if result[0] != None and len(result[0]) >= 20:
    #                 # 전체 optimized step에 대해서 loss 측정...
    #                 ############ Coarse Loss - ray
    #                 for coarse_optimized_src_cams in output["camera_outputs_coarse"]["optimized_camera"]:
    #                     # coarse_optimized_src_cams: (N_src)
    #                     pair_rays = get_pair_rays(pair_idx, coarse_optimized_src_cams, 800, 800, result)
    #                     loss = proj_ray_dist_loss_single(result[0], result[1], coarse_optimized_src_cams, pair_idx, pair_rays, 
    #                                                     "train", 800, 800, proj_ray_dist_threshold=self.hparams.proj_ray_dist_threshold)[0]
    #                     coarse_proj_ray_dist += torch.nan_to_num(loss)
    #                 ############ Fine Loss - ray
    #                 for fine_optimized_src_cams in output["camera_outputs_fine"]["optimized_camera"]:
    #                     # fine_optimized_src_cams: (N_src)
    #                     pair_rays = get_pair_rays(pair_idx, fine_optimized_src_cams, 800, 800, result)
    #                     loss = proj_ray_dist_loss_single(result[0], result[1], fine_optimized_src_cams, pair_idx, pair_rays, 
    #                                                     "train", 800, 800, proj_ray_dist_threshold=self.hparams.proj_ray_dist_threshold)[0]
    #                     fine_proj_ray_dist += torch.nan_to_num(loss)

    #                 projected_ray_results[str(pair_idx)] = {"result": result, "pair_rays": pair_rays}
    #             else:
    #                 projected_ray_results[str(pair_idx)] = None

    #         coarse_loss += (coarse_proj_ray_dist * self.hparams.lambda_ray)
    #         fine_loss += (fine_proj_ray_dist * self.hparams.lambda_ray)
    #     else:
    #         projected_ray_results = None
            
    #     # Total Loss
    #     total_loss = coarse_loss + fine_loss

    #     ########################## logging ##########################
    #     # Loss
    #     self.log('val/coarse_rgb_loss', coarse_rgb_loss)
    #     self.log('val/coarse_mask_loss', coarse_mask_loss)
        
    #     self.log('val/fine_rgb_loss', fine_rgb_loss)
    #     self.log('val/fine_mask_loss', fine_mask_loss)
        
    #     self.log('val/total_loss', total_loss)
        
    #     if self.hparams.model_type == "BA" and self.training_phase == 1:
    #         self.log('val/coarse_delta_pose_reg_loss', coarse_pose_reg_loss)
    #         self.log('val/coarse_proj_ray_dist_loss', coarse_proj_ray_dist)
            
    #         self.log('val/fine_delta_pose_reg_loss', fine_pose_reg_loss)
    #         self.log('val/fine_proj_ray_dist', fine_proj_ray_dist)
            
    #     # 마지막 배치의 시각화에 필요한 값들만 전달
    #     if batch_idx == self.hparams.val_len - 1 and self.current_epoch % self.hparams.epoch_val_img == 0:
    #         val_step_output = {
    #                             "srcs":srcs,
    #                             "ray_sampler":ray_sampler,
    #                             "feature_maps":feature_maps,
    #                             "output":output,
    #                             "initial_src_cams":initial_src_cams,
    #                             "projected_ray_results":projected_ray_results,
    #                         }

    #         return val_step_output


    # # validation 데이터 시각화 
    # def validation_epoch_end(self, outputs):
    #     if self.current_epoch % self.hparams.epoch_val_img == 0:
    #         print(f"Epoch[{self.current_epoch+1}]/Step[{self.global_step+1}]: Validation 시각화 결과 저장...")
            
    #         val_outputs = outputs[-1]

    #         val_ray_sampler = val_outputs["ray_sampler"]
    #         val_srcs = val_outputs["srcs"]
    #         val_feature_maps = val_outputs["feature_maps"]

    #         output = val_outputs["output"]
    #         initial_src_cams = val_outputs["initial_src_cams"]
    #         projected_ray_results = val_outputs["projected_ray_results"]
            
    #         H, W = val_ray_sampler.H, val_ray_sampler.W
    #         gt_img = val_ray_sampler.rgb.reshape(H, W, 3)
    #         gt_mask = val_ray_sampler.mask.reshape(H, W, 1)
    #         gt_data = {"img":gt_img, "mask":gt_mask}

    #         # 전체 이미지 렌더하기...
    #         tb_imgs = log_view_to_tensorboard_pl(self.nerformer, self.nerformer, val_ray_sampler, val_feature_maps, val_srcs, self.hparams.pe_dim, gt_data,
    #                                             self.hparams.chunk_size, self.hparams.render_stride,
    #                                             self.hparams.N_samples, self.hparams.N_importance, self.hparams.inv_uniform, self.hparams.det, 
    #                                             self.hparams.model_type, self.hparams.with_colmap, self.training_phase)

    #         self.logger.experiment.add_image("val/[rgb]sources", tb_imgs["src"], self.global_step)
    #         self.logger.experiment.add_image("val/[rgb]GT-coarse-fine", tb_imgs["rgb"], self.global_step)
    #         self.logger.experiment.add_image("val/[depth]coarse-fine", tb_imgs["depth"], self.global_step)
    #         self.logger.experiment.add_image("val/[mask]GT-coarse-fine", tb_imgs["mask"], self.global_step)
            
    #         # camera optimizing 시각화
    #         if self.hparams.model_type == "BA" and self.training_phase == 1:
    #             # 전체 카메라 포즈 변화 시각화
    #             all_cam_figure = get_noise_camera_figure(val_srcs["camera"], output["camera_outputs_fine"]["optimized_camera"], True)
    #             self.logger.experiment.add_figure("val/camera/all_optimized_cams", all_cam_figure, self.global_step)
                
    #             # 카메라 각각 하나씩 시각화
    #             cam_figures = get_noise_camera_figure(val_srcs["camera"], output["camera_outputs_fine"]["optimized_camera"], False)
    #             for i, cam_fig in enumerate(cam_figures):
    #                 self.logger.experiment.add_figure("val/camera/optimized_cam_%d"%(i+1), cam_fig, self.global_step)
                               
    #             # correspondences 재투영 결과 비교 시각화
    #             # COLMAP GT 카메라 vs Optimized 카메라 (from Noise or from COLMAP GT)
    #             final_optimized_camera = output["camera_outputs_fine"]["optimized_camera"][-1]
    #             for pair_idx in self.pair_idxs:
    #                 if projected_ray_results[str(pair_idx)] == None:
    #                     continue
    #                 corr_result = projected_ray_results[str(pair_idx)]["result"]
                    
    #                 # COLMAP 또는 (COLMAP + Noise) 카메라로 정의된 레이와 비교
    #                 initial_pair_rays = get_pair_rays(pair_idx, initial_src_cams, 800, 800, corr_result)
    #                 optim_pair_rays = projected_ray_results[str(pair_idx)]["pair_rays"]
                    
    #                 fig = get_correspondences_figure(val_srcs["rgb"].permute(0, 2, 3, 1), initial_src_cams, initial_pair_rays, final_optimized_camera, optim_pair_rays, pair_idx, corr_result,
    #                                                  self.hparams.with_colmap)
    #                 self.logger.experiment.add_figure("val/camera/projected_ray_on_(%d, %d)"%(pair_idx[0], pair_idx[1]), fig, self.global_step)