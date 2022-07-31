import torch
import torch.nn as nn
import os

from src.utils.rendering import positional_embedding
from src.utils.utils import get_rotation_matrix_from_RPY, get_relative_pose, camera_update_and_feature_resampling

class BANerFormerArchitecture(nn.Module):
    def __init__(self, d_z, pe_dim):
        super(BANerFormerArchitecture, self).__init__()

        self.d_z = d_z  # Input feature의 차원
        self.pe_dim = pe_dim
        self.camera_embed_dim = pe_dim*3*2*4
        self.resampling_dim = d_z - pe_dim*3*2
        # camera pose encoding layer
        self.pose_encoding = nn.Linear(self.camera_embed_dim, self.d_z)
        ##################################################################
        # input: (N_rays(=Batch), N_s, N_src, d_z)
        self.linear_1 = nn.Linear(d_z, 80, bias=False)
        self.p_head_1 = nn.Linear(80, 6)    # output: pitch, roll, yaw, Translation(3)
        torch.nn.init.zeros_(self.p_head_1.weight)
        torch.nn.init.zeros_(self.p_head_1.bias)
        self.resampling_feature_linear_1 = nn.Linear(self.resampling_dim, 80)
        self.resampling_camera_linear_1 = nn.Linear(self.d_z, 80)
        self.layer_norm_1 = nn.LayerNorm(80)
        # (N_rays, N_s, N_src, 80)
        self.TE_1 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        # (N_rays, N_s, N_src, 80)
        self.TE_1_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        # (N_rays, N_s, N_src, 80)
        self.TE_1_3 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        ##################################################################
        self.dim_linear_1 = nn.Linear(80, 40)
        self.p_head_2 = nn.Linear(40, 6)   # output: pitch, roll, yaw, Translation(3)
        torch.nn.init.zeros_(self.p_head_2.weight)
        torch.nn.init.zeros_(self.p_head_2.bias)
        self.resampling_feature_linear_2 = nn.Linear(self.resampling_dim, 40)
        self.resampling_camera_linear_2 = nn.Linear(self.d_z, 40)
        self.layer_norm_2 = nn.LayerNorm(40)
        # (N_rays, N_s, N_src, 40)
        self.TE_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=4),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=4)           # Ray transformer encoder
        )
        # (N_rays, N_s, N_src, 40)
        self.TE_2_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=4),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=4)           # Ray transformer encoder
        )
        # (N_rays, N_s, N_src, 40)
        self.TE_2_3 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=4),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=4)           # Ray transformer encoder
        )

        self.dim_linear_2 = nn.Linear(40, 20)
        # output shape: (N_rays, N_s, N_src, 20)

        self.weight_layer = nn.Sequential(
            nn.Linear(20, 1),
            nn.Softmax(dim=-2)      # 특정 sample에서 각 src들에 대한 값의 합이 1이 되도록 차원을 설정
        )

        # color function head
        # Output shape: (N_s, 3)
        self.c_head = nn.Sequential(
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 3)
        )

        # opacity function head
        # Output shape: (N_s, 1)
        self.f_head = nn.Sequential(
            nn.Linear(20, 1),
            nn.ReLU()
        )

    
    def make_cam_token(self, tgt_cam, srcs_cam):
        relative_pose = get_relative_pose(tgt_cam, srcs_cam)   # row major
        relative_R = relative_pose[:, :3, :3]
        relative_T = relative_pose[:, 3, :3]
        
        # PE에는 마지막 차원값이 한 축의 (x, y, z) 값이므로
        # row major로 넣어주어야 한다.
        rotation_embed = positional_embedding(self.pe_dim, 0.1, relative_R)     # (N_src, 3, pe_dim*2*3)
        x_embed = rotation_embed[:, 0, :]
        y_embed = rotation_embed[:, 1, :]
        z_embed = rotation_embed[:, 2, :]
        
        translation_embed = positional_embedding(self.pe_dim, 0.1, relative_T)  # (N_src, pe_dim*2*3)
        
        cam_pose = torch.cat((x_embed, y_embed, z_embed, translation_embed), dim=1)    # (N_src, pe_dim*2*3*4)
        
        cam_token = self.pose_encoding(cam_pose)   # (N_src, D_z)

        return cam_token
    
    
    def feature_update(self, x, feature_maps, initial_feature, tgt_cam, srcs_cam, pts, stage):
        if stage == "first":
            delta_pose = self.p_head_1(x[:, 0, :, :].clone())     # (N_rays, N_src, 6)
        else:
            delta_pose = self.p_head_2(x[:, 0, :, :].clone())     # (N_rays, N_src, 6)
        delta_pose = delta_pose.mean(dim=0)  # (N_src, 6)    # 모든 rays에서의 결과를 평균
        
        roll, pitch, yaw = delta_pose[..., 0], delta_pose[..., 1], delta_pose[..., 2]
        delta_rotation = get_rotation_matrix_from_RPY(roll, pitch, yaw)
        delta_translation = delta_pose[..., 3:]
        
        # srcs_cam update & feature resampling
        resampling_tensor = camera_update_and_feature_resampling(feature_maps, pts, self.pe_dim, tgt_cam, srcs_cam,
                                                                 delta_rotation, delta_translation)   # (N_rays, N_s, N_src, D_z)
        # initial tensor에서 PE 제외한 image feature 부분만 빼준다.
        resampling_tensor -= initial_feature[:, 1:, :, :resampling_tensor.shape[-1]]

        optimized_cam_token = self.make_cam_token(tgt_cam, srcs_cam)     # (1, N_src, D_z)
        optimized_cam_token -= initial_feature[0, 0, :, :]              # initial tensor에서 카메라 토큰끼리 빼준다. (800개 레이에서 동일)
        
        if stage == "first":
            optimized_cam_token = self.resampling_camera_linear_1(optimized_cam_token)  # (1, N_src, 80)
            resampling_tensor = self.resampling_feature_linear_1(resampling_tensor)     # (N_rays, N_s, N_src, 80)
        else:
            optimized_cam_token = self.resampling_camera_linear_2(optimized_cam_token)  # (1, N_src, 40)
            resampling_tensor = self.resampling_feature_linear_2(resampling_tensor)     # (N_rays, N_s, N_src, 40)
        optimized_cam_token = optimized_cam_token.repeat(x.shape[0], 1, 1)  # (N_rays, N_src, 80 or 40)

        # camera feature 업데이트 (=add)
        x[:,0,:,:] += optimized_cam_token
        # image feature 업데이트 (=add)
        x[:,1:,:,:] += resampling_tensor
        

        return x, delta_pose
    
    
    def forward(self, input_tensor, feature_maps, tgt_cam, srcs_cam, pts, cam_optim=0, visualize=False):
        # input_tensor: (N_rays(=Batch), N_s, N_src, D_z)

        srcs_cam = srcs_cam.clone()
        
        cam_token = self.make_cam_token(tgt_cam, srcs_cam)     # (N_src, D_z)
        cam_token = cam_token.repeat(input_tensor.shape[0], 1, 1, 1)  # (N_rays, 1, N_src, D_z)
        # feature tensor에 카메라 토큰 추가 >> sample 차원에서 concat
        input_tensor = torch.cat((cam_token, input_tensor), dim=1)   # (N_rays, N_s+1, N_src, D_z)
        
        optimized_cams = []
        deltas = []
        ######## stage 1 ########
        x = self.linear_1(input_tensor)     # (N_rays, N_s+1, N_src, 80)
        
        x = self.TE_1(x)                    # (N_rays, N_s+1, N_src, 80)
        if cam_optim:
            x, delta = self.feature_update(x, feature_maps, input_tensor, tgt_cam, srcs_cam, pts, stage="first")
            optimized_cams.append(srcs_cam.clone())
            deltas.append(delta)
            x = self.layer_norm_1(x)
        x = self.TE_1_2(x)                    # (N_rays, N_s+1, N_src, 80)
        if cam_optim:
            x, delta = self.feature_update(x, feature_maps, input_tensor, tgt_cam, srcs_cam, pts, stage="first")
            optimized_cams.append(srcs_cam.clone())
            deltas.append(delta)
            x = self.layer_norm_1(x)
        x = self.TE_1_3(x)                    # (N_rays, N_s+1, N_src, 80)
        ########################
        x = self.dim_linear_1(x)        # (N_rays, N_s, N_src, 40)
        ######## stage 2 ########
        x = self.TE_2(x)                # (N_rays, N_s, N_src, 40)
        if cam_optim:
            x, delta = self.feature_update(x, feature_maps, input_tensor, tgt_cam, srcs_cam, pts, stage="second")
            optimized_cams.append(srcs_cam.clone())
            deltas.append(delta)
            x = self.layer_norm_2(x)
        x = self.TE_2_2(x)              # (N_rays, N_s, N_src, 40)
        if cam_optim:
            x, delta = self.feature_update(x, feature_maps, input_tensor, tgt_cam, srcs_cam, pts, stage="second")
            optimized_cams.append(srcs_cam.clone())
            deltas.append(delta)
            x = self.layer_norm_2(x)
        x = self.TE_2_3(x)              # (N_rays, N_s, N_src, 40)
        ########################
        x = self.dim_linear_2(x)        # (N_rays, N_s+1, N_src, 20)

        # CLS 토큰 제거
        x = x[:, 1:, :, :]              # (N_rays, N_s, N_src, 20)

        # weighted sum along dim 1
        weight = self.weight_layer(x)       # (N_rays, N_s, N_src, 1)
        per_point_features = torch.sum(weight*x, dim=-2)      # (N_rays, N_s, 20)

        # Color function
        ray_colors = self.c_head(per_point_features) # (N_rays, N_s, 3)
        # Opacity function
        ray_densities = self.f_head(per_point_features) # (N_rays, N_s, 1)

        if cam_optim:
            return ray_densities, ray_colors, deltas, optimized_cams
        else:
            return ray_densities, ray_colors, None, None


# (N_s, N_src, D_z) -> (N_s, N_src, D_z)
class TransformerEncoder(nn.Module):
    def __init__(self, along_dim, feature_dim, num_heads):
        super(TransformerEncoder, self).__init__()

        self.along_dim = along_dim
        # Multi-head attention along dim
        # num_heads = 8 (Transformer 논문에서의 세팅)
        self.multi_head_att = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.Q_weights = nn.Linear(feature_dim, feature_dim)
        self.K_weights = nn.Linear(feature_dim, feature_dim)
        self.V_weights = nn.Linear(feature_dim, feature_dim)
        
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)

        self.layer_norm_1 = nn.LayerNorm(feature_dim)
        self.layer_norm_2 = nn.LayerNorm(feature_dim)

        self.two_layer_MLP = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        

    def forward(self, input_tensor):
        # input_tensor = Z

        # Multi Head Att = MHA(Z, dim=dim)

        # MultiHead(Q,K,V)
        # Q: (sequence length, batch, embedding)
        # K: (sequence length, batch, embedding)
        # V: (sequence length, batch, embedding)

        # Pooling transformer enc
        if self.along_dim == "src":
            # 배치로 들어오는 각 샘플들에 대해, N_src개 소스뷰 시퀀스를 입력으로 줌.
            # (Seq_len, Batch, Features) = (N_src, N_rays*N_s, D_z)
            input_tensor = input_tensor.permute(2, 0, 1, 3)
            shape = input_tensor.shape

            # Pooling transformer의 Batch에 해당하는
            # `N_rays` 차원과 `N_s` 차원을 합쳐준다.
            input_tensor = input_tensor.reshape(shape[0], shape[1]*shape[2], shape[3])

        # Ray transformer enc
        else:
            # 배치로 들어오는 각 소스뷰에 대해, N_s개 샘플 시퀀스를 입력으로 줌.
            # (Seq_len, Batch, Features) = (N_s, N_rays*N_src, D_Z) 
            input_tensor = input_tensor.permute(1, 0, 2, 3)
            shape = input_tensor.shape

            # Ray transformer의 Batch에 해당하는
            # `N_rays` 차원과 `N_src` 차원을 합쳐준다.
            input_tensor = input_tensor.reshape(shape[0], shape[1]*shape[2], shape[3])
        
        query = self.Q_weights(input_tensor)
        key = self.K_weights(input_tensor)
        value = self.V_weights(input_tensor)

        x, _ = self.multi_head_att(query, key, value)
        # Sub-layer MLP
        x_skip = self.layer_norm_1(input_tensor + self.dropout_1(x))    # Skip + LayerNorm  = Z'
        x = self.two_layer_MLP(x_skip)                                  # Two-Layer MLP = MLP(Z')
        x = self.layer_norm_2(x_skip + self.dropout_2(x))               # Skip + LayerNorm = TE^dim(Z)

        x = x.reshape(shape[0], shape[1], shape[2], shape[3])   # N_rays 차원을 분리
        if self.along_dim == "src":
            x = x.permute(1, 2, 0, 3)           # 원래 차원 순서인 (N_rays, N_s, N_src, D_z)로 변환
        else:
            x = x.permute(1, 0, 2, 3)           # 원래 차원 순서인 (N_rays, N_s, N_src, D_z)로 변환

        return x        # shape: (N_rays, N_s, N_src, c_out)