import torch
import torch.nn as nn
import os


class NerFormerArchitecture(nn.Module):
    def __init__(self, d_z):
        super(NerFormerArchitecture, self).__init__()

        self.d_z = d_z  # Input feature의 차원

        # input: (N_rays(=Batch), N_s, N_src, d_z)
        self.linear_1 = nn.Linear(d_z, 80, bias=False)
        
        # (N_rays, N_s, N_src, 80)
        self.TE_1 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        self.TE_1_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        self.TE_1_3 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        self.dim_linear_1 = nn.Linear(80, 40)
        # (N_rays, N_s, N_src, 40)
        self.TE_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=4),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=4)           # Ray transformer encoder
        )
        self.TE_2_2 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=8)           # Ray transformer encoder
        )
        self.TE_2_3 = nn.Sequential(
            TransformerEncoder(along_dim="src", feature_dim=40, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim="sample", feature_dim=40, num_heads=8)           # Ray transformer encoder
        )
        self.dim_linear_2 = nn.Linear(40, 20)
        # (N_rays, N_s, N_src, 20)

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


    def forward(self, input_tensor):
        # input_tensor: (N_rays(=Batch), N_s, N_src, D_z)

        x = self.linear_1(input_tensor)     # (N_rays, N_s, N_src, 80)

        x = self.TE_1(x)                    # (N_rays, N_s, N_src, 80)
        x = self.TE_1_2(x)                    # (N_rays, N_s, N_src, 80)
        x = self.TE_1_3(x)                    # (N_rays, N_s, N_src, 80)
        x = self.dim_linear_1(x)              # (N_rays, N_s, N_src, 40)

        x = self.TE_2(x)                    # (N_rays, N_s, N_src, 40)
        x = self.TE_2_2(x)                    # (N_rays, N_s, N_src, 40)
        x = self.TE_2_3(x)                    # (N_rays, N_s, N_src, 40)
        x = self.dim_linear_2(x)              # (N_rays, N_s, N_src, 20)
        
        # weighted sum along dim 1
        weight = self.weight_layer(x)       # (N_rays, N_s, N_src, 1)
        per_point_features = torch.sum(weight*x, dim=-2)      # (N_rays, N_s, 20)

        # Color function
        ray_colors = self.c_head(per_point_features) # (N_rays, N_s, 3)
        # Opacity function
        ray_densities = self.f_head(per_point_features) # (N_rays, N_s, 1)

        return ray_densities, ray_colors


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