import torch
import torch.nn as nn
import os


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)

        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NerFormer(nn.Module):
    def __init__(self, d_z=80):
        super(NerFormer, self).__init__()

        self.d_z = d_z  # Input feature의 차원

        # input: (N_s, N_src, d_z)
        self.linear_1 = nn.Linear(d_z, 80, bias=False)
        
        # (N_s, N_src, 80)
        self.TE_1 = nn.Sequential(
            TransformerEncoder(along_dim=1, feature_dim=80, num_heads=8),          # Pooling transformer encoder
            TransformerEncoder(along_dim=0, feature_dim=80, num_heads=8)           # Ray transformer encoder
        )
        self.dim_linear_1 = nn.Linear(80, 40)
        # (N_s, N_src, 40)
        self.TE_2 = nn.Sequential(
            TransformerEncoder(along_dim=1, feature_dim=40, num_heads=4),          # Pooling transformer encoder
            TransformerEncoder(along_dim=0, feature_dim=40, num_heads=4)           # Ray transformer encoder
        )
        self.dim_linear_2 = nn.Linear(80, 40)
        # (N_s, N_src, 20)

        self.weight_layer = nn.Sequential(
            nn.Linear(20, 1),
            nn.Softmax()
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
        # input_tensor: (N_s, N_src, D_z)

        x = self.linear_1(input_tensor)     # (N_s, N_src, 80)

        x = self.TE_1(x)                    # (N_s, N_src, 80)
        x = self.dim_linear_1(x)              # (N_s, N_src, 40)

        x = self.TE_2(x)                    # (N_s, N_src, 40)
        x = self.dim_linear_2(x)              # (N_s, N_src, 20)
        
        # weighted sum along dim 1
        weight = self.weight_layer(x)       # (N_s, N_src, 1)
        per_point_features = torch.sum(weight*x, dim=1)      # (N_s, 20)

        # Color function
        ray_colors = self.c_head(per_point_features) # (N_s, 3)
        # Opacity function
        ray_densities = self.f_head(per_point_features) # (N_s, 1)

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

        ###### 1. along dim=1인 경우
        # 하나의 샘플에 대해, N_src개 소스뷰에서의 features 시퀀스를 입력으로 줌.
        # (Batch, Seq_len, Features) = (N_s, N_src, D_z)
        # 그대로 입력으로 들어감.

        ###### 2. along dim=0인 경우
        # 하나의 소스뷰에 대해, N_s개 샘플에서의 features 시퀀스를 입력으로 줌.
        # (Batch, Seq_len, Features) = (N_src, N_s, D_Z) 
        # N_s와 N_src의 차원을 바꿔주어야 한다.
        if self.along_dim == 0:
            input_tensor = input_tensor.transpose(0, 1).contiguous()
        
        query = self.Q_weights(input_tensor)
        key = self.K_weights(input_tensor)
        value = self.V_weights(input_tensor)

        x, _ = self.multi_head_att(query, key, value)
        # Sub-layer MLP
        x_skip = self.layer_norm_1(input_tensor + self.dropout_1(x))    # Skip + LayerNorm  = Z'
        x = self.two_layer_MLP(x_skip)                                  # Two-Layer MLP = MLP(Z')
        x = self.layer_norm_2(x_skip + self.dropout_2(x))               # Skip + LayerNorm = TE^dim(Z)

        if self.along_dim == 0:
            x = x.transpose(0, 1).contiguous()

        return x        # shape: (N_s, N_src, c_out)