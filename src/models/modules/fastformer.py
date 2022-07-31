import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertSelfOutput, BertIntermediate, BertOutput

# Fastformer의 핵심인 기존 self-attention의 단순화 버전
class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()

        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))

        self.attention_head_size = int(config.hidden_size /config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_dim= config.hidden_size
        
        self.query = nn.Linear(self.input_dim, self.all_head_size)
        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)

        self.key = nn.Linear(self.input_dim, self.all_head_size)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)

        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
                
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape    # (Batch, Seq, d)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        # batch_size, num_head, seq_len
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / self.attention_head_size**0.5
        # # add attention mask
        # query_for_score += attention_mask

        # batch_size, num_head, 1, seq_len
        query_weight = self.softmax(query_for_score).unsqueeze(2)

        # batch_size, num_head, seq_len, head_dim
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # batch_size, num_head, head_dim, 1
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1)
        # batch_size, num_head, seq_len, head_dim

        # batch_size, num_head, seq_len
        mixed_query_key_layer=mixed_key_layer* pooled_query_repeat
        
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        
        # # add attention mask
        # query_key_score +=attention_mask

        # batch_size, num_head, 1, seq_len
        query_key_weight = self.softmax(query_key_score).unsqueeze(2)

        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.transform(weighted_value) + mixed_query_layer
      
        return weighted_value


class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()

        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor):
        self_output = self.self(input_tensor)
        attention_output = self.output(self_output, input_tensor)

        return attention_output


class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class FastNerFormerArchitecture(nn.Module):
    def __init__(self, d_z, config):
        super(FastNerFormerArchitecture, self).__init__()

        self.d_z = d_z  # Input feature의 차원

        # input: (N_rays(=Batch), N_s, N_src, d_z)
        self.linear_1 = nn.Linear(d_z, 80, bias=False)
        
        # (N_rays, N_s, N_src, 80)
        self.TE_1_1 = nn.Sequential(
            
            FastformerEncoder(along_dim="src", config=config, step=0),          # Pooling transformer encoder
            FastformerEncoder(along_dim="sample", config=config, step=0)           # Ray transformer encoder
        )
        self.TE_1_2 = nn.Sequential(
            
            FastformerEncoder(along_dim="src", config=config, step=0),          # Pooling transformer encoder
            FastformerEncoder(along_dim="sample", config=config, step=0)           # Ray transformer encoder
        )
        self.dim_linear_1 = nn.Linear(80, 40)
        # (N_rays, N_s, N_src, 40)
        self.TE_2_1 = nn.Sequential(
            
            FastformerEncoder(along_dim="src", config=config, step=1),          # Pooling transformer encoder
            FastformerEncoder(along_dim="sample", config=config, step=1)           # Ray transformer encoder
        )
        self.TE_2_2 = nn.Sequential(
            
            FastformerEncoder(along_dim="src", config=config, step=1),          # Pooling transformer encoder
            FastformerEncoder(along_dim="sample", config=config, step=1)           # Ray transformer encoder
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

        x = self.TE_1_1(x)                    # (N_rays, N_s, N_src, 80)
        x = self.TE_1_2(x)
        x = self.dim_linear_1(x)              # (N_rays, N_s, N_src, 40)

        x = self.TE_2_1(x)                    # (N_rays, N_s, N_src, 40)
        x = self.TE_2_2(x)
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
class FastformerEncoder(nn.Module):
    def __init__(self, along_dim, config, step):
        super(FastformerEncoder, self).__init__()

        self.along_dim = along_dim
        config.hidden_size = config.hidden_size_per_step[step]
        config.intermediate_size = config.hidden_size_per_step[step]
        # Multi-head attention along dim
        # num_heads = 8 (Transformer 논문에서의 세팅)
        self.fastformer_layer = FastformerLayer(config=config)
        

    def forward(self, input_tensor):
        # Pooling transformer enc
        if self.along_dim == "src":
            # 배치로 들어오는 각 샘플들에 대해, N_src개 소스뷰 시퀀스를 입력으로 줌.
            # (Batch, Seq_len, Features) = (N_rays*N_s, N_src, D_z)
            shape = input_tensor.shape

            # Pooling transformer의 Batch에 해당하는
            # `N_rays` 차원과 `N_s` 차원을 합쳐준다.
            input_tensor = input_tensor.reshape(shape[0]*shape[1], shape[2], shape[3])

        # Ray transformer enc
        else:
            # 배치로 들어오는 각 소스뷰에 대해, N_s개 샘플 시퀀스를 입력으로 줌.
            # (Batch, Seq_len, Features) = (N_rays*N_src, N_s, D_Z) 
            input_tensor = input_tensor.permute(0, 2, 1, 3)
            shape = input_tensor.shape

            # Ray transformer의 Batch에 해당하는
            # `N_rays` 차원과 `N_src` 차원을 합쳐준다.
            input_tensor = input_tensor.reshape(shape[0]*shape[1], shape[2], shape[3])

        x = self.fastformer_layer(input_tensor)

        x = x.reshape(shape[0], shape[1], shape[2], shape[3])   # N_rays 차원을 분리
        
        # 원래 차원 순서인 (N_rays, N_s, N_src, D_z)로 변환
        if self.along_dim == "sample":
            x = x.permute(0, 2, 1, 3)

        return x        # shape: (N_rays, N_s, N_src, c_out)