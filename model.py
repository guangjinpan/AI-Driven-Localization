import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from random import randrange
from torchvision.models import resnet34
import math
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)    

def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoidal Position Encoding: (1, n_position, d_hid)"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def cosine_similarity_loss(H1, H2, eps = 1e-1 ):


    H1r = H1[:,0,:,:] 
    H1i = H1[:,1,:,:]  
    H2r = H2[:,0,:,:]  
    H2i = H2[:,1,:,:] 
    B, M, N = H1r.shape
    # 1) reshape to [B, M*N]
    xi = H1r.reshape(B,-1)  # x real
    xj = H1i.reshape(B,-1)  # x imag
    yi = H2r.reshape(B,-1)  # y real
    yj = H2i.reshape(B,-1)  # y imag

    # 2) calculate dot product
    #    xy_dot_r = Σ(xr * yr + xi * yi)
    #    xy_dot_i = Σ(xr * yi - xi * yr)
    xy_dot_r = torch.sum(xi * yi + xj * yj, dim=1)
    xy_dot_i = torch.sum(xi * yj - xj * yi, dim=1)

    # 3) calculate ||x|| and ||y||
    norm_x = torch.sqrt(torch.sum(xi**2, dim=1) + torch.sum(xj**2, dim=1))
    norm_y = torch.sqrt(torch.sum(yi**2, dim=1) + torch.sum(yj**2, dim=1))

    # 5) calculate sqrt( real^2 + imag^2 ) / (||x|| * ||y||)
    dot_mod = torch.sqrt(xy_dot_r**2 + xy_dot_i**2)
    cos_loss = 1- dot_mod / (norm_x * norm_y + eps)

    return torch.mean(cos_loss)

class FCNN_Model(nn.Module):
    """
    MLP model for input [B, M] and output [B, 2]
    """
    def __init__(self, input_dim, hidden_dims=[1024, 1024, 256], output_dim=2):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN_ResNet32_Model(nn.Module):
    """
    input  : [B, M, 128, 32] 
    output  : [B, 2]
    """
    def __init__(self, input_feature_dim = 2):
        super().__init__()

        self.net  = resnet34()    # torchvision>=0.13

        self.net.conv1 = nn.Conv2d(input_feature_dim, 64,
                                   kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.net.conv1.weight, mode="fan_out", nonlinearity="relu")


        # MLP layer
        self.net.fc = nn.Linear(self.net.fc.in_features, 2)

    def forward(self, x):
        return self.net(x)


class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_size=512, num_layers=4, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.3
        )

        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, 2)

    def forward(self, x):
        B, C, M, N = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()   # [B, N, 2, M]
        x = x.view(B, N, C * M)                  # [B, N, 2×M]

        lstm_out, _ = self.lstm(x)               # [B, N, H]
        pooled = lstm_out.mean(dim=1)            # mean pooling
        return self.fc(pooled)                   # [B, 2]


# class LSTM_Model(nn.Module):
#     def __init__(self, input_dim, input_channels = 2, embed_dim=16, hidden_size=256, num_layers=4, bidirectional=False):
#         super().__init__()

#         # CNN：不改变空间维度 M × N
#         self.cnn = nn.Sequential(
#             nn.Conv2d(input_channels, embed_dim, kernel_size=3, stride=1, padding=1),  # -> [B, D, M, N]
#             nn.ReLU(),
#         )
#         input_dim = 64
#         self.embed_dim = embed_dim
#         direction_factor = 2 if bidirectional else 1

#         self.lstm = nn.LSTM(
#             input_size=embed_dim * input_dim,   # 每个时间步的特征维度
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=bidirectional,
#             dropout=0.3
#         )

#         self.fc = nn.Linear(hidden_size * direction_factor, 2)

#     def forward(self, x):
#         B, C, M, N = x.shape                   # x: [B, C, M, N]
#         x = self.cnn(x)                        # -> [B, D, M, N]
#         x = x.permute(0, 3, 1, 2).contiguous()   # [B, N, 2, M]
#         x = x.view(B, N, self.embed_dim * M)                  # [B, N, 2×M]
#         lstm_out, _ = self.lstm(x)             # -> [B, T, H]
#         pooled = lstm_out.mean(dim=1)          # mean pooling over T
#         return self.fc(pooled)                 # -> [B, 2]
 

def pad_to_multiple_of_4(x):
    # x: [B, C, M, N]
    B, C, H, W = x.shape
    target_H = math.ceil(H / 4) * 4
    target_W = math.ceil(W / 4) * 4

    pad_H = target_H - H
    pad_W = target_W - W

    # pad: (left, right, top, bottom)
    padding = (0, pad_W, 0, pad_H)
    x = F.pad(x, padding, mode='constant', value=0)
    return x   



# class Transformer_Model(nn.Module):
#     """
#     Input:  [B, 2, M, N]
#     Output: [B, 2]  (use [CLS] token for position prediction)
#     """
#     def __init__(self, input_channels=2, M=32, N=64, patch_embed_dim=512, num_layers=6, nhead=8, dim_feedforward=512):
#         super().__init__()
        
#         # CNN Patch Extractor


#         # self.cnn = nn.Sequential(
#         #     nn.Conv2d(input_channels, patch_embed_dim, kernel_size=7, stride=8, padding=3),  # -> [B, patch_embed_dim, M/8, N/8]
#         #     nn.ReLU(),
#         # )

#         self.patch_num = N
#         self.fc_embd = nn.Linear(input_channels*M, patch_embed_dim)

#         # Positional Embedding + CLS token

        
#         self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
#         self.pos_embedding = get_sinusoid_encoding(self.patch_num + 1, patch_embed_dim)
#         self.pos_embedding = nn.Parameter(self.pos_embedding, requires_grad=False)
        

#         # Transformer Encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=patch_embed_dim,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output projection
#         self.fc = nn.Linear(patch_embed_dim, 2)

#     def forward(self, x):

#         B, C, M, N = x.shape
#         x = x.permute(0, 3, 1, 2).contiguous()   # [B, N, 2, M]
#         x = x.view(B, N, C * M)                  # [B, N, 2×M]
#         x = self.fc_embd(x)
        

#         # CLS token
#         cls_token = self.cls_token.expand(B, -1, -1)     # [B, 1, 256]
#         x = torch.cat([cls_token, x], dim=1)             # [B, patch_num + 1, 256]
#         x = x + self.pos_embedding[:, :x.size(1), :]     # 加上位置编码

#         x = self.transformer(x)                          # [B, patch_num + 1, 256]
#         # cls_output = x[:, 0, :]                          # [B, 256]
#         cls_output = x.mean(dim=1)            # mean pooling
#         out = self.fc(cls_output)                        # [B, 2]
#         return out

class Transformer_Model(nn.Module):
    """
    Input:  [B, 2, M, N]
    Output: [B, 2]  (use [CLS] token for position prediction)
    """
    def __init__(self, input_channels=2, M=32, N=64, patch_embed_dim=512, num_layers=4, nhead=16, dim_feedforward=2048):
        super().__init__()
        
        # CNN Patch Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # -> [B, 64, M/2, N/2]
            nn.ReLU(),
            nn.Conv2d(64, patch_embed_dim, kernel_size=3, stride=2, padding=1),              # -> [B, 128, M/4, N/4]
            # nn.ReLU(),
            # nn.Conv2d(128, patch_embed_dim, kernel_size=3, stride=2, padding=1),# -> [B, 256, M/8, N/8]
            # nn.ReLU(),
        )

        # self.cnn = nn.Sequential(
        #     nn.Conv2d(input_channels, patch_embed_dim, kernel_size=7, stride=8, padding=3),  # -> [B, patch_embed_dim, M/8, N/8]
        #     nn.ReLU(),
        # )

        self.M_p = math.ceil(M / 4)
        self.N_p = math.ceil(N / 4)
        self.patch_num = self.M_p * self.N_p  # ~ M*N/64
        print(self.patch_num, self.M_p, self.N_p )

        # Positional Embedding + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_embed_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_num + 1, patch_embed_dim))
        self.pos_embedding = get_sinusoid_encoding(self.patch_num + 1, patch_embed_dim)
        self.pos_embedding = nn.Parameter(self.pos_embedding, requires_grad=False)


        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout = 0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.fc = nn.Sequential(
                nn.Linear(patch_embed_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 2),            # 每个位置输出一个 logit
            )

    def forward(self, x):
        B = x.shape[0]
        x = pad_to_multiple_of_4(x)
        x = self.cnn(x)                            # [B, 256, M', N']
        x = x.flatten(2).transpose(1, 2)           # → [B, patch_num, 256]

        # CLS token
        cls_token = self.cls_token.expand(B, -1, -1)     # [B, 1, 256]
        x = torch.cat([cls_token, x], dim=1)             # [B, patch_num + 1, 256]
        x = x + self.pos_embedding[:, :x.size(1), :]     # 加上位置编码

        x = self.transformer(x)                          # [B, patch_num + 1, 256]
        cls_output = x[:, 0, :]                          # [B, 256]
        out = self.fc(cls_output)                        # [B, 2]
        return out
    
# class Transformer_Model(nn.Module):
#     """
#     Input:  [B, 2, M, N]
#     Output: [B, 2]  (use [CLS] token for position prediction)
#     """
#     def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=256):
#         super().__init__()
#         self.num_tokens = input_dim
#         self.d_model = d_model

#         # Token embedding: project 2 → d_model
#         self.token_embed = nn.Linear(2, d_model)

#         # Learnable [CLS] token
#         self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)

#         # Learnable positional embedding: for M*N + 1 (cls)
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens + 1, d_model))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # Output head
#         self.fc = nn.Linear(d_model, 2)

#     def forward(self, x):
#         B, C, M, N = x.shape  # [B, 2, M, N]
#         x = x.permute(0, 2, 3, 1).contiguous()  # [B, M, N, 2]
#         x = x.view(B, self.num_tokens, C)       # [B, M*N, 2]

#         # Token embedding
#         x = self.token_embed(x)  # [B, M*N, d_model]

#         # Expand cls token for batch
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
#         x = torch.cat((cls_tokens, x), dim=1)          # [B, M*N+1, d_model]

#         # Add positional encoding
#         x = x + self.pos_embedding                     # [B, M*N+1, d_model]

#         # Transformer encoding
#         x = self.transformer(x)                         # [B, M*N+1, d_model]

#         # Use CLS token output (at index 0)
#         cls_output = x[:, 0, :]                         # [B, d_model]
#         out = self.fc(cls_output)                       # [B, 2]
#         return out




def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoidal Position Encoding: (1, n_position, d_hid)"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)





class wireless_loc_model(nn.Module):

    def __init__(
        self,
        EnvPara = None,
    ):
        super().__init__()
        
        self.EnvPara = EnvPara
        self.device = EnvPara["device"]


        self.MSELoss = nn.MSELoss()
        if (self.EnvPara["model"] == "FCNN"):
            self.FCNN = FCNN_Model(input_dim = 2 * EnvPara["input_subcarrier_dim"] * EnvPara["input_antenna_dim"])
        elif (self.EnvPara["model"] == "ResNet32"):
            self.ResNet32 = CNN_ResNet32_Model()
        elif (self.EnvPara["model"] == "LSTM"):
            self.LSTM = LSTM_Model(input_dim = 2 * EnvPara["input_antenna_dim"])
        elif (self.EnvPara["model"] == "Transformer"):
            self.Transformer = Transformer_Model(input_channels = 2, M=EnvPara["input_antenna_dim"], N=EnvPara["input_subcarrier_dim"]  )





    def ResNet32_forward(self, channel, UElocation):
        
        B = channel.shape[0]
        x = self.ResNet32(channel)
        # print(x.shape,channel.shape,UElocation.shape)
        loss = self.MSELoss(x, UElocation)

        return x, loss
    

    def FCNN_forward(self, channel, UElocation):
        
        B = channel.shape[0]
        channel = channel.view((B,-1))
        x = self.FCNN(channel)
        # print(x.shape,channel.shape,UElocation.shape)
        loss = self.MSELoss(x, UElocation)
        
        return x, loss
    
    def LSTM_forward(self, channel, UElocation):
        
        B = channel.shape[0]

        x = self.LSTM(channel)
        # print(x.shape,channel.shape,UElocation.shape)
        loss = self.MSELoss(x, UElocation)
        
        return x, loss
    

    def Transformer_forward(self, channel, UElocation):
        
        B = channel.shape[0]

        x = self.Transformer(channel)
        # print(x.shape,channel.shape,UElocation.shape)
        loss = self.MSELoss(x, UElocation)
        
        return x, loss
    
    
   



    def forward(self, channel_Antenna_subcarrier, UElocation):
        """
        x: (B, input_fmap, input_fdim, input_tdim)
        task: 
            - "SingleBSLoc", "inference_SingleBSLoc"
            - "MultiBSLoc", "inference_MultiBSLoc"
            - "SingleBSLoc", "inference_SingleBSLoc"
            - "toa", "inference_toa"
            - "aoa", "inference_aoa"
            - "pretrain_mix" 
        """
        #x, y_position, BSconf = None, x_aug = None, y_position_aug = None, BSconf_aug = None
        # x = channel_data, UElocation, BSconf
        # x = x.transpose(2, 3)  # [B, in_chans, H, W]
        # if x_aug !=None:
        #     x_aug = x_aug.transpose(2, 3)
        # torch.autograd.set_detect_anomaly(True)

        # channel_Antenna_subcarrier = channel_Antenna_subcarrier.transpose(3, 4)
        # channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.transpose(3, 4)
        # channel_delay_angle_ri = channel_delay_angle_ri.transpose(3, 4)
        # distance = UElocation_all[:,:,2]
        # angle = UElocation_all[:,:,3]
        # UElocation_all = UElocation_all[:,:,:2]
        if self.EnvPara["model"] == "FCNN":
            pred, mse = self.FCNN_forward(channel_Antenna_subcarrier, UElocation)
            return pred,mse
        elif self.EnvPara["model"] == "ResNet32":
            pred, mse = self.ResNet32_forward(channel_Antenna_subcarrier, UElocation)
            return pred,mse
        elif self.EnvPara["model"] == "LSTM":
            pred, mse = self.LSTM_forward(channel_Antenna_subcarrier, UElocation)
            return pred,mse
        elif self.EnvPara["model"] == "Transformer":
            pred, mse = self.Transformer_forward(channel_Antenna_subcarrier, UElocation)
            return pred,mse
       

