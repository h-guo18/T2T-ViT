# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath
from .efficient_attention import AttentionFactory

# def get_EF(input_size, dim, method="no_params", head_dim=None, bias=True):
#     """
#     Retuns the E or F matrix, initialized via xavier initialization.
#     This is the recommended way to do it according to the authors of the paper.
#     Includes a method for convolution, as well as a method for no additional params.
#     """
#     assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
#     if method == "convolution":
#         conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
#         return conv
#     if method == "no_params":
#         mat = torch.zeros((input_size, dim))
#         torch.nn.init.normal_(mat, mean=0.0, std=1/dim)
#         return mat
#     lin = nn.Linear(input_size, dim, bias)
#     torch.nn.init.xavier_normal_(lin.weight)
#     return lin

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    #projection_matrix : (h,m,d_head)
    # projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    # projection = projection.type_as(data)

    # data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_dash = torch.einsum('bhni,hmi->bhnm',data_normalizer*data, projection_matrix)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 linformer=False,kernel_method=None,kernel_ratio=0.5,input_size=197):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.epsilon = 1e-8  # for stable in division
        self.kernel=kernel_method
        self.linformer=linformer

        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.linformer:
            input_size= input_size
            linformer_dim = 64
            
            EF_proj = torch.zeros((2,input_size, linformer_dim))
            torch.nn.init.normal_(EF_proj, mean=0.0, std=1/linformer_dim)
            self.E_proj = nn.Linear(input_size, linformer_dim,bias=False)
            self.F_proj = nn.Linear(input_size, linformer_dim,bias=False)
            # self.post_proj_ln = nn.LayerNorm(self.head_dim)
            # self.E_proj = nn.Parameter(EF_proj[0],requires_grad=False)
            # self.F_proj = nn.Parameter(EF_proj[1],requires_grad=False) #(H,N,D)
            
        # if self.kernel is not None:
        #     self.m = int(self.head_dim * kernel_ratio)
        #     self.w = torch.randn(self.num_heads,self.m, self.head_dim)
        #     self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)
        # self.linformer_E = torch.randn()
        
    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, H, N, D)
        # w = (m, H*D)
        # return : x : B, H, N, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1,1, self.m) / 2
        wtx = torch.einsum('bhni,hmi->bhnm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)
    
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #(B,H,N,D)3
        if self.linformer:
            k=self.E_proj(torch.transpose(k,2,3))
            k = torch.transpose(k,2,3)
            # k = self.post_proj_ln(k)
            v=self.F_proj(torch.transpose(v,2,3))
            v=torch.transpose(v,2,3)
            # v = self.post_proj_ln(v)
            
        
        if self.kernel is not None:
            if self.kernel == 'softmax':
                kp, qp = self.prm_exp(k), self.prm_exp(q)
            elif self.kernel == 'relu':
                kp, qp =generalized_kernel(k,projection_matrix=None), generalized_kernel(q,projection_matrix=None)
            elif self.kernel == 'rnn':
                kp =generalized_kernel(k,projection_matrix=None,kernel_fn= lambda x: (nn.ELU()(x) + 1))
                qp = generalized_kernel(q,projection_matrix=None,kernel_fn= lambda x: (nn.ELU()(x) + 1))
            D = torch.einsum('bhni,bhi->bhn', qp, kp.sum(dim=2)).unsqueeze(dim=3)  # (B, H, N,m) * (B, H, m) -> (B, H,N, 1)
            # if self.linformer:
            #     k=self.E_proj(torch.transpose(k,2,3))
            #     k = torch.transpose(k,2,3)
            #     v=self.F_proj(torch.transpose(v,2,3))
            #     v=torch.transpose(v,2,3)
            kptv = torch.einsum('bhid,bhim->bhdm', v.float(), kp)  # (B, H, D,m)
            x = torch.einsum('bhni,bhdi->bhnd', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)  # (B, H, N, D)/Diag
            x = x.reshape(B,N,C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 input_size=197,eva=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if eva:
            attn_args = {
            # **vars(args.attn_specific_args),
            **{
            'dim': dim, 
            'num_heads': num_heads, 
            'qkv_bias': qkv_bias, 
            'attn_drop': attn_drop, 
            'proj_drop': 0.,
            }
        }
            self.attn = AttentionFactory.build_attention(attn_name = 'eva', attn_args = attn_args)
        else:
            self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,input_size=input_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
