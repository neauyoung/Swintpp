""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional



# ------------------------- DropPath -------------------------
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

# ------------------------- window utils -------------------------
def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# ------------------------- 高维通道投影模块 -------------------------
class PatchEmbedHighDim(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, high_dim=144, patch_size=4, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj_high = nn.Conv2d(embed_dim, high_dim, kernel_size=1)
        self.norm = norm_layer(high_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.proj_high(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class PatchMergingHighDim(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = norm_layer(4 * in_dim)
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B,H,W,C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0,0,0,W%2,0,H%2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0,x1,x2,x3], -1)
        x = x.view(B,-1,4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (H+1)//2, (W+1)//2

class MLPHighDim(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.channel_expand = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.channel_expand(x)
        x = self.act(x)
        x = x.transpose(1,2)
        x = self.fc(x)
        x = self.drop(x)
        return x

# ------------------------- WindowAttention -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B_,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ------------------------- SwinTransformerBlockFoldBack -------------------------
class SwinTransformerBlockFoldBack(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=2.0, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = MLPHighDim(dim, hidden_dim=int(dim*mlp_ratio), drop=drop)
        self.attn = WindowAttention(dim, window_size=(window_size,window_size), num_heads=num_heads)
        self.drop_path = nn.Identity() if drop_path==0. else nn.Dropout(drop_path)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    def forward(self, x, H, W):
        if isinstance(x, tuple):
            x = x[0]
        B,L,C = x.shape
        shortcut = x
        x = self.norm1(x).view(B,H,W,C)
        x_conv = x.clone()
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_pad = F.pad(x, (0,0,pad_t,pad_b,pad_l,pad_r))
        _, Hp, Wp, _ = x_pad.shape
        x_windows = window_partition(x_pad, self.window_size).view(-1,self.window_size*self.window_size,C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1,self.window_size,self.window_size,C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if pad_r>0 or pad_b>0:
            x = x[:,:H,:W,:].contiguous()
        x_attn = x.permute(0,3,1,2)
        x_conv = x_conv.permute(0,3,1,2)
        local_detail = self.local_conv(x_conv)
        global_gate = self.global_gate(x_conv).view(B,C,1,1)
        x = x_attn + local_detail + x_conv * global_gate
        x = self.fuse_conv(x)
        x = x.permute(0,2,3,1).view(B,H*W,C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,H,W

# ------------------------- BasicLayer -------------------------
class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=2.0, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlockFoldBack(dim, num_heads, window_size, mlp_ratio, drop, attn_drop, drop_path[i] if isinstance(drop_path,list) else drop_path, norm_layer)
            for i in range(depth)
        ])
        self.downsample = downsample(in_dim=dim, out_dim=2*dim, norm_layer=norm_layer) if downsample else None
    def forward(self, x,H,W):
        for blk in self.blocks:
            x,H,W = blk(x,H,W)
        if self.downsample is not None:
            x,H,W = self.downsample(x,H,W)
        return x,H,W

# ------------------------- SwinTransformer -------------------------
class SwinTransformer(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, num_classes=3,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=7, mlp_ratio=2.0, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim*2**(self.num_layers-1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbedHighDim(in_channels=in_chans, embed_dim=embed_dim, high_dim=embed_dim, patch_size=patch_size, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim*2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                downsample=PatchMergingHighDim if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
    def forward(self,x):
        x,H,W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x,H,W = layer(x,H,W)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1,2))
        x = torch.flatten(x,1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model

