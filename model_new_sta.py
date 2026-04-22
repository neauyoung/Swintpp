"""Swin Transformer with Fold-back + Spectral-Temporal Attention + Global Gate

在尽量保持原始代码结构不变的前提下，只做如下最小必要修改：
1. 新增 SpectralTemporalAttention 模块；
2. 在 SwinTransformerBlockFoldBack 中加入该注意力；
3. 保留原有 PatchEmbed / PatchMerging / Local-Global Fusion / Head 接口；
4. 仅修正与新增注意力存在冲突或明显不稳妥的地方。

说明：
- 输入/输出接口保持不变；
- forward 仍返回分类 logits；
- BasicLayer / SwinTransformer / 工厂函数名称保持不变，便于直接替换原 model_new.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


# ------------------------- DropPath -------------------------
def drop_path_f(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
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
    """x: [B, H, W, C] -> windows: [B*num_windows, window_size, window_size, C]"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """windows: [B*num_windows, window_size, window_size, C] -> x: [B, H, W, C]"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, H, W


class PatchMergingHighDim(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        self.norm = norm_layer(4 * in_dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, (H + 1) // 2, (W + 1) // 2


class MLPHighDim(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.0):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.channel_expand = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.transpose(1, 2)          # [B, C, L]
        x = self.channel_expand(x)
        x = self.act(x)
        x = x.transpose(1, 2)          # [B, L, C]
        x = self.fc(x)
        x = self.drop(x)
        return x


# ------------------------- Spectral-Temporal Attention -------------------------
class SpectralTemporalAttention(nn.Module):
    """Channel-wise spectral-temporal attention.

    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C, H, W]

    设计思路（尽量贴近论文思想，但保持实现简洁稳定）:
    - temporal attention: 对时间维 W 建模，得到 [B, C, 1, W]
    - spectral attention: 对频率维 H 建模，得到 [B, C, H, 1]
    - 外积形成 [B, C, H, W] 的二维注意力图
    - 使用缩放因子，避免注意力值过小导致梯度不稳定
    """

    def __init__(self, dim, attn_scale=2.0):
        super().__init__()
        self.dim = dim
        self.attn_scale = attn_scale

        # 轻量投影，尽量不破坏原结构
        self.temp_proj = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
        self.spec_proj = nn.Conv1d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # temporal attention: 在频率维聚合后，对时间维做 softmax
        x_temp = x.mean(dim=2)                         # [B, C, W]
        a_temp = self.temp_proj(x_temp)               # [B, C, W]
        a_temp = torch.softmax(a_temp, dim=-1).unsqueeze(2)   # [B, C, 1, W]

        # spectral attention: 在时间维聚合后，对频率维做 softmax
        x_spec = x.mean(dim=3)                        # [B, C, H]
        a_spec = self.spec_proj(x_spec)               # [B, C, H]
        a_spec = torch.softmax(a_spec, dim=-1).unsqueeze(3)   # [B, C, H, 1]

        # outer-product style 组合
        attn = a_spec * a_temp                        # [B, C, H, W]
        attn = attn * self.attn_scale

        out = x * attn
        return out


# ------------------------- WindowAttention -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 这里保留 mask 接口，当前最小改动实现中不额外引入 SW-MSA mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ------------------------- SwinTransformerBlockFoldBack -------------------------
class SwinTransformerBlockFoldBack(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop
        )

        self.mlp = MLPHighDim(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 新增：频谱-时间注意力
        self.st_attention = SpectralTemporalAttention(dim)

        # 原有 local branch 保留
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        )

        # 原有 global gate 保留
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x, H, W):
        # 兼容可能错误传 tuple 的情况
        if isinstance(x, tuple):
            x = x[0]

        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size: L={L}, H*W={H*W}"

        shortcut = x

        # LN 后恢复二维空间
        x = self.norm1(x).view(B, H, W, C)
        x_conv = x.clone()

        # window attention
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x_pad = F.pad(x, (0, 0, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, _ = x_pad.shape

        x_windows = window_partition(x_pad, self.window_size)   # [nW*B, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        # [B, H, W, C] -> [B, C, H, W]
        x_attn = x.permute(0, 3, 1, 2).contiguous()
        x_conv = x_conv.permute(0, 3, 1, 2).contiguous()

        # local branch
        local_detail = self.local_conv(x_conv)

        # global gate branch
        global_gate = self.global_gate(x_conv).view(B, C, 1, 1)

        # 原融合
        x_fused = x_attn + local_detail + x_conv * global_gate

        # 新增：频谱-时间注意力，对融合特征进行重标定
        x_fused = self.st_attention(x_fused)

        # 最终线性融合
        x_fused = self.fuse_conv(x_fused)

        # 回到 token 形式
        x = x_fused.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        # 残差 + MLP
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


# ------------------------- BasicLayer -------------------------
class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        if isinstance(drop_path, list):
            drop_path_list = drop_path
        else:
            drop_path_list = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlockFoldBack(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_list[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample = downsample(in_dim=dim, out_dim=2 * dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x, H, W = checkpoint.checkpoint(lambda inp, h, w, blk=blk: blk(inp, h, w), x, H, W)
            else:
                x, H, W = blk(x, H, W)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


# ------------------------- SwinTransformer -------------------------
class SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=1,
        num_classes=3,
        embed_dim=96,
        high_dim=144,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # 保持原 patch_embed 设计：high_dim=embed_dim，不额外改动通道体系
        self.patch_embed = PatchEmbedHighDim(
            in_channels=in_chans,
            embed_dim=embed_dim,
            high_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMergingHighDim if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))   # [B, C, 1]
        x = torch.flatten(x, 1)               # [B, C]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    """保持原函数名，方便外部训练脚本直接调用。

    注意：
    - 这里仍默认 in_chans=3，以兼容常见 ImageNet 预训练权重。
    - 若你的肺音谱图是单通道，可在外部调用时传入 in_chans=1。
      例如：create_model(num_classes=3, in_chans=1)
    """
    model = SwinTransformer(
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        num_classes=num_classes,
        **kwargs
    )
    return model
