import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=96, patch_size=4, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.high_proj = nn.Sequential(
            nn.Conv2d(embed_dim, 144, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(144, embed_dim, kernel_size=1, bias=False)
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x + self.high_proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)

        return x, (H + 1) // 2, (W + 1) // 2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):  # 调整drop为0.1
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlockFoldBack(nn.Module):
    """
    改进版 FoldBack Block：
    不再只使用 attention 输出 x_attn 进行 fold-back，
    而是同时利用 attention 前的原始 token 特征 x_raw 进行二维局部/全局建模。
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.1, drop_path=0.05):  # 调整drop/drop_path
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=7, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, H, W):
        shortcut = x
        B, N, C = x.shape

        # 1. Attention 分支：token 空间建模
        x_attn = self.attn(self.norm1(x))

        # 2. Attention 输出 fold-back 到二维空间
        x_attn_2d = x_attn.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 3. 原始 token 分支：attention 前的 x 也 fold-back 到二维空间
        x_raw_2d = shortcut.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 4. 基于原始二维特征做局部建模
        local_detail = self.local_conv(x_raw_2d)

        # 5. 基于原始二维特征做全局门控
        gate = self.global_gate(x_raw_2d)
        gated = x_raw_2d * gate

        # 6. 并行融合：attention二维特征 + 原始二维局部特征 + 原始二维全局门控特征
        x_fused = x_attn_2d + local_detail + gated

        # 7. 回到 token 空间
        x_fused = x_fused.permute(0, 2, 3, 1).contiguous().view(B, N, C)

        # 8. 残差 + MLP
        x = shortcut + self.drop_path(x_fused)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.1, drop_path=0.05):  # 调整drop/drop_path
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=7, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, H, W


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, downsample=None, use_foldback=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlockFoldBack(dim, num_heads) if use_foldback else SwinTransformerBlock(dim, num_heads)
            for _ in range(depth)
        ])
        self.downsample = downsample(dim) if downsample else None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x, H, W = blk(x, H, W)

        if self.downsample:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        use_foldback=False,
        pos_drop=0.0  # 可根据需求调整，这里保持0.0
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(in_c=in_chans, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        self.pos_drop = nn.Dropout(pos_drop)  # 位置编码dropout

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim=dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                use_foldback=use_foldback
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def swin_tiny_patch4_window7_224(num_classes=3, use_foldback=True):
    return SwinTransformer(
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        use_foldback=use_foldback,
        pos_drop=0.0
    )