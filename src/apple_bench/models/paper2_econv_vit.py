"""Paper 2 (Huang et al. 2025) reproduction: EConv-ViT (ConvNeXt + ViT fusion).

ECA attention is added to each ConvNeXt block; DropKey replaces attention-dropout
in the ViT branch. Reproduction notes documented in docs/reproduction-report.md.

Architecture:
  Conv branch : ConvNeXt-Small backbone (3/3/27/3 blocks at dims 96/192/384/768)
                with ECA appended to every block. Output: (B, 768, 7, 7).
  ViT branch  : Patch-embed 16x16 → 20 ViT blocks at dim 384 / 6 heads with
                DropKey. Output: cls-token (B, 384).
  Fusion      : concat [conv_pool, cls] → (B, 1152) → LayerNorm → Linear.
  Param count : ~85.3M (paper claims 83.7M; ~2% drift from reproduced counts).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from apple_bench.models import registry


# ---- ECA (Efficient Channel Attention) ----------------------------------
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)       # (B, 1, C)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)     # (B, C, 1, 1)
        return x * self.sigmoid(y)


# ---- E-ConvNeXt block ---------------------------------------------------
class EConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.eca = ECA(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)              # NCHW -> NHWC for LayerNorm + Linear
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)              # back to NCHW
        x = self.eca(x)
        return residual + x


class _ChannelLN(nn.Module):
    """LayerNorm over the channel dim only (NCHW input)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = _ChannelLN(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class EConvNeXtBranch(nn.Module):
    """ConvNeXt-Small with ECA: 4 stages (3/3/27/3 blocks at dims 96/192/384/768).

    Spatial progression for 224x224 input:
      stem  : stride-4  -> (B, 96, 56, 56)
      down12: stride-2  -> (B, 192, 28, 28)
      down23: stride-2  -> (B, 384, 14, 14)
      down34: stride-2  -> (B, 768, 7, 7)
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4),
            _ChannelLN(96),
        )
        self.stage1 = nn.Sequential(*[EConvNeXtBlock(96) for _ in range(3)])
        self.down12 = _Downsample(96, 192)
        self.stage2 = nn.Sequential(*[EConvNeXtBlock(192) for _ in range(3)])
        self.down23 = _Downsample(192, 384)
        self.stage3 = nn.Sequential(*[EConvNeXtBlock(384) for _ in range(27)])
        self.down34 = _Downsample(384, 768)
        self.stage4 = nn.Sequential(*[EConvNeXtBlock(768) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down12(x)
        x = self.stage2(x)
        x = self.down23(x)
        x = self.stage3(x)
        x = self.down34(x)
        return self.stage4(x)               # (B, 768, 7, 7)


# ---- DropKey attention --------------------------------------------------
class DropKeyAttention(nn.Module):
    """MHSA with attention-key dropout instead of post-softmax dropout."""

    def __init__(self, dim: int, num_heads: int, dropkey_p: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropkey_p = dropkey_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.training and self.dropkey_p > 0.0:
            mask = torch.empty_like(attn).bernoulli_(self.dropkey_p).bool()
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropkey_p: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DropKeyAttention(dim, num_heads, dropkey_p)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBranch(nn.Module):
    """Patch-embed 16x16 -> 20 ViT blocks at dim 384 / 6 heads (DeiT-Small depth).

    Token layout: [cls | 196 patch tokens], total 197 tokens.
    DropKey schedule decays linearly from 0.1 (block 0) to 0.0 (block 19).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 384, kernel_size=16, stride=16)  # 224/16 = 14 -> 196 tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 384))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        n_blocks = 20
        # DropKey schedule: linear from 0.1 (early) down to 0.0 (late).
        schedule = torch.linspace(0.1, 0.0, n_blocks).tolist()
        self.blocks = nn.ModuleList([ViTBlock(384, 6, dropkey_p=schedule[i]) for i in range(n_blocks)])
        self.norm = nn.LayerNorm(384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 196, 384)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed     # (B, 197, 384)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # class token, (B, 384)


# ---- Top-level fusion ---------------------------------------------------
class EConvViT(nn.Module):
    """Dual-branch model: EConvNeXt (768-d) + ViT (384-d) fused via concatenation.

    Concat dim = 768 + 384 = 1152.  Total params ~85.3M.
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv_branch = EConvNeXtBranch()
        self.conv_pool = nn.AdaptiveAvgPool2d(1)
        self.vit_branch = ViTBranch(num_classes)
        self.fusion_norm = nn.LayerNorm(1152)
        self.head = nn.Linear(1152, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv_branch(x)              # (B, 768, 7, 7)
        c = self.conv_pool(c).flatten(1)     # (B, 768)
        v = self.vit_branch(x)               # (B, 384)
        f = torch.cat([c, v], dim=1)         # (B, 1152)
        f = self.fusion_norm(f)
        return self.head(f)


# ---- Pretrained-weight loader ------------------------------------------
def _load_pretrained_into(model: EConvViT) -> tuple[int, int]:
    """Initialize ConvNeXt branch from torchvision ConvNeXt-Small (ImageNet)
    and ViT branch from timm DeiT-Small (ImageNet). New layers (ECA, last
    8 ViT blocks, fusion head) stay randomly initialized.

    The paper underspecifies pretraining; this is a documented deviation
    that materially affects convergence on small training sets.

    Returns: (n_loaded_keys, n_total_keys)
    """
    import timm  # noqa: PLC0415
    from torchvision.models import ConvNeXt_Small_Weights, convnext_small  # noqa: PLC0415

    # --- ConvNeXt-Small from torchvision ---
    cn = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    cn_sd = cn.state_dict()

    # Source-stage to (target-stage-name, n-blocks) — torchvision ConvNeXt-Small
    # stages live at features[1,3,5,7] with block counts (3, 3, 27, 3).
    cn_stage_map = [(1, "stage1", 3), (3, "stage2", 3),
                    (5, "stage3", 27), (7, "stage4", 3)]
    cn_down_map = [(2, "down12"), (4, "down23"), (6, "down34")]

    target_sd: dict[str, torch.Tensor] = {}

    # stem
    target_sd["conv_branch.stem.0.weight"] = cn_sd["features.0.0.weight"]
    target_sd["conv_branch.stem.0.bias"] = cn_sd["features.0.0.bias"]
    target_sd["conv_branch.stem.1.weight"] = cn_sd["features.0.1.weight"]
    target_sd["conv_branch.stem.1.bias"] = cn_sd["features.0.1.bias"]

    # stages: each block maps Sequential indices {0, 2, 3, 5} to {dwconv, norm, pwconv1, pwconv2}.
    for src_idx, tgt_name, n_blocks in cn_stage_map:
        for b in range(n_blocks):
            for src_sub, tgt_sub in [(0, "dwconv"), (2, "norm"),
                                       (3, "pwconv1"), (5, "pwconv2")]:
                for suffix in ("weight", "bias"):
                    src = f"features.{src_idx}.{b}.block.{src_sub}.{suffix}"
                    tgt = f"conv_branch.{tgt_name}.{b}.{tgt_sub}.{suffix}"
                    if src in cn_sd:
                        target_sd[tgt] = cn_sd[src]
            # layer_scale exists in torchvision but not in our block; skip.

    # downsamples: features[2,4,6] each contain LayerNorm at .0 and Conv2d at .1.
    for src_idx, tgt_name in cn_down_map:
        for src_sub, tgt_sub in [(0, "norm"), (1, "conv")]:
            for suffix in ("weight", "bias"):
                src = f"features.{src_idx}.{src_sub}.{suffix}"
                tgt = f"conv_branch.{tgt_name}.{tgt_sub}.{suffix}"
                if src in cn_sd:
                    target_sd[tgt] = cn_sd[src]

    # --- DeiT-Small from timm ---
    deit = timm.create_model("deit_small_patch16_224", pretrained=True, num_classes=0)
    deit_sd = deit.state_dict()

    target_sd["vit_branch.cls_token"] = deit_sd["cls_token"]
    target_sd["vit_branch.pos_embed"] = deit_sd["pos_embed"]
    target_sd["vit_branch.patch_embed.weight"] = deit_sd["patch_embed.proj.weight"]
    target_sd["vit_branch.patch_embed.bias"] = deit_sd["patch_embed.proj.bias"]
    target_sd["vit_branch.norm.weight"] = deit_sd["norm.weight"]
    target_sd["vit_branch.norm.bias"] = deit_sd["norm.bias"]

    # First 12 of our 20 blocks take DeiT-Small weights; last 8 stay random.
    for b in range(12):
        for sub in ("norm1", "norm2"):
            for suffix in ("weight", "bias"):
                target_sd[f"vit_branch.blocks.{b}.{sub}.{suffix}"] = \
                    deit_sd[f"blocks.{b}.{sub}.{suffix}"]
        for sub in ("qkv", "proj"):
            for suffix in ("weight", "bias"):
                target_sd[f"vit_branch.blocks.{b}.attn.{sub}.{suffix}"] = \
                    deit_sd[f"blocks.{b}.attn.{sub}.{suffix}"]
        # MLP: timm's blocks.X.mlp.fc{1,2} → ours blocks.X.mlp.{0,2} (Sequential).
        for src_sub, tgt_sub in [("fc1", "0"), ("fc2", "2")]:
            for suffix in ("weight", "bias"):
                target_sd[f"vit_branch.blocks.{b}.mlp.{tgt_sub}.{suffix}"] = \
                    deit_sd[f"blocks.{b}.mlp.{src_sub}.{suffix}"]

    # Apply: load with strict=False so ECA, last-8 ViT blocks, fusion head stay random.
    own_sd = model.state_dict()
    n_loaded = 0
    for k, v in target_sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k] = v
            n_loaded += 1
    model.load_state_dict(own_sd)
    return n_loaded, len(own_sd)


@registry.register("paper2_econv_vit")
def build_paper2(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """Build EConv-ViT.

    pretrained=True initializes the ConvNeXt branch from torchvision ConvNeXt-Small
    ImageNet weights and the first 12 ViT blocks from timm DeiT-Small ImageNet
    weights. The paper itself doesn't specify pretraining; we make the choice
    explicit because training from scratch on small datasets (e.g., the apple
    PV subset) collapses to majority-class predictions in 1 epoch.
    """
    model = EConvViT(num_classes=num_classes)
    if pretrained:
        n_loaded, n_total = _load_pretrained_into(model)
        print(f"[paper2_econv_vit] loaded {n_loaded}/{n_total} pretrained params "
              f"({n_loaded/n_total*100:.1f}%)")
    return model
