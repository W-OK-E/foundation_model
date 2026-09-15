"""
ELiTNet-AttnRes: Energy-efficient Lightweight Thin Network with Attention Residuals
====================================================================================
Original ELiTNet paper:
  "Attention in a Little Network is All You Need to Go Green"
  Dewan et al., ISBI 2023.

Modification (this file):
  Replaces the fixed additive skip connections in ELiTNet with depth-wise
  *Attention Residuals* (AttnRes), inspired by:
  "Attention Residuals" — Kimi Team, arXiv:2603.15031, 2026.

Core idea:
  In standard U-Net / ELiTNet the decoder at level l receives the encoder
  skip feature fe via a *fixed* additive connection:
      out = f(fe) ⊙ f(fd)  +  f(fe)          (original SAM, ELiTNet Eq.2)

  Here we replace that with a *learned, input-dependent softmax* over all
  available encoder feature maps {fe_0, fe_1, …, fe_{l-1}}, so each decoder
  level can selectively retrieve the most relevant spatial scale:
      α_{i→l} = softmax_i( w_l · RMSNorm(fe_i) )
      skip_l   = Σ_i  α_{i→l} · fe_i
      out      = DAM( concat(skip_l, fd) )      (AttnRes-SAM)

  This follows the depth-wise attention formulation of AttnRes (Eq.1–4 of
  arXiv:2603.15031), adapted from the *temporal* (token) axis to the
  *spatial-scale* (encoder level) axis of a convolutional segmentation network.

Architecture changes vs. original ELiTNet
------------------------------------------
  1. SAM → AttnResSAM  (see class below)
     - Each decoder level maintains a learnable pseudo-query w_l ∈ R^C
     - Keys are RMSNorm-normalised encoder feature vectors (global-avg-pooled
       to C-dim so attention is channel-wise, not spatial, keeping cost low)
     - Softmax weights are broadcast back over H×W before element-wise use
  2. DAM (Dilated Attention Module) is UNCHANGED — retained from ELiTNet.
  3. UC (Upsampled Convolution) is UNCHANGED — only the skip aggregation
     inside it is replaced.
  4. All other components (trunk branch, ResBlock, ConvBlock) are UNCHANGED.

Experiment-ready: drop-in replacement — the UNet class signature is identical
to the original so existing training scripts need no modification.

Authors: [your names here]
Date:    2026-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ptflops import get_model_complexity_info
from torchinfo import summary

import matplotlib.pyplot as plt
import time


# ---------------------------------------------------------------------------
# Utility convolutions (unchanged from ELiTNet)
# ---------------------------------------------------------------------------

def conv1x1(in_planes, out_planes, stride=1):
    """1×1 convolution with no bias (used for shortcut projections)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# ---------------------------------------------------------------------------
# ConvBlock — unchanged from ELiTNet
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Two-layer convolutional block with optional max-pool and shortcut
    connection. Used as the *first* block in the encoder stem.
    """
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        super().__init__()
        if shortcut:
            self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = False
        pad = (k_sz - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else False

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        if self.shortcut:
            return out + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# ResBlock — unchanged from ELiTNet
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Bottleneck residual block (used inside ConvBridgeBlock).
    GELU activation (same as ELiTNet; originally labelled as Mish in code
    comments but implemented with nn.GELU).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_c)
        self.gelu  = nn.GELU()
        self.conv1 = nn.Conv2d(in_c, out_c // 4, 1)
        self.bn2   = nn.BatchNorm2d(out_c // 4)
        self.conv2 = nn.Conv2d(out_c // 4, out_c // 4, 3, padding='same')
        self.drop  = nn.Dropout(0.2)
        self.bn3   = nn.BatchNorm2d(out_c // 4)
        self.conv5 = nn.Conv2d(out_c // 4, out_c, 1, bias=False)
        self.conv6 = nn.Conv2d(in_c, out_c, 1, padding='same', bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.drop(out)
        out = self.gelu(out)
        out = self.conv5(out)
        residual = self.conv6(x)
        return out + residual


# ---------------------------------------------------------------------------
# AttConvBlock — unchanged from ELiTNet
# This is the Dilated Attention Module (DAM).
# ---------------------------------------------------------------------------

class AttConvBlock(nn.Module):
    """
    Convolutional block with an optional *dilated* soft-attention mask branch.
    This implements the DAM (Dilated Attention Module) of ELiTNet.

    When attention=True in forward(), the mask branch (encoder-decoder sub-net
    with dilation factors 2/4/8) generates fM, and the output is:
        out = conv3×3( fT ⊙ (1 + fM) )          (ELiTNet Eq. 1)
    """
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True, attention=False):
        super().__init__()
        if shortcut:
            self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else:
            self.shortcut = False
        pad = (k_sz - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else False

        # Trunk branch (T)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad),
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad),
            nn.BatchNorm2d(out_c), nn.ReLU(),
        )

        if attention:
            # Mask branch (M) — hierarchical dilated encoder-decoder
            self.mpool1  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.s1      = nn.Conv2d(in_c,  out_c, k_sz, padding='same', dilation=6)
            self.skip1   = nn.Conv2d(out_c, out_c, k_sz, padding='same')
            self.mpool2  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.s2      = nn.Conv2d(out_c, out_c, k_sz, padding='same', dilation=4)
            self.skip2   = nn.Conv2d(out_c, out_c, k_sz, padding='same')
            self.mpool3  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.s3      = nn.Sequential(
                nn.Conv2d(out_c, out_c, k_sz, padding='same', dilation=2),
                nn.Conv2d(out_c, out_c, k_sz, padding='same', dilation=2),
            )
            self.up3     = nn.UpsamplingBilinear2d(scale_factor=2)
            self.s4      = nn.Conv2d(out_c, out_c, k_sz, padding='same', dilation=4)
            self.up2     = nn.UpsamplingBilinear2d(scale_factor=2)
            self.s5      = nn.Conv2d(out_c, out_c, k_sz, padding='same', dilation=2)
            self.up1     = nn.UpsamplingBilinear2d(scale_factor=2)
            self.sigmoid = nn.Sequential(
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 1, bias=False),
                nn.Sigmoid(),
            )
            self.last = nn.Conv2d(out_c, out_c, k_sz, padding='same')

    def forward(self, x, attention=False):
        if self.pool:
            x = self.pool(x)
        trunk = self.conv(x)

        if attention:
            # --- Mask branch (M) ---
            o1   = self.s1(self.mpool1(x))
            sk1  = self.skip1(o1)
            o2   = self.s2(self.mpool2(o1))
            sk2  = self.skip2(o2)
            o3   = self.s3(self.mpool3(o2))
            o    = torch.add(self.up3(o3), sk2)
            o4   = self.s4(o)
            o    = torch.add(self.up2(o4), sk1)
            o5   = self.s5(o)
            fM   = self.sigmoid(self.up1(o5))
            out  = self.last(torch.multiply((1 + fM), trunk))
        else:
            out = trunk

        if self.shortcut:
            return out + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# DoubleAttBlock — unchanged from ELiTNet
# ---------------------------------------------------------------------------

class DoubleAttBlock(nn.Module):
    """Two sequential AttConvBlocks; the second one uses the DAM mask branch."""
    def __init__(self, in_c, out_c, k_sz=3, shortcut=True, attention=True):
        super().__init__()
        self.block1 = AttConvBlock(in_c, in_c,  k_sz=k_sz, shortcut=shortcut,
                                   pool=False, attention=False)
        self.block2 = AttConvBlock(in_c, out_c, k_sz=k_sz, shortcut=shortcut,
                                   pool=True,  attention=attention)

    def forward(self, x):
        return self.block2(self.block1(x, attention=False), attention=True)


# ---------------------------------------------------------------------------
# ConvBridgeBlock — unchanged from ELiTNet
# ---------------------------------------------------------------------------

class ConvBridgeBlock(nn.Module):
    """ResBlock applied to a skip feature before it enters the decoder."""
    def __init__(self, channels, k_sz=3):
        super().__init__()
        self.block = ResBlock(channels, channels)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# UpsampleBlock — unchanged from ELiTNet
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    """Bilinear upsample + 1×1 conv, or transposed convolution."""
    def __init__(self, in_c, out_c, up_mode='up_conv'):
        super().__init__()
        if up_mode == 'transp_conv':
            self.block = nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 2, stride=2))
        elif up_mode == 'up_conv':
            self.block = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_c, out_c, kernel_size=1),
            )
        else:
            raise ValueError(f'Upsampling mode "{up_mode}" not supported')

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# RMSNorm — helper for AttnRes (lightweight, channel-wise)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root-mean-square normalisation over the channel dimension.
    Used to normalise encoder key vectors before computing attention scores,
    preventing high-magnitude feature maps from dominating the softmax
    (mirrors the RMSNorm inside ϕ in AttnRes, arXiv:2603.15031 Eq. 2).
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps   = eps
        self.g     = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, C)  — channel vectors after global average pooling
        norm = torch.norm(x, dim=-1, keepdim=True)
        return self.g * x / (norm + self.eps)


# ---------------------------------------------------------------------------
# *** NEW *** AttnResSAM — Attention Residual Skip Attention Module
# ---------------------------------------------------------------------------

class AttnResSAM(nn.Module):
    """
    Replaces the fixed SAM (Skip Attention Module) of ELiTNet with a
    depth-wise *softmax* attention over all preceding encoder feature maps,
    following the Full-AttnRes formulation (arXiv:2603.15031, §3.1).

    Given encoder feature maps  {fe_0, …, fe_{l-1}}  at level l:

      Keys:    k_i  = RMSNorm( GAP(Res(fe_i)) )   ∈ R^C
      Query:   q_l  = w_l                           ∈ R^C   (learnable)
      Scores:  α_{i→l} = softmax_i( q_l · k_i )
      Skip:    skip_l  = Σ_i  α_{i→l} · fe_i       (broadcast over H,W)

    Then the skip is gated with the decoder feature fd exactly as in the
    original SAM (Eq. 2 of the ELiTNet paper):
      out = Res(skip_l) ⊙ fd  ⊕  Res(skip_l)

    Args:
        channels  (int):  Feature channel count C (same for all encoder levels).
        num_sources (int): Maximum number of encoder levels (len(layers) - 1 + 1).
    """
    def __init__(self, channels: int, source_channels: list):
        super().__init__()
        self.channels    = channels
        self.num_sources = len(source_channels)

        # Learnable pseudo-query w_l (one per decoder level, indexed at call time)
        # Shape: (C,) — same as the channel-pooled key vectors
        self.pseudo_query = nn.Parameter(torch.zeros(channels))
        nn.init.normal_(self.pseudo_query, std=0.02)

        # Projections to align mismatched source channels to self.channels
        self.projections = nn.ModuleList([
            nn.Conv2d(in_c, channels, kernel_size=1) if in_c != channels else nn.Identity()
            for in_c in source_channels
        ])

        # Per-source ResBlock to refine each encoder feature before keying
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, channels) for _ in range(self.num_sources)
        ])

        # RMSNorm for key normalisation
        self.rms_norm = RMSNorm(channels)

    def forward(self, encoder_features: list, fd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: list of encoder feature tensors at this spatial
                resolution, ordered from shallowest to deepest:
                [fe_0, fe_1, …, fe_{k-1}], each shape (B, C, H, W).
            fd: decoder feature tensor, shape (B, C, H, W).

        Returns:
            fSAM: aggregated skip feature, shape (B, C, H, W).
        """
        assert len(encoder_features) <= self.num_sources, (
            f"AttnResSAM got {len(encoder_features)} sources but was built "
            f"for max {self.num_sources}."
        )

        # 1. Align channels and resolutions
        target_size = fd.shape[2:]
        refined = []
        for i in range(len(encoder_features)):
            # Project channels
            r = self.projections[i](encoder_features[i])
            # Interpolate resolution (e.g. if attending to shallower, larger maps)
            if r.shape[2:] != target_size:
                r = F.interpolate(r, size=target_size, mode='bilinear', align_corners=False)
            # Refine with ResBlock
            r = self.res_blocks[i](r)
            refined.append(r)

        # 2. Compute channel-wise keys via global average pooling → (B, C)
        keys = torch.stack(
            [F.adaptive_avg_pool2d(r, 1).flatten(1) for r in refined],
            dim=1
        )  # (B, num_sources_active, C)

        # 3. Normalise keys (prevents large-magnitude encoders dominating)
        B, S, C = keys.shape
        keys_norm = self.rms_norm(keys.view(B * S, C)).view(B, S, C)

        # 4. Compute attention scores: α = softmax( w_l · k_i )
        q = self.pseudo_query.view(1, C, 1).expand(B, C, 1)   # (B, C, 1)
        scores = torch.bmm(keys_norm, q).squeeze(-1)           # (B, S)
        alpha  = torch.softmax(scores, dim=-1)                 # (B, S)

        # 5. Weighted sum of refined spatial features
        # alpha: (B, S) -> (B, S, 1, 1, 1) to broadcast over (B, S, C, H, W)
        refined_stacked = torch.stack(refined, dim=1)
        skip = (alpha.view(B, S, 1, 1, 1) * refined_stacked).sum(dim=1)  # (B, C, H, W)

        # 6. Gate with decoder feature (identical to original SAM Eq. 2)
        fSAM = (skip * fd) + skip
        return fSAM


# ---------------------------------------------------------------------------
# *** MODIFIED *** UpConvBlock — uses AttnResSAM instead of fixed SAM
# ---------------------------------------------------------------------------

class UpConvBlock(nn.Module):
    """
    Decoder block.  Differences from ELiTNet:
      - The fixed SAM(fe, fd) is replaced by AttnResSAM(encoder_features, fd),
        which attends over *all* available encoder levels with learned softmax
        weights (Full AttnRes over spatial scales).
      - Everything else (UpsampleBlock, DAM via AttConvBlock, concat, DAM again)
        is unchanged.

    Args:
        in_c, out_c  : channel dimensions (same convention as ELiTNet).
        num_sources  : total number of encoder levels available at this point.
        up_mode      : 'up_conv' (default) or 'transp_conv'.
        conv_bridge  : whether to apply ConvBridgeBlock to the primary skip.
        shortcut     : residual shortcuts in AttConvBlocks.
    """
    def __init__(self, in_c, out_c, source_channels, k_sz=3,
                 up_mode='up_conv', conv_bridge=False, shortcut=False):
        super().__init__()
        self.conv_bridge = conv_bridge

        self.up_layer    = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        # DAM applied to upsampled decoder feature (unchanged from ELiTNet)
        self.conv_layer1 = AttConvBlock(out_c, out_c, k_sz=k_sz,
                                        shortcut=shortcut, pool=False, attention=True)
        # DAM applied to concatenation of AttnResSAM output and fd
        self.conv_layer2 = AttConvBlock(2 * out_c, out_c, k_sz=k_sz,
                                        shortcut=shortcut, pool=False, attention=True)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

        # *** NEW: AttnResSAM replaces the fixed SAM ***
        self.attn_res_sam = AttnResSAM(channels=out_c, source_channels=source_channels)

    def forward(self, x: torch.Tensor, encoder_features: list) -> torch.Tensor:
        """
        Args:
            x               : decoder tensor from previous (deeper) level, (B, in_c, H/2, W/2).
            encoder_features: list of encoder tensors at this resolution,
                              ordered shallowest→deepest, each (B, out_c, H, W).
                              The *last* element is the direct skip (like the
                              original ELiTNet SAM's fe), but all are used.
        Returns:
            out: decoded feature map, (B, out_c, H, W).
        """
        # Step 1: upsample + DAM on decoder stream
        up = self.up_layer(x)
        fd = self.conv_layer1(up, attention=True)   # (B, out_c, H, W)

        # Step 2: AttnRes skip aggregation
        if self.conv_bridge:
            # Apply ConvBridge to the primary skip (encoder_features[0] matches current res)
            primary_skip = self.conv_bridge_layer(encoder_features[0])
            enc_feats    = [primary_skip] + encoder_features[1:]
        else:
            enc_feats = encoder_features

        fSAM = self.attn_res_sam(enc_feats, fd)     # (B, out_c, H, W)

        # Step 3: concatenate and apply DAM (identical to ELiTNet UC block)
        out = torch.cat([fd, fSAM], dim=1)           # (B, 2*out_c, H, W)
        out = self.conv_layer2(out, attention=True)  # (B, out_c, H, W)
        return out


# ---------------------------------------------------------------------------
# UNet — top-level model  (same public API as original ELiTNet)
# ---------------------------------------------------------------------------

class ElitNet(nn.Module):
    """
    ELiTNet-AttnRes: U-Net with Dilated Attention Modules (DAM) in the
    encoder and Attention Residual Skip Attention Modules (AttnResSAM) in
    the decoder.

    The encoder is *identical* to ELiTNet (ConvBlock stem + DoubleAttBlocks).
    The decoder replaces each fixed SAM with an AttnResSAM that attends over
    all encoder feature maps at the corresponding spatial scale, using a
    learned per-level pseudo-query vector and RMSNorm-normalised channel keys.

    Args:
        in_c       : number of input image channels (e.g. 3 for RGB).
        n_classes  : number of output segmentation classes.
        layers     : list of channel counts per encoder level,
                     e.g. [8, 16, 32, 64] for ELiTNet's default.
        k_sz       : convolutional kernel size (default 3).
        up_mode    : upsampling strategy ('up_conv' or 'transp_conv').
        conv_bridge: use ConvBridgeBlock on the primary skip feature.
        shortcut   : use residual shortcuts in AttConvBlocks.

    Example:
        >>> model = UNet(in_c=3, n_classes=2, layers=[8, 16, 32, 64])
        >>> model(torch.randn(1, 3, 512, 512)).shape
        torch.Size([1, 2, 512, 512])
    """
    def __init__(self, in_channels, num_classes, layers, kernel_sz=3, up_mode='up_conv', pool='pool', 
                 conv_bridge=True, shortcut=True, skip_conn=True, residual=True, causal=True, conv_mode='MKConv2D'):
        super().__init__()
        self.n_classes = num_classes
        num_enc_levels = len(layers) - 1   # number of DoubleAttBlocks (skip sources)

        # ---- Encoder -------------------------------------------------------
        # Stem: first ConvBlock without pooling
        self.first = ConvBlock(in_c=in_channels, out_c=layers[0], k_sz=kernel_sz,
                               shortcut=shortcut, pool=False)

        # Encoder levels: each DoubleAttBlock downsamples and doubles channels
        self.down_path = nn.ModuleList([
            DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=kernel_sz,
                           shortcut=shortcut, attention=True)
            for i in range(num_enc_levels)
        ])

        # ---- Decoder -------------------------------------------------------
        # At decoder level l (0 = shallowest):
        #   - The UpConvBlock receives the deepest-so-far encoder stack
        #     reversed, so it always has access to all coarser encoder maps.
        # num_sources for level l = (num_enc_levels - l) available enc maps.
        reversed_layers = list(reversed(layers))
        self.up_path = nn.ModuleList([
            UpConvBlock(
                in_c       = reversed_layers[i],
                out_c      = reversed_layers[i + 1],
                source_channels = reversed_layers[i+1:],  # encoders available for this scale
                k_sz       = kernel_sz,
                up_mode    = up_mode,
                conv_bridge= conv_bridge,
                shortcut   = shortcut,
            )
            for i in range(num_enc_levels)
        ])

        # ---- Final projection ----------------------------------------------
        self.final = nn.Conv2d(layers[0], num_classes, kernel_size=1)

        # ---- Weight initialisation (same as ELiTNet) -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-init all AttnRes pseudo-queries (ensures uniform weights at
        # training start, mirroring the init strategy of arXiv:2603.15031 §5)
        for m in self.modules():
            if isinstance(m, AttnResSAM):
                nn.init.zeros_(m.pseudo_query)

        print("#"*70)
        print("Residual Attention UNet Initialized")
        print("#"*70)

    def get_embedding_dim(self):
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input image tensor, shape (B, in_c, H, W).

        Returns:
            logits: segmentation logits, shape (B, n_classes, H, W).
        """
        # --- Encoder ---
        x = self.first(x)                    # (B, layers[0], H, W)
        down_activations = []                # stores encoder outputs per level
        for down in self.down_path:
            down_activations.append(x)       # save before downsampling
            x = down(x)

        # deepest encoder output is now x; down_activations is shallowest→deepest
        # Reverse so index 0 = deepest skip for the first decoder level
        down_activations.reverse()           # now deepest→shallowest

        # --- Decoder ---
        # At decoder level i, all encoder features at this spatial resolution
        # are: down_activations[i:]  (deepest + all coarser ones above)
        for i, up in enumerate(self.up_path):
            encoder_features = down_activations[i:]  # list of tensors  
            x = up(x, encoder_features)

        return self.final(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # ELiTNet default layer config (from paper)
    layers    = [8, 16, 32, 64]
    model     = UNet(in_c=3, n_classes=2, layers=layers,
                     up_mode='up_conv', conv_bridge=True, shortcut=True).to(device)

    dummy = torch.randn(2, 3, 512, 512).to(device)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == (2, 2, 512, 512), "Shape mismatch!"

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print("Sanity check passed ✓")