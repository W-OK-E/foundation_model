import torch
import torch.nn as nn
from typing import Dict, List, Optional

import sys
sys.path.insert(0, "/mnt/data/omkumar/foundation_model/shared_foundation/models/network-sep-decoder")
try:
    from blocks import ConvBlock, DoubleAttBlock, UpConvBlock
except ImportError:
    from .blocks import ConvBlock, DoubleAttBlock, UpConvBlock


# ─────────────────────────────────────────────────────────────
#  ENCODER  (shared across all datasets)
# ─────────────────────────────────────────────────────────────
class ELiTNetEncoder(nn.Module):
    """
    Shared backbone. Learns dataset-agnostic spatial features.
    Trained at a lower learning rate than any decoder so that
    foundational representations are not disrupted by task-specific
    gradients.
    """
    def __init__(
        self,
        in_c: int,
        latent_c: int,
        layers: List[int],
        shortcut: bool = True,
        pool: str = 'pool',
        residual: bool = True,
        causal: bool = True,
        conv_mode: str = 'MKConv2D',
        num_blocks: int = 2,
        bottleneck_factor: int = 4,
        num_repetitions: int = 1,
    ):
        super().__init__()
        self.first = ConvBlock(
            in_c=in_c, out_c=layers[0], k_sz=3,
            shortcut=shortcut, pool=False, conv_mode=conv_mode
        )

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            attention = (i <= 7)
            stage_blocks = nn.ModuleList()

            stage_blocks.append(DoubleAttBlock(
                in_c=layers[i], out_c=layers[i + 1], k_sz=3,
                shortcut=shortcut, pool=pool,
                attention=attention,
                residual=residual, causal=causal, conv_mode=conv_mode,
                num_blocks=num_blocks, bottleneck_factor=bottleneck_factor,
            ))

            for _ in range(num_repetitions - 1):
                stage_blocks.append(DoubleAttBlock(
                    in_c=layers[i + 1], out_c=layers[i + 1], k_sz=3,
                    shortcut=shortcut, pool=False,
                    attention=attention,
                    residual=residual, causal=causal, conv_mode=conv_mode,
                    num_blocks=num_blocks, bottleneck_factor=bottleneck_factor,
                ))

            self.down_path.append(stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        for stage in self.down_path:
            for block in stage:
                x = block(x)
        return x


# ─────────────────────────────────────────────────────────────
#  DECODER  (one independent instance per dataset)
# ─────────────────────────────────────────────────────────────
class ELiTNetDecoder(nn.Module):
    """
    Per-dataset decoder path.  Each dataset owns its own copy so
    US-Nerve's simple gradients cannot pull IDRiD's decoder toward
    trivial blob-detection solutions.

    layers  : the SAME list used by the encoder (reversed internally).
    """
    def __init__(
        self,
        k_sz: int,
        layers: List[int],
        up_mode: str = 'up_conv',
        conv_bridge: bool = True,
        shortcut: bool = True,
        skip_conn: bool = True,
        residual: bool = True,
        causal: bool = True,
        conv_mode: str = 'MKConv2D',
    ):
        super().__init__()

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(
                in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                up_mode=up_mode, conv_bridge=conv_bridge,
                shortcut=shortcut, skip_conn=skip_conn,
                residual=residual, causal=causal, conv_mode=conv_mode,
            )
            self.up_path.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for up in self.up_path:
            x = up(x)
        return x


# ─────────────────────────────────────────────────────────────
#  HEAD  (dataset-specific final projection, no sharing)
# ─────────────────────────────────────────────────────────────
class ELiTNetHead(nn.Module):
    """
    1×1 conv that maps decoder features to class logits.
    Completely separate per dataset — no shared weights.
    """
    def __init__(self, in_c: int, n_classes: int):
        super().__init__()
        self.final = nn.Conv2d(in_c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(x)


# ─────────────────────────────────────────────────────────────
#  MULTI-HEAD ElitNet  (shared encoder, per-dataset decoders)
# ─────────────────────────────────────────────────────────────
class ElitNet(nn.Module):
    """
    Foundation segmentation model.

    Architecture
    ────────────
    Shared encoder  →  per-dataset decoder  →  per-dataset 1×1 head

    The encoder is intentionally trained at a *lower* learning rate
    (slow-moving foundation).  Each dataset's decoder + head can be
    given its own learning rate — in particular IDRiD's decoder
    should receive a *higher* LR than US-Nerve's to compensate for
    gradient dominance.

    Args
    ────
    in_channels     : input channels (same for all datasets, e.g. 3)
    dataset_classes : dict  dataset_name → num_classes
                      e.g. {'US_Nerve': 2, 'IDRiD': 6}
    layers          : channel-width schedule, e.g. [32, 64, 128, 256]
    kernel_sz       : convolution kernel size
    up_mode         : 'pixelshuffle' | 'up_conv' | 'transp_conv'
    pool            : 'pool' | 'conv' | False
    conv_bridge     : conv bridge in decoder skip connections
    shortcut        : residual shortcuts in blocks
    skip_conn       : skip connections in decoder
    residual        : residual blocks in trunk
    causal          : causal convolutions in trunk
    conv_mode       : 'Conv2d' | 'MKConv2D' | 'DecomConv2D' | 'SeparableConv2d'
    num_blocks      : ResBlocks per Trunk
    bottleneck_factor: compression ratio inside ResBlock
    num_repetitions : extra DoubleAttBlocks per encoder stage

    Forward
    ───────
    x            : (B, C, H, W) input tensor
    dataset_name : str key matching one of dataset_classes
    → logits     : (B, num_classes_for_dataset, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        dataset_classes: Dict[str, int],
        layers: List[int],
        kernel_sz: int = 3,
        up_mode: str = 'pixelshuffle',
        pool: str = 'pool',
        conv_bridge: bool = True,
        shortcut: bool = True,
        skip_conn: bool = True,
        residual: bool = True,
        causal: bool = True,
        conv_mode: str = 'MKConv2D',
        num_blocks: int = 2,
        bottleneck_factor: int = 4,
        num_repetitions: int = 1,
    ):
        super().__init__()

        print("=" * 70)
        print(f"Initializing ElitNet (shared encoder + per-dataset decoders)")
        print(f"  Datasets : {list(dataset_classes.keys())}")
        print(f"  Classes  : {dataset_classes}")
        print(f"  Layers   : {layers}")
        print("=" * 70)

        # ── Shared backbone (slow LR) ─────────────────────────────────────
        self.encoder = ELiTNetEncoder(
            in_c=in_channels,
            latent_c=layers[-1],
            layers=layers,
            pool=pool,
            residual=residual,
            causal=causal,
            conv_mode=conv_mode,
            num_blocks=num_blocks,
            bottleneck_factor=bottleneck_factor,
            num_repetitions=num_repetitions,
        )

        # ── Per-dataset decoders (independent; each can have its own LR) ──
        self.decoders = nn.ModuleDict({
            dataset_name: ELiTNetDecoder(
                k_sz=kernel_sz,
                layers=layers,
                up_mode=up_mode,
                conv_bridge=conv_bridge,
                shortcut=shortcut,
                skip_conn=skip_conn,
                residual=residual,
                causal=causal,
                conv_mode=conv_mode,
            )
            for dataset_name in dataset_classes
        })

        # ── Per-dataset final projection heads (no shared params) ─────────
        self.heads = nn.ModuleDict({
            dataset_name: ELiTNetHead(
                in_c=layers[0],
                n_classes=n_classes,
            )
            for dataset_name, n_classes in dataset_classes.items()
        })

        self.dataset_classes = dataset_classes

    # ── Convenience accessors ─────────────────────────────────────────────
    def _get_decoder(self, dataset_name: str) -> ELiTNetDecoder:
        if dataset_name not in self.decoders:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(self.decoders.keys())}"
            )
        return self.decoders[dataset_name]

    def _get_head(self, dataset_name: str) -> ELiTNetHead:
        if dataset_name not in self.heads:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(self.heads.keys())}"
            )
        return self.heads[dataset_name]

    # ── Parameter groups for optimizer setup ─────────────────────────────
    def parameter_groups(
        self,
        encoder_lr: float,
        default_decoder_lr: float,
        dataset_lr_overrides: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Build optimizer parameter groups with per-component learning rates.

        Typical use:
            groups = model.parameter_groups(
                encoder_lr=1e-4,
                default_decoder_lr=3e-4,
                dataset_lr_overrides={'IDRiD': 5e-4},
            )
            optimizer = torch.optim.Adam(groups)

        Args
        ────
        encoder_lr           : LR for the shared encoder (should be low)
        default_decoder_lr   : LR applied to all decoders + heads by default
        dataset_lr_overrides : optional per-dataset LR override for decoder+head
                               e.g. {'IDRiD': 5e-4} gives IDRiD a higher LR
        """
        if dataset_lr_overrides is None:
            dataset_lr_overrides = {}

        groups = [{'params': list(self.encoder.parameters()), 'lr': encoder_lr}]

        for ds_name in self.dataset_classes:
            lr = dataset_lr_overrides.get(ds_name, default_decoder_lr)
            params = (
                list(self.decoders[ds_name].parameters()) +
                list(self.heads[ds_name].parameters())
            )
            groups.append({'params': params, 'lr': lr})

        return groups

    # ── Legacy single-module parameter accessors ──────────────────────────
    def encoder_parameters(self):
        return self.encoder.parameters()

    def decoder_parameters(self, dataset_name: str):
        return self._get_decoder(dataset_name).parameters()

    def head_parameters(self, dataset_name: str):
        return self._get_head(dataset_name).parameters()

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """
        Args
        ────
        x            : (B, C, H, W) input image tensor
        dataset_name : which decoder+head to use, e.g. 'US_Nerve' or 'IDRiD'

        Returns
        ───────
        logits : (B, num_classes, H, W)
        """
        features = self.encoder(x)
        dec_features = self._get_decoder(dataset_name)(features)
        return self._get_head(dataset_name)(dec_features)


if __name__ == "__main__":
    in_channels = 3
    dataset_classes = {
        'US_Nerve': 2,
        'IDRiD': 6,
    }
    layers = [4, 8, 16, 24, 32]

    model = ElitNet(
        in_channels=in_channels,
        dataset_classes=dataset_classes,
        layers=layers,
        conv_mode='Conv2d',
        num_blocks=2,
        bottleneck_factor=4,
        num_repetitions=1,
    )

    dummy_input = torch.randn(2, 3, 256, 256)

    print("\nForward pass tests...")
    for ds_name, n_cls in dataset_classes.items():
        out = model(dummy_input, ds_name)
        print(f"  {ds_name:10} | input {tuple(dummy_input.shape)} | output {tuple(out.shape)}")

    print("\nParameter group LRs:")
    groups = model.parameter_groups(
        encoder_lr=1e-4,
        default_decoder_lr=3e-4,
        dataset_lr_overrides={'IDRiD': 5e-4},
    )
    for g in groups:
        total = sum(p.numel() for p in g['params'])
        print(f"  lr={g['lr']:.0e}  params={total:,}")

    print("\nVerifying no shared params between decoder heads...")
    us_ids  = {id(p) for p in model.decoder_parameters('US_Nerve')}
    idr_ids = {id(p) for p in model.decoder_parameters('IDRiD')}
    assert us_ids.isdisjoint(idr_ids), "Decoder heads share parameters!"
    head_us  = {id(p) for p in model.head_parameters('US_Nerve')}
    head_idr = {id(p) for p in model.head_parameters('IDRiD')}
    assert head_us.isdisjoint(head_idr), "Final heads share parameters!"
    print("  OK — all dataset components are fully independent.")
