import torch
import torch.nn as nn
from typing import List, Dict

import sys
sys.path.insert(0, "/mnt/data/omkumar/foundation_model/shared_foundation/models/network")
try:
    from blocks import ConvBlock, DoubleAttBlock, UpConvBlock
except:
    from .blocks import ConvBlock, DoubleAttBlock, UpConvBlock


# ─────────────────────────────────────────────────────────────
#  BACKBONE  (shared across all datasets)
# ─────────────────────────────────────────────────────────────
class ELiTNetEncoder(nn.Module):
    """
    Shared backbone. Learns dataset-agnostic spatial features.
    in_c        : number of input channels (e.g. 3 for RGB)
    latent_c    : channel width of the bottleneck (replaces the old 'num_classes' misuse)
    layers      : list of channel widths per encoder stage
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
        num_repetitions: int = 1
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
            
            # First block: handles pooling and channel expansion
            stage_blocks.append(DoubleAttBlock(
                in_c=layers[i], out_c=layers[i + 1], k_sz=3,
                shortcut=shortcut, pool=pool,
                attention=attention,
                residual=residual, causal=causal, conv_mode=conv_mode,
                num_blocks=num_blocks, bottleneck_factor=bottleneck_factor
            ))
            
            # Subsequent blocks: depth only, no pooling, same channels
            for _ in range(num_repetitions - 1):
                stage_blocks.append(DoubleAttBlock(
                    in_c=layers[i + 1], out_c=layers[i + 1], k_sz=3,
                    shortcut=shortcut, pool=False,
                    attention=attention,
                    residual=residual, causal=causal, conv_mode=conv_mode,
                    num_blocks=num_blocks, bottleneck_factor=bottleneck_factor
                ))
            
            self.down_path.append(stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first(x)
        for idx,stage in enumerate(self.down_path):
            for block in stage:
                x = block(x)
        return x


# ─────────────────────────────────────────────────────────────
#  HEAD  (one per dataset)
# ─────────────────────────────────────────────────────────────
class ELiTNetDecoder(nn.Module):
    """
    Shared decoder path. Learns dataset-agnostic spatial reconstruction.
    layers      : SAME list used by the encoder (reversed internally)
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
        conv_mode: str = 'MKConv2D'
    ):
        super().__init__()

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(
                in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                up_mode=up_mode, conv_bridge=conv_bridge,
                shortcut=shortcut, skip_conn=skip_conn,
                residual=residual, causal=causal, conv_mode=conv_mode
            )
            self.up_path.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for up in self.up_path:
            x = up(x)
        return x


class ELiTNetHead(nn.Module):
    """
    Dataset-specific segmentation head (final layer).
    n_classes   : number of output classes for this dataset
    in_c        : input channels (usually layers[0])
    """
    def __init__(self, in_c: int, n_classes: int):
        super().__init__()
        self.final = nn.Conv2d(in_c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(x)


# ─────────────────────────────────────────────────────────────
#  MULTI-HEAD ElitNet
# ─────────────────────────────────────────────────────────────
class ElitNet(nn.Module):
    """
    Foundation segmentation model with a shared backbone and
    dataset-specific decoder heads.

    Args:
        in_channels  : input channels (must be the same for all datasets, e.g. 3)
        dataset_classes : dict mapping dataset_name -> num_classes
                          e.g. {'US_Nerve': 2, 'IDRiD': 6}
        layers       : channel-width schedule shared by encoder and all decoders
        kernel_sz    : convolution kernel size
        up_mode      : upsampling strategy ('pixelshuffle' | 'up_conv' | 'transp_conv')
        pool         : downsampling strategy ('pool' | 'conv' | False)
        conv_bridge  : use conv bridge in decoder skip connections
        shortcut     : residual shortcuts in blocks
        skip_conn    : skip connections in decoder
        residual     : residual blocks in 
        causal       : causal convolutions in trunk
        conv_mode    : conv variant ('Conv2d' | 'MKConv2D' | 'DecomConv2D' | 'SeparableConv2d')

    Forward:
        x            : (B, C, H, W) input tensor
        dataset_name : str key matching one of dataset_classes

    Returns:
        logits       : (B, num_classes_for_dataset, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        dataset_classes: Dict[str, int],   # {'US_Nerve': 2, 'IDRiD': 6}
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
        num_repetitions: int = 1
    ):
        super().__init__()

        print("=" * 70)
        print(f"Initializing ElitNet (multi-head)")
        print(f"  Datasets : {list(dataset_classes.keys())}")
        print(f"  Classes  : {dataset_classes}")
        print(f"  Layers   : {layers}")
        print("=" * 70)

        # ── Shared backbone ──────────────────────────────────────────────
        self.encoder = ELiTNetEncoder(
            in_c=in_channels,
            latent_c=layers[-1],      # bottleneck width
            layers=layers,
            pool=pool,
            residual=residual,
            causal=causal,
            conv_mode=conv_mode,
            num_blocks=num_blocks,
            bottleneck_factor=bottleneck_factor,
            num_repetitions=num_repetitions
        )

        # ── Shared decoder ────────────────────────────────────────────────
        self.decoder = ELiTNetDecoder(
            k_sz=kernel_sz,
            layers=layers,
            up_mode=up_mode,
            conv_bridge=conv_bridge,
            shortcut=shortcut,
            skip_conn=skip_conn,
            residual=residual,
            causal=causal,
            conv_mode=conv_mode
        )

        # ── Per-dataset final heads ───────────────────────────────────────
        self.heads = nn.ModuleDict({
            dataset_name: ELiTNetHead(
                in_c=layers[0],
                n_classes=n_classes
            )
            for dataset_name, n_classes in dataset_classes.items()
        })

        self.dataset_classes = dataset_classes

    # ── Convenience ──────────────────────────────────────────────────────
    def get_head(self, dataset_name: str) -> ELiTNetHead:
        if dataset_name not in self.heads:
            raise KeyError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {list(self.heads.keys())}"
            )
        return self.heads[dataset_name]

    def encoder_parameters(self):
        """Useful for freezing / different LR schedules."""
        return self.encoder.parameters()

    def decoder_parameters(self):
        """Shared decoder parameters."""
        return self.decoder.parameters()

    def head_parameters(self, dataset_name: str):
        """Dataset-specific final layer parameters."""
        return self.get_head(dataset_name).parameters()

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, dataset_name: str) -> torch.Tensor:
        """
        Args:
            x            : input image tensor (B, C, H, W)
            dataset_name : which head to use, e.g. 'US_Nerve' or 'IDRiD'
        Returns:
            segmentation logits (B, num_classes, H, W)
        """
        features = self.encoder(x)
        dec_features = self.decoder(features)
        return self.get_head(dataset_name)(dec_features)


if __name__ == "__main__":
    # Test configuration
    in_channels = 3
    dataset_classes = {
        'US_Nerve': 2,
        'IDRiD': 6,
        'Cholec8k': 13
    }
    layers = [4,8,16,24,32]
    
    # Initialize model with heavier configuration
    model = ElitNet(
        in_channels=in_channels,
        dataset_classes=dataset_classes,
        layers=layers,
        conv_mode='Conv2d',  # Using standard Conv2d for simple testing
        num_blocks=2,        # Option 3
        bottleneck_factor=4, # Option 4
        num_repetitions=2  # Option 2: 2 blocks per resolution stage
    )
    
    # Create dummy input (B, C, H, W)
    dummy_input = torch.randn(2, 3, 640,640)
    
    print("\nStarting forward pass tests...")
    for ds_name, n_cls in dataset_classes.items():
        output = model(dummy_input, ds_name)
        print(f"Dataset: {ds_name:10} | Input: {dummy_input.shape} | Output: {output.shape} (Expected classes: {n_cls})")
        
        # Verify shape
        assert output.shape == (2, n_cls, 512, 640), f"Shape mismatch for {ds_name}!"
        
    print("\nAll tests passed successfully!")