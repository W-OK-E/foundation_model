import sys
sys.path.append("/mnt/data/omkumar/foundation_phase1/models/network_3d")

import torch.nn as nn
from typing import List
try:
    from .blocksv2_3d import ConvBlock3d, DoubleAttBlock3d, UpConvBlock3d
except ImportError:
    from blocksv2_3d import ConvBlock3d, DoubleAttBlock3d, UpConvBlock3d

class ELiTNetEncoder3d(nn.Module):
    def __init__(
        self,
        in_c: int,
        k_sz: int,
        layers: List[int],
        shortcut: bool = True,
        pool='pool',
        residual=True,
        causal=True,
        conv_mode='Conv3d'
    ):
        super().__init__()
        self.first = ConvBlock3d(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False, conv_mode=conv_mode)
        
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            if i <= 7:
                block = DoubleAttBlock3d(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
            else:
                block = DoubleAttBlock3d(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool, attention=False, residual=residual, causal=causal, conv_mode=conv_mode)
            self.down_path.append(block)
        
    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        return x, down_activations

class ELiTNetDecoder3d(nn.Module):
    def __init__(
        self,
        n_classes: int,
        k_sz: int,
        layers: List[int],
        up_mode='up_conv',
        conv_bridge: bool = True,
        shortcut: bool = True,
        skip_conn: bool = True,
        residual=True,
        causal=True,
        conv_mode='Conv3d'
    ):
        super().__init__()
        
        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock3d(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut, skip_conn=skip_conn, 
                                residual=residual, causal=causal, conv_mode=conv_mode)
            self.up_path.append(block)


    def forward(self, x, down_activations):
        for i, up in enumerate(self.up_path):
            skip = down_activations[i] if i < len(down_activations) else None
            x = up(x, skip)
        return x

class ELitNetFinalBlock3d(nn.Module):
    def __init__(self, in_c, out_c, k_sz, conv_mode='Conv3d'):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=k_sz)

    
    def forward(self, x):
        return self.conv(x)
        
class ElitNet3d(nn.Module):
    def __init__(self, in_channels, num_classes, layers, kernel_sz=3, up_mode='up_conv', pool='pool', 
                 conv_bridge=True, shortcut=True, skip_conn=True, residual=True, causal=True, conv_mode='Conv3d'):
        """
        ELiTNet 3D Architecture
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            layers: List of channel dimensions for each layer
            kernel_sz: Kernel size
            up_mode: Upsampling mode ('transp_conv', 'up_conv', 'pixelshuffle')
            pool: Pooling mode ('pool', 'conv', False)
            conv_bridge: Whether to use convolutional bridge in decoder
            shortcut: Whether to use shortcut connections
            skip_conn: Whether to use skip connections
            residual: Whether to use residual blocks
            causal: Whether to use causal convolutions
            conv_mode: Type of convolution to use (defaults to 'Conv3d')
        """
        super(ElitNet3d, self).__init__()
        print("="*70)
        print("Initializing 3D ElitNet")
        print("="*70)
        
        self.encoder = ELiTNetEncoder3d(in_channels, kernel_sz, layers, pool=pool, residual=residual, causal=causal, conv_mode=conv_mode)
        self.decoder = ELiTNetDecoder3d(num_classes, kernel_sz, layers, up_mode, conv_bridge, shortcut, skip_conn, residual, causal, conv_mode=conv_mode)
        self.final = ELitNetFinalBlock3d(layers[0], num_classes, 1)
        # Weight initialisation (matches reference ELiTNet3D)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        inp_shape = x.shape
        x, down_activations = self.encoder(x)
        x = self.decoder(x, down_activations)
        #If shape mismatch occurs, interpolate to match input shape (common in segmentation tasks)
        if x.shape[2:] != inp_shape[2:]:
            x = nn.functional.interpolate(x, size=inp_shape[2:], mode='trilinear', align_corners=False)
        x = self.final(x)
        return x
