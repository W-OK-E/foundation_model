import torch
import torch.nn as nn
from .pixel_shuffle import PixelShuffle3d 
# from module_3d import MKConv2D, CausalConv2d, DecomConv2D

def conv1x1_3d(in_planes, out_planes, stride=1):
    """1x1x1 3D convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_conv_layer_3d(in_c, out_c, k_sz=3, padding='same', conv_mode='Conv3d'):
    """
    Factory function to create 3D convolution layer based on conv_mode
    
    Args:
        in_c: Input channels
        out_c: Output channels
        k_sz: Kernel size
        padding: Padding mode ('same' or int)
        conv_mode: Type of convolution ('Conv3d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
    
    Returns:
        3D Convolution layer
    """
    if conv_mode in ['Conv3d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d']:
        # All custom modes fallback to standard Conv3d for 3D
        return nn.Conv3d(in_c, out_c, kernel_size=k_sz, padding=padding)
    else:
        raise ValueError(f"Unsupported conv_mode: {conv_mode}")

class ConvBlock3d(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True, conv_mode='Conv3d'):
        '''
        3D Convolution Block
        pool_mode can be False (no pooling) or True ('maxpool')
        conv_mode: Type of convolution (defaults to Conv3d for 3D)
        '''
        super(ConvBlock3d, self).__init__()
        if shortcut==True: 
            self.shortcut = nn.Sequential(conv1x1_3d(in_c, out_c), nn.BatchNorm3d(out_c))
        else: 
            self.shortcut=False

        block = []
        if pool: 
            self.pool = nn.MaxPool3d(kernel_size=2)
        else: 
            self.pool = False

        block.append(get_conv_layer_3d(in_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm3d(out_c))

        block.append(get_conv_layer_3d(out_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm3d(out_c))

        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        if self.pool: 
            x = self.pool(x)
        out = self.block(x)
        if self.shortcut: 
            return out + self.shortcut(x)
        else: 
            return out
        
class ResBlock3d(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, conv_mode='Conv3d'):
        super(ResBlock3d, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_c)
        self.mish = nn.GELU()
        self.conv1 = nn.Conv3d(in_c, out_c // 4, 1)
        self.bn2 = nn.BatchNorm3d(out_c // 4)
        self.conv2 = get_conv_layer_3d(out_c // 4, out_c // 4, k_sz=k_sz, padding='same', conv_mode=conv_mode)
        self.dropout = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm3d(out_c // 4)
        self.conv5 = nn.Conv3d(out_c // 4, out_c, 1, 1, bias=False)
        self.conv6 = nn.Conv3d(in_c, out_c, 1, 1, padding='same', bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.mish(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.mish(out)
        out = self.conv5(out)
        
        residual = self.conv6(x)
        out += residual
        return out

class Trunk3d(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, residual=True, causal=True, conv_mode='Conv3d'):
        '''
        3D Trunk block with residual and optional causal convolutions
        '''
        super(Trunk3d, self).__init__()
        if residual:   
            self.conv = nn.Sequential(
                ResBlock3d(in_c, out_c, k_sz=k_sz, conv_mode=conv_mode),
                ResBlock3d(out_c, out_c, k_sz=k_sz, conv_mode=conv_mode),
            )
        else:
            # Standard convolution path for non-residual
            self.conv = nn.Sequential(
                get_conv_layer_3d(in_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode),
                nn.BatchNorm3d(out_c),
                nn.ReLU(),
                get_conv_layer_3d(out_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode),
                nn.BatchNorm3d(out_c),
                nn.ReLU()
            )
    def forward(self, x):
        out = self.conv(x)
        return out
    
class AttConvBlock3d(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool='pool', attention=False, residual=True, causal=True, conv_mode='Conv3d'):
        '''
        3D Attention Convolution Block
        '''
        super(AttConvBlock3d, self).__init__()
        if shortcut==True: 
            self.shortcut = nn.Sequential(conv1x1_3d(in_c, out_c), nn.BatchNorm3d(out_c))
        else: 
            self.shortcut=False

        if pool=='pool':
            self.pool = nn.MaxPool3d(kernel_size=2)
        elif pool=='conv':
            self.pool = nn.Conv3d(in_c, in_c, kernel_size=2, stride=2)
        else:
            self.pool = False

        self.conv = Trunk3d(in_c, out_c, k_sz=k_sz, residual=residual, causal=causal, conv_mode=conv_mode)
        
        if attention==True:
            self.mpool1 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
            self.softmax1_blocks = nn.Conv3d(in_c, out_c, kernel_size=k_sz, padding='same', dilation=2)
            self.skip1_connection_residual_block = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
            self.softmax2_blocks = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same', dilation=2)
            self.skip2_connection_residual_block = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.softmax3_blocks = nn.Sequential(
                nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same', dilation=8),
                nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same', dilation=8)
            )

            self.interpolation3 = nn.Upsample(scale_factor=2, mode='nearest')
            self.softmax4_blocks = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same', dilation=2)

            self.interpolation2 = nn.Upsample(scale_factor=2, mode='nearest')
            self.softmax5_blocks = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same', dilation=2)

            self.interpolation1 = nn.Upsample(scale_factor=2, mode='nearest')
            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.Sigmoid()
            )

            self.last_blocks = nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding='same')
        
    def forward(self, x, attention=False):
        if self.pool: 
            x = self.pool(x)
        out_trunk = self.conv(x)
        
        if attention==True:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            
            out_interp3 = self.interpolation3(out_softmax3)
            out = torch.add(out_interp3, out_skip2_connection)
            
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4)
            out = torch.add(out_interp2, out_skip1_connection)
            
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5)
            out_softmax6 = self.softmax6_blocks(out_interp1)
            
            out = torch.multiply((1 + out_softmax6), out_trunk)
            out = self.last_blocks(out)
        else:
            out = out_trunk
            
        if self.shortcut: 
            return out + self.shortcut(x)
        else: 
            return out
        
class UpsampleBlock3d(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock3d, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose3d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.Upsample(scale_factor=2, mode='nearest'))
            block.append(nn.Conv3d(in_c, out_c, kernel_size=1))
        elif up_mode == 'pixelshuffle':
            # For 3D, use ConvTranspose3d as equivalent to PixelShuffle
            block.append(nn.ConvTranspose3d(in_c, 4*out_c, kernel_size=1))
            block.append(PixelShuffle3d(upscale_factor=2))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    
class DoubleAttBlock3d(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=True, pool='pool', attention=True, residual=True, causal=True, conv_mode='Conv3d'):
        super(DoubleAttBlock3d, self).__init__()
        
        self.attention = attention
        self.block1 = AttConvBlock3d(in_c, in_c, k_sz=k_sz,
                              shortcut=shortcut, pool=False, attention=False, residual=residual, causal=causal, conv_mode=conv_mode)
        self.block2 = AttConvBlock3d(in_c, out_c, k_sz=k_sz,
                              shortcut=shortcut, pool=pool, attention=self.attention, residual=residual, causal=causal, conv_mode=conv_mode)

    def forward(self, x):
        out = self.block1(x, attention=False)
        out = self.block2(out, attention=self.attention)
        return out

class ConvBridgeBlock3d(torch.nn.Module):
    def __init__(self, channels, k_sz=3, conv_mode='Conv3d'):
        super(ConvBridgeBlock3d, self).__init__()
        self.block = ResBlock3d(channels, channels, k_sz=k_sz, conv_mode=conv_mode)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock3d(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False, skip_conn=True, residual=True, causal=True, conv_mode='Conv3d'):
        super(UpConvBlock3d, self).__init__()
        self.conv_bridge = conv_bridge
        self.skip_conn = skip_conn
        self.up_layer = UpsampleBlock3d(in_c, out_c, up_mode=up_mode)
        self.conv_layer1 = AttConvBlock3d(out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
        self.conv_layer2 = AttConvBlock3d(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock3d(out_c, k_sz=k_sz, conv_mode=conv_mode)

    def forward(self, x, skip=None):
        up = self.up_layer(x)
        up = self.conv_layer1(up, attention=True)
        
        if skip is not None and self.skip_conn:
            if self.conv_bridge:
                skip = self.conv_bridge_layer(skip)
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1) 
            else:
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1)
            out = self.conv_layer2(out, attention=True)
        else:
            out = up
        return out
