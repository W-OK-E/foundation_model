import torch.nn as nn

from typing import List
from .blocksv2 import ConvBlock,DoubleAttBlock,UpConvBlock
class ELiTNetEncoder(nn.Module):
    def __init__(
        self,
        in_c: int,
        k_sz: int,
        layers: List[int],
        shortcut: bool = True,
        pool='pool',
        residual=True,
        causal=True,
        conv_mode='MKConv2D'
    ):
        super().__init__()
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz,
                               shortcut=shortcut, pool=False, conv_mode=conv_mode)
        
        # in_out_widths = list(zip(layers, layers[1:]))
        # create drop paths probabilities (one for each stage)
        # drop_probs = [x.item() for x in torch.linspace(0, drop_p, sum(depths))]
        
        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            if i <= 7:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
            else:
                block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz,
                                shortcut=shortcut, pool=pool, attention=False, residual=residual, causal=causal, conv_mode=conv_mode)
            self.down_path.append(block)
        
    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        return x#, down_activations

class ELiTNetDecoder(nn.Module):
    def __init__(
        self,
        n_classes: int,
        k_sz: int,
        #latent_features: int,
        layers: List[int],
        up_mode='up_conv',
        conv_bridge: bool = True,
        shortcut: bool = True,
        skip_conn: bool = True,
        residual=True,
        causal=True,
        conv_mode='MKConv2D'
    ):
        super().__init__()
        
        self.up_path = nn.ModuleList()
        #print(layers)
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=k_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut, skip_conn=skip_conn, 
                                residual=residual, causal=causal, conv_mode=conv_mode)
            self.up_path.append(block)
            
        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)
        

    def forward(self, x):#, down_activations):
        for i, up in enumerate(self.up_path):
            x = up(x)#, down_activations[i])
        return self.final(x)


        
class ElitNet(nn.Module):
    def __init__(self, in_channels, num_classes, layers, kernel_sz=3, up_mode='pixelshuffle', pool='pool', 
                 conv_bridge=True, shortcut=True, skip_conn=True, residual=True, causal=True, conv_mode='MKConv2D'):
        """
        ELiTNet 2D Architecture
        
        Args:
            in_c: Number of input channels
            n_classes: Number of output classes
            layers: List of channel dimensions for each layer
            k_sz: Kernel size
            up_mode: Upsampling mode ('pixelshuffle', 'up_conv', 'transp_conv')
            pool: Pooling mode ('pool', 'conv', False)
            conv_bridge: Whether to use convolutional bridge in decoder
            shortcut: Whether to use shortcut connections
            skip_conn: Whether to use skip connections
            residual: Whether to use residual blocks
            causal: Whether to use causal convolutions
            conv_mode: Type of convolution to use ('Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
        """
        super(ElitNet, self).__init__()
        print("="*70)
        print("Initializing ")
        #self.n_classes = n_classes
        self.encoder = ELiTNetEncoder(in_channels, num_classes, layers, pool=pool, residual=residual, causal=causal, conv_mode=conv_mode)
        #self.latent = nn.Conv1d(widths[-1],latent_features,1)
        self.decoder = ELiTNetDecoder(num_classes, kernel_sz, layers, up_mode, conv_bridge, shortcut, skip_conn, residual, causal, conv_mode=conv_mode)

    def forward(self, x):
        #x, down_activations = self.encoder(x)
        x= self.encoder(x)
        #print(x.size())
        x = self.decoder(x)#, down_activations)
        return x