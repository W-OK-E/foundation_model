import torch
import torch.nn as nn
from typing import List
from separableconv.nn import SeparableConv2d
from module import MKConv2D, CausalConv2d, DecomConv2D
from ptflops import get_model_complexity_info



def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def get_conv_layer(in_c, out_c, k_sz=3, padding='same', conv_mode='MKConv2D'):
    """
    Factory function to create convolution layer based on conv_mode
    
    Args:
        in_c: Input channels
        out_c: Output channels
        k_sz: Kernel size
        padding: Padding mode ('same' or int)
        conv_mode: Type of convolution ('Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
    
    Returns:
        Convolution layer
    """
    if conv_mode == 'Conv2d':
        return nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=padding)
    elif conv_mode == 'MKConv2D':
        return MKConv2D(in_c, out_c, mode='max', decom_conv='conv', padding=padding)
    elif conv_mode == 'DecomConv2D':
        return DecomConv2D(in_c, out_c, kernel_size=k_sz, padding=padding)
    elif conv_mode == 'SeparableConv2d':
        return SeparableConv2d(in_c, out_c, kernel_size=k_sz, padding=padding if isinstance(padding, int) else 1)
    else:
        raise ValueError(f"Unsupported conv_mode: {conv_mode}. Choose from ['Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d']")

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True, conv_mode='MKConv2D'):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        conv_mode: Type of convolution ('Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
        '''
        super(ConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False

        block = []
        if pool: self.pool = nn.MaxPool1d(kernel_size=2)
        else: self.pool = False

        block.append(get_conv_layer(in_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(get_conv_layer(out_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)
    def forward(self, x):
        if self.pool: x = self.pool(x)
        out = self.block(x)
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, conv_mode='MKConv2D'):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        #self.relu = nn.ReLU(inplace=True)
        self.mish = nn.GELU()
        self.conv1 = nn.Conv2d(in_c, out_c // 4 , 1)
        self.bn2 = nn.BatchNorm2d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv2 = get_conv_layer(out_c // 4, out_c // 4, k_sz=k_sz, padding='same', conv_mode=conv_mode)
        # self.conv3 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[1])
        # self.conv4 = nn.Conv2d(out_c * 4, out_c, 3, padding='same', dilation=dilation[2])
        
        
        #self.conv2 = nn.Conv2d(output_channels/4, output_channels/4, 3, stride, padding = 1, bias = False)
        self.dropout = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm2d(out_c // 4)
        #self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(out_c // 4, out_c, 1, 1, bias = False)
        self.conv6 = nn.Conv2d(in_c, out_c , 1, 1, padding='same', bias = False)
        
    def forward(self, x):
        #residual = x
        #out = self.bn1(x)
        #out = self.mish(out)
        out = self.conv1(x)
        #out = self.bn2(out)
        out = self.mish(out)
        out = self.conv2(out)
        # out2 = self.conv3(out)
        # out3 = self.conv4(out)
        # out = torch.add(torch.add(out1,out2),out3)
        #out =  self.conv2(out)+ self.conv3(out)+ self.conv4(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = self.mish(out)
        out = self.conv5(out)
        
       
        #if (self.input_channels != self.output_channels) or (self.stride !=1 ):
        residual = self.conv6(x)
        out += residual
        return out

class Trunk(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, residual=True, causal=True, conv_mode='MKConv2D'):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        conv_mode: Type of convolution ('Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
        '''
        super(Trunk, self).__init__()
        if residual:   
            self.conv = nn.Sequential(
                ResBlock(in_c, out_c, k_sz=k_sz, conv_mode=conv_mode),
                ResBlock(out_c, out_c, k_sz=k_sz, conv_mode=conv_mode),
            )
        else:
            if causal:
                self.conv =  nn.Sequential(
                    CausalConv2d(in_c, out_c, kernel_size=k_sz),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(),
                    CausalConv2d(out_c, out_c, kernel_size=k_sz),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU()
                )
            else:    
                self.conv = nn.Sequential(
                    get_conv_layer(in_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(),
                    get_conv_layer(out_c, out_c, k_sz=k_sz, padding='same', conv_mode=conv_mode),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU()
                )
    def forward(self, x):
        #print(x.shape)
        out = self.conv(x)
        return out
    


class AttConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool='pool', attention=False, residual=True, causal=True, conv_mode='MKConv2D'):
        '''
        pool_mode can be False (no pooling) or True ('maxpool')
        conv_mode: Type of convolution ('Conv2d', 'MKConv2D', 'DecomConv2D', 'SeparableConv2d')
        '''
        super(AttConvBlock, self).__init__()
        if shortcut==True: self.shortcut = nn.Sequential(conv1x1(in_c, out_c), nn.BatchNorm2d(out_c))
        else: self.shortcut=False
        pad = (k_sz - 1) // 2

        if pool=='pool':
            self.pool = nn.MaxPool2d(kernel_size=2)
        elif pool=='conv':
            self.pool = nn.Conv2d(in_c, in_c, kernel_size = 2, stride=2)
        else:
            self.pool = False

        self.conv = Trunk(in_c, out_c, k_sz=k_sz, residual=residual, causal=causal, conv_mode=conv_mode)
        
        if attention==True:
            self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            #self.softmax1_blocks = DiResBlock(in_c, out_c, dilation= [1,2,4])
            self.softmax1_blocks = nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding='same', dilation= 2)

            self.skip1_connection_residual_block = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            #self.softmax2_blocks = DiResBlock(out_c, out_c, dilation= [2,4,8])
            self.softmax2_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 4)

            self.skip2_connection_residual_block = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')

            self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            self.softmax3_blocks = nn.Sequential(
                nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 8),
                nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 8)
            )

            self.interpolation3 = nn.Upsample(scale_factor=2)

            #self.softmax4_blocks = DiResBlock(out_c, out_c, dilation= [2,4,8])
            self.softmax4_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 4)

            self.interpolation2 = nn.Upsample(scale_factor=2)

            #self.softmax5_blocks = DiResBlock(out_c, out_c, dilation= [1,2,4])
            self.softmax5_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same', dilation= 2)

            self.interpolation1 = nn.Upsample(scale_factor=2)

            self.softmax6_blocks = nn.Sequential(
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c , kernel_size = 1, stride = 1, bias = False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c , kernel_size = 1, stride = 1, bias = False),
                nn.Sigmoid()
            )

            self.last_blocks = nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding='same')
        
    def forward(self, x, attention = False):
        input_size = x.size(-1)
        dilation_rate = input_size // 4
        if self.pool: x = self.pool(x)
        #if self.pool: x = F.conv1d(x, self.pool.weight, bias=self.pool.bias, stride=self.pool.stride,
        #                            padding=self.pool.padding, dilation=dilation_rate)
        #print(x.shape)
        out_trunk = self.conv(x)
        if attention==True:
            out_mpool1 = self.mpool1(x)
            out_softmax1 = self.softmax1_blocks(out_mpool1)
            out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
            out_mpool2 = self.mpool2(out_softmax1)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            #print(out_softmax2.data.shape)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            #
            out_interp3 = self.interpolation3(out_softmax3)
            #print(out_skip2_connection.data.shape)
            #print(out_interp3.data.shape)
            out = torch.add(out_interp3, out_skip2_connection)
            out_softmax4 = self.softmax4_blocks(out)
            out_interp2 = self.interpolation2(out_softmax4)
            out = torch.add(out_interp2, out_skip1_connection)
            out_softmax5 = self.softmax5_blocks(out)
            out_interp1 = self.interpolation1(out_softmax5)
            out_softmax6 = self.softmax6_blocks(out_interp1)
            #print(out_softmax6.shape)
            #print(out_trunk.shape)
            out = torch.multiply((1 + out_softmax6), out_trunk)
            out = self.last_blocks(out)
        else:
            out = out_trunk
        if self.shortcut: return out + self.shortcut(x)
        else: return out
        
        
class UpsampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, up_mode='transp_conv'):
        super(UpsampleBlock, self).__init__()
        block = []
        if up_mode == 'transp_conv':
            block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
        elif up_mode == 'up_conv':
            block.append(nn.Upsample(scale_factor=2))
            block.append(nn.Conv2d(in_c, out_c, kernel_size=1))
        elif up_mode == 'pixelshuffle':
            block.append(nn.Conv2d(in_c, 4*out_c, kernel_size=1))
            block.append(nn.PixelShuffle(2))
        else:
            raise Exception('Upsampling mode not supported')

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
    
class DoubleAttBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, shortcut=True, pool='pool', attention=True, residual=True, causal=True, conv_mode='MKConv2D'):
        super(DoubleAttBlock, self).__init__()
        
        self.attention = attention
        self.block1 = AttConvBlock(in_c, in_c, k_sz=k_sz,
                              shortcut=shortcut, pool=False, attention=False, residual=residual, causal=causal, conv_mode=conv_mode)
        self.block2 = AttConvBlock(in_c, out_c, k_sz=k_sz,
                              shortcut=shortcut, pool=pool, attention=self.attention, residual=residual, causal=causal, conv_mode=conv_mode)

    def forward(self, x):
        out = self.block1(x, attention = False)
        out = self.block2(out, attention = self.attention)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3, conv_mode='MKConv2D'):
        super(ConvBridgeBlock, self).__init__()
        self.block = ResBlock(channels, channels, k_sz=k_sz, conv_mode=conv_mode)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False, skip_conn=True, residual=True, causal=True, conv_mode='MKConv2D'):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge
        self.skip_conn=skip_conn
        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer1 = AttConvBlock(out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
        self.conv_layer2 = AttConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False, attention=True, residual=residual, causal=causal, conv_mode=conv_mode)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz, conv_mode=conv_mode)

    def forward(self, x, skip=None):
        #print(x.shape)
        up = self.up_layer(x)
        #print(up.shape)
        up = self.conv_layer1(up, attention = True)
        #skip = torch.multiply((1 + up), skip)
        if skip is not None and self.skip_conn:
            if self.conv_bridge:
                skip = self.conv_bridge_layer(skip)
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1) 
            else:
                skip = torch.multiply((1 + up), skip)
                out = torch.cat([up, skip], dim=1)
            out = self.conv_layer2(out, attention = True)
        else:
            out=up
        return out
    
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


class EliTNet2D(nn.Module):
    def __init__(self, in_c, n_classes, layers, k_sz=3, up_mode='pixelshuffle', pool='pool', 
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
        super(EliTNet2D, self).__init__()
        #self.n_classes = n_classes
        self.encoder = ELiTNetEncoder(in_c, k_sz, layers, pool=pool, residual=residual, causal=causal, conv_mode=conv_mode)
        #self.latent = nn.Conv1d(widths[-1],latent_features,1)
        self.decoder = ELiTNetDecoder(n_classes, k_sz, layers, up_mode, conv_bridge, shortcut, skip_conn, residual, causal, conv_mode=conv_mode)

    def forward(self, x):
        #x, down_activations = self.encoder(x)
        x= self.encoder(x)
        #print(x.size())
        x = self.decoder(x)#, down_activations)
        return x
    
# def test_model():
#     mod = EliTNet1D(in_c=3, n_classes=2, layers=[4,8,16,24], k_sz=3, up_mode='pixelshuffle', pool='conv', conv_bridge=True, shortcut=True, skip_conn=True, residual= True, causal=False)
#     #mod = ConvNextForTSPrediction(in_channels=1, stem_features=4, depths=[2,2,6,2], widths=[4,8,12,16])
#     #mod = sumnet_auto()
#     #x = torch.randn((1, 3, 512, 512))
#     #mod.cuda()
#     #pred = mod(x)
#     #print(stat(mod, input_size=(3, 512, 512)))
#     #print(summary(mod, input_size=(1, 3, 512, 512)))
#     macs, params = get_model_complexity_info(mod, (3, 1024,1024),
#                                              as_strings=True,
#                                              print_per_layer_stat=False)
#     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#     print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
#     #print(pred.shape)

# test_model()

from torchinfo import summary

# Move model to device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = torch.randn(1, 3, 1024, 1024).to(device)

# Example usage with different conv_mode options:
# conv_mode='Conv2d'          - Standard PyTorch Conv2d
# conv_mode='MKConv2D'        - Multi-Kernel Convolution (default)
# conv_mode='DecomConv2D'     - Decomposed Convolution (vertical + horizontal)
# conv_mode='SeparableConv2d' - Separable Convolution (depthwise + pointwise)

model = EliTNet2D(in_c=3, n_classes=2, layers=[4,8,16,32], k_sz=3, up_mode='pixelshuffle', 
                  pool='conv', conv_bridge=True, shortcut=True, skip_conn=True, 
                  residual=True, causal=False, conv_mode='MKConv2D')

#model = SeparableConv2d(in_channels=3, out_channels=32, kernel_size = 115, stride=1, padding='same')


model.to(device)



# Forward pass to ensure everything is on the same device
output = model(input_tensor)

# Print summary of the model
summary(model, input_size=(1, 3, 1024, 1024))