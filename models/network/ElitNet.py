import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, UpConvBlock, ResBlock, DoubleAttBlock

class ElitNet(nn.Module):
    def __init__(
            self, 
            in_channels, 
            num_classes, 
            layers, 
            kernel_sz=3, 
            up_mode='up_conv', 
            conv_bridge=True, 
            shortcut=True
        ):
        """
        A light weight Attention Embedded Unet.
        Args:
            in_channels (int): Number of channels in the input images.
            num_classes (int): Number of classes i.e. number of output channels.
            layers (list): Number of channels in each layer of the encoder. The length of the list defines the number of layers.
            kernel_sz (int): Kernel size for convolutional layers.
            up_mode (str): Upsampling strategy - 'up_conv' for transposed convolution,
            conv_bridge (bool): If True, use a convolutional layer to bridge the encoder and decoder.
            shortcut (bool): If True, use residual connections in the ConvBlocks.
        """

        super(ElitNet, self).__init__()
        self.num_classes = num_classes
        self.first = ConvBlock(in_c=in_channels, out_c=layers[0], k_sz=kernel_sz,
                               shortcut=shortcut, pool=False)

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = DoubleAttBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=kernel_sz,
                              shortcut=shortcut, attention=True)
            self.down_path.append(block)

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            block = UpConvBlock(in_c=reversed_layers[i], out_c=reversed_layers[i + 1], k_sz=kernel_sz,
                                up_mode=up_mode, conv_bridge=conv_bridge, shortcut=shortcut)
            self.up_path.append(block)

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(layers[0], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for i, down in enumerate(self.down_path):
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)      