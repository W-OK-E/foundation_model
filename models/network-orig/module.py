import sys
sys.path.append('/mnt/data/omkumar/foundation_phase1/models/network')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from separableconv.nn import SeparableConv2d

import sys
sys.path.append("/mnt/data/omkumar/foundation_model/shared_foundation/models/network")
from utils import generate_series, select_n_elements_with_repeats_and_fallback

class DecomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type='conv', stride=1, padding='same', dilation = 1, bias=True):
        super(DecomConv2D, self).__init__()
        conv_layers = {
            'conv': nn.Conv2d,
            'sep': SeparableConv2d,  # Replace with actual SeparableConv2d implementation
            'dwconv': nn.Conv2d,  # Depthwise convolution will be handled by setting groups=in_channels
        }
        
        if conv_type not in conv_layers:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        
        ConvLayer = conv_layers[conv_type]
        
        if padding == 'same':
            padding_vertical = padding
            padding_horizontal = padding
        else:
            padding_vertical = (padding, 0)
            padding_horizontal = (0, padding)
        
        if conv_type == 'dwconv':
            self.vertical_conv = ConvLayer(in_channels, in_channels, (kernel_size, 1), stride=stride, padding=padding_vertical, bias=bias, groups=in_channels)
            self.horizontal_conv = ConvLayer(in_channels, in_channels, (1, kernel_size), stride=stride, padding=padding_horizontal, bias=bias, groups=in_channels)
        else:
            self.vertical_conv = ConvLayer(in_channels, out_channels, (kernel_size, 1), stride=stride, padding=padding_vertical, dilation = dilation, bias=bias)
            self.horizontal_conv = ConvLayer(in_channels, out_channels, (1, kernel_size), stride=stride, padding=padding_horizontal, dilation = dilation, bias=bias)

    def forward(self, x):
        vertical_output = self.vertical_conv(x)
        horizontal_output = self.horizontal_conv(x)
        return vertical_output * horizontal_output
    
    
class MKConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mode = None, decom_conv = 'conv', stride=1, padding='same', bias=True):
        super(MKConv2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.decom_conv = decom_conv
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.convs = nn.ModuleList()
        self.kernel_sizes = []  # Store kernel sizes for inspection
    
    def _initialize_convs(self, input_size):
        # Determine the maximum allowable kernel size based on input dimensions
        # Cap at 31 to avoid excessive memory usage and slow convolutions
        max_kernel_size = min(31, (min(input_size) // 3) * 2) 
        
        # Generate kernel sizes within the allowed range
        possible_kernel_sizes = generate_series(3, max_kernel_size)  # Odd sizes: 1, 3, 5, ..., max_kernel_size

        selected_kernel_sizes = select_n_elements_with_repeats_and_fallback(possible_kernel_sizes, self.out_channels, mode=self.mode)

        self.kernel_sizes = selected_kernel_sizes

        
        self.convs = nn.ModuleList([
            self._get_conv_layer(i) for i in range(self.out_channels)
        ])
        
        
    def _get_conv_layer(self, i):       
        if self.kernel_sizes[i] > 10:
            return DecomConv2D(self.in_channels, 1, kernel_size=self.kernel_sizes[i], conv_type = self.decom_conv, stride=self.stride, padding=self.padding, bias=self.bias)
        else:
            return nn.Conv2d(self.in_channels, 1, kernel_size=self.kernel_sizes[i], stride=self.stride, padding=self.padding, bias=self.bias)
    
    def forward(self, x):
        # Initialize convolution layers dynamically based on input size during the first forward pass
        if len(self.convs) == 0:
            input_size = x.shape[2:]  # Get height and width from the input tensor
            self._initialize_convs(input_size)
            
        device = x.device
        self.convs = self.convs.to(device)
        
        # Perform convolution for each filter
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Combine the outputs along the channel dimension
        output = torch.cat(conv_outputs, dim=1)
        
        return output
    
    def get_kernel_sizes(self):
        """Returns the kernel sizes of all filters."""
        return self.kernel_sizes
    
class CausalConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        if padding is None:
            padding = [int((kernel_size[i] -1) * dilation[i]) for i in range(len(kernel_size))]
        else:
           padding = padding * 2
        self.left_padding = _pair(padding)
        super().__init__(in_channels, out_channels, kernel_size,
                                           stride=stride, padding=0, dilation=dilation,
                                           groups=groups, bias=bias)
    def forward(self, inputs):
        inputs = F.pad(inputs, (self.left_padding[1], 0, self.left_padding[0], 0))
        output = super().forward(inputs)
        return output