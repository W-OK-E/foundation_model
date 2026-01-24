import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from separableconv.nn import SeparableConv2d
from utils import generate_series, select_n_elements_with_repeats_and_fallback

# class SeparableConv2d(nn.Module):
#     def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
#         super(SeparableConv2d, self).__init__()
#         self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
#         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out

class DecomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, conv_type='conv', stride=1, padding=1, dilation = 1, bias=True):
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
    def __init__(self, in_channels, out_channels, mode = None, decom_conv = 'conv', stride=1, padding=0, bias=True):
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
        max_kernel_size = (min(input_size) // 3) * 2  # Two-thirds of the smallest input dimension
        
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
    
class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"
