import torch
from torch import nn
from typing import Union, List, Tuple
import torch.nn.functional as F

class StackedConvBlocks(nn.Sequential):
    def __init__(self, 
                 num_convs: int, 
                 conv_op: Union[nn.Conv2d, nn.Conv3d], 
                 input_channels: int, 
                 output_channels: int, 
                 kernel_size: Union[int, List[int], Tuple[int, ...]], 
                 stride: Union[int, List[int], Tuple[int, ...]], 
                 norm_op: Union[nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d], 
                 norm_op_kwargs: dict, 
                 nonlin: nn.Module, 
                 nonlin_kwargs: dict,
                 dropout_op: Union[nn.Dropout2d, nn.Dropout3d] = None,
                 dropout_op_kwargs: dict = None):
        super().__init__()
        
        ops = []
        for i in range(num_convs):
            curr_in = input_channels if i == 0 else output_channels
            padding = [(i - 1) // 2 for i in kernel_size] if isinstance(kernel_size, (list, tuple)) else (kernel_size - 1) // 2
            
            ops.append(conv_op(curr_in, output_channels, kernel_size, stride, padding, bias=False))
            if norm_op is not None:
                ops.append(norm_op(output_channels, **norm_op_kwargs))
            if nonlin is not None:
                ops.append(nonlin(**nonlin_kwargs))
            if dropout_op is not None and dropout_op_kwargs is not None:
                ops.append(dropout_op(**dropout_op_kwargs))
            
            # after first conv, stride is always 1
            stride = 1
            
        self.add_module("stacked_convs", nn.Sequential(*ops))

class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Union[nn.Conv2d, nn.Conv3d],
                 kernel_sizes: Union[List[int], List[List[int]]],
                 strides: Union[List[int], List[List[int]]],
                 num_classes: int,
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_conv_per_stage_decoder: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[nn.Dropout2d, nn.Dropout3d] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: nn.Module = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False):
        """
        A flexible UNet implementation inspired by nnU-Net (dynamic-network-architectures).
        """
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}

        self.deep_supervision = deep_supervision

        # Encoder
        self.encoder = nn.ModuleList()
        for s in range(n_stages):
            in_c = input_channels if s == 0 else features_per_stage[s-1]
            out_c = features_per_stage[s]
            num_convs = n_conv_per_stage if isinstance(n_conv_per_stage, int) else n_conv_per_stage[s]
            
            self.encoder.append(StackedConvBlocks(
                num_convs, conv_op, in_c, out_c, kernel_sizes[s], strides[s],
                norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
            ))

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for s in range(n_stages - 2, -1, -1):
            # Upsampler
            # strides[s+1] is the stride used in the encoder stage below this one
            self.upsamplers.append(self.get_upsampler(conv_op, features_per_stage[s+1], features_per_stage[s], strides[s+1]))
            
            # Decoder blocks
            num_convs = num_conv_per_stage_decoder if isinstance(num_conv_per_stage_decoder, int) else num_conv_per_stage_decoder[s]
            self.decoder.append(StackedConvBlocks(
                num_convs, conv_op, 2 * features_per_stage[s], features_per_stage[s], kernel_sizes[s], 1,
                norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
            ))

        # Output heads
        self.seg_heads = nn.ModuleList()
        if deep_supervision:
            for s in range(n_stages - 1):
                self.seg_heads.append(conv_op(features_per_stage[s], num_classes, 1, 1, 0, bias=True))
        else:
            self.seg_heads.append(conv_op(features_per_stage[0], num_classes, 1, 1, 0, bias=True))

    def get_upsampler(self, conv_op, in_c, out_c, stride):
        if conv_op == nn.Conv2d:
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=stride, stride=stride, bias=False)
        else:
            return nn.ConvTranspose3d(in_c, out_c, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x):
        skips = []
        for s in range(len(self.encoder)):
            x = self.encoder[s](x)
            if s < len(self.encoder) - 1:
                skips.append(x)
        
        skips.reverse()
        outputs = []
        
        for s in range(len(self.decoder)):
            x = self.upsamplers[s](x)
            x = torch.cat([x, skips[s]], dim=1)
            x = self.decoder[s](x)
            if self.deep_supervision:
                outputs.append(self.seg_heads[len(self.decoder) - 1 - s](x))
            elif s == len(self.decoder) - 1:
                outputs.append(self.seg_heads[0](x))
                
        if self.deep_supervision:
            outputs.reverse()
            return outputs
        else:
            return outputs[0]

class DynamicUNet(PlainConvUNet):
    def __init__(self, 
                 spatial_dims: int,
                 in_channels: int,
                 num_classes: int,
                 depth: int = 5,
                 base_channels: int = 32,
                 n_conv_per_stage: int = 2,
                 n_conv_per_stage_decoder: int = 2,
                 deep_supervision: bool = False,
                 **kwargs):
        conv_op, norm_op, dropout_op = get_network_factory(spatial_dims)
        
        # Simple heuristic for features, kernel sizes and strides
        features = [base_channels * (2**i) for i in range(depth)]
        # Cap features at 512 like nnU-Net often does
        features = [min(f, 512) for f in features]
        
        kernel_sizes = [[3] * spatial_dims] * depth
        strides = [[1] * spatial_dims] + [[2] * spatial_dims] * (depth - 1)
        
        super().__init__(
            input_channels=in_channels,
            n_stages=depth,
            features_per_stage=features,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            num_conv_per_stage_decoder=n_conv_per_stage_decoder,
            norm_op=norm_op,
            dropout_op=dropout_op,
            deep_supervision=deep_supervision,
            **kwargs
        )

def get_network_factory(spatial_dims: int = 2):
    if spatial_dims == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.Dropout2d
    else:
        return nn.Conv3d, nn.BatchNorm3d, nn.Dropout3d
