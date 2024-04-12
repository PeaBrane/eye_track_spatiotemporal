import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
    f'{category.__name__}: {message}\n'

class CausalGroupNorm(nn.GroupNorm):
    """A GroupNorm that does not use temporal statistics, to ensure causality
    """
    def __init__(self, num_groups, num_channels, **kwargs):
        super().__init__(num_groups, num_channels, **kwargs)
        
    def forward(self, input):
        x = input.moveaxis(1, 2)  # (B, T, C, H, W)
        x_shape = x.shape
        x = x.flatten(0, 1)  # (B * T, C, H, W)
        x = super().forward(x).reshape(x_shape)
        return x.moveaxis(1, 2)  # (B, C, T, H, W)


act_layer = lambda: nn.ReLU()
bn_block = lambda features: nn.Sequential(nn.BatchNorm3d(features), act_layer())
gn_block = lambda features: nn.Sequential(CausalGroupNorm(4, features), act_layer())
pw_conv = lambda in_channels, out_channels: nn.Conv3d(in_channels, out_channels, 1, bias=False)


class SpatialBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 depthwise=False, 
                 kernel_size=1,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        kernel = (kernel_size,3,3)
        self.kernel_size = kernel_size
        self.full_conv3d = full_conv3d
        self.norms = norms
        self.streaming_mode = False
        self.fifo = None  # for streaming inference

        if self.norms=='all_gn':
            norm_block = gn_block
        else :
            norm_block = bn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, (1, 2, 2), (0, 1, 1), groups=in_channels, bias=False), 
                norm_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (0, 1, 1), bias=False), 
                norm_block(out_channels), 
            )
        
    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.full_conv3d: 
            if self.streaming_mode:
                return self._streaming_forward(input)
            input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
            return self.block(input)
        else:         
            return self.block(input)
            
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)


class TemporalBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 depthwise=False,
                 full_conv3d=False, 
                 norms='mixed'):
        super().__init__()
        assert out_channels % 4 == 0  # needed for group norm to work
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.norms = norms
        kernel = (kernel_size,3,3) if full_conv3d else (kernel_size,1,1)
        
        self.streaming_mode = False
        self.fifo = None  # for streaming inference
        
        if self.norms=='mixed':
            norm1_block = bn_block
            norm2_block = gn_block
        elif self.norms=='all_bn':
            norm1_block = bn_block
            norm2_block = bn_block
        elif self.norms=='all_gn':
            norm1_block = gn_block
            norm2_block = gn_block

        if depthwise:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel, groups=in_channels, bias=False), 
                norm1_block(in_channels), 
                pw_conv(in_channels, out_channels), 
                norm2_block(out_channels), 
            )
            
        else:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel, bias=False), 
                norm2_block(out_channels), 
            )

    def streaming(self, enabled=True):
        if enabled:
            assert not self.training, "Can only use streaming mode during evaluation."
        self.streaming_mode = enabled
        
    def reset_memory(self):
        self.fifo = None
    
    def forward(self, input):
        if self.streaming_mode:
            return self._streaming_forward(input)
                  
        input = F.pad(input, (0, 0, 0, 0, self.kernel_size - 1, 0))
        return self.block(input)
    
    def _streaming_forward(self, input):
        if self.fifo is None:
            self.fifo = torch.zeros(*input.shape[:2], self.kernel_size, *input.shape[3:]).type_as(input)
        self.fifo = torch.cat([self.fifo[:, :, 1:], input], dim=2)
        return self.block(self.fifo)
        

class TennSt(nn.Module):
    def __init__(
        self, 
        channels, 
        t_kernel_size, 
        n_depthwise_layers, 
        detector_head, 
        detector_depthwise, 
        full_conv3d=False,
        norms='mixed',
    ):
        super().__init__()
        self.detector = detector_head
        
        depthwises = [False] * (10 - n_depthwise_layers) + [True] * n_depthwise_layers
        temporals = [True, False] * 5
        
        self.backbone = nn.Sequential()
        for i in range(len(depthwises)):
            in_channels, out_channels = channels[i], channels[i+1]
            depthwise = depthwises[i]
            temporal = temporals[i]
            
            if temporal:
                self.backbone.append(TemporalBlock(in_channels, out_channels, 
                                                   kernel_size=t_kernel_size, depthwise=depthwise,
                                                   full_conv3d=full_conv3d, norms=norms))
            else:
                self.backbone.append(SpatialBlock(in_channels, out_channels, depthwise=depthwise,
                                                  full_conv3d=full_conv3d,
                                                  kernel_size=t_kernel_size if full_conv3d else 1,
                                                  norms=norms))
        
        if detector_head:
            self.head = nn.Sequential(
                TemporalBlock(channels[-1], channels[-1], t_kernel_size, depthwise=detector_depthwise), 
                nn.Conv3d(channels[-1], channels[-1], (1, 3, 3), (1, 1, 1), (0, 1, 1)), 
                act_layer(), 
                nn.Conv3d(channels[-1], 3, 1), 
            )
        else:
            self.head = nn.Sequential(
                nn.Conv1d(channels[-1], channels[-1], 1), 
                act_layer(), 
                nn.Conv1d(channels[-1], 2, 1), 
            )
    
    def streaming(self, enabled=True):
        if enabled:
            warnings.warn("You have enabled the streaming mode of the network. It is expected, but not checked, that the input will be of shape (batch, 1, H, W).")
        for name, module in self.named_modules():
            if name and hasattr(module, 'streaming'):
                module.streaming(enabled)
                
    def reset_memory(self):
        for name, module in self.named_modules():
            if name and hasattr(module, 'reset_memory'):
                module.reset_memory()
         
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.detector:
            return self.head((self.backbone(input)))
        else:
            return self.head(self.backbone(input).mean((-2, -1)))
        