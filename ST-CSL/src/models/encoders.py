import torch
import torch.nn as nn
from collections import OrderedDict
from .backbone import conv3x3, StackedResidualUnits

class BaseEncoder(nn.Module):
    """Base encoder with residual architecture"""
    def __init__(self, in_channels=2, base_channels=64, num_res_blocks=12):
        super(BaseEncoder, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', conv3x3(in_channels, base_channels)),
            ('res_units', StackedResidualUnits(base_channels, num_res_blocks // 2)),
            ('activation', nn.ReLU(inplace=True)),
            ('conv_out', conv3x3(base_channels, in_channels))
        ]))
    
    def forward(self, x):
        return self.encoder(x)

class ClosenessEncoder(BaseEncoder):
    """Temporal closeness dependency encoder"""
    def __init__(self, in_channels=2, base_channels=64):
        super(ClosenessEncoder, self).__init__(in_channels, base_channels, 12)

class PeriodEncoder(BaseEncoder):
    """Periodic dependency encoder"""
    def __init__(self, in_channels=2, base_channels=64):
        super(PeriodEncoder, self).__init__(in_channels, base_channels, 12)

class TrendEncoder(BaseEncoder):
    """Trend dependency encoder"""
    def __init__(self, in_channels=2, base_channels=64):
        super(TrendEncoder, self).__init__(in_channels, base_channels, 12)
