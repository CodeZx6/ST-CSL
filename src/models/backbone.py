import torch
import torch.nn as nn

def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

class ResidualBlock(nn.Module):
    """Residual unit with BN-ReLU-Conv structure"""
    def __init__(self, channels, use_bn=False):
        super(ResidualBlock, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3(channels, channels)
    
    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class StackedResidualUnits(nn.Module):
    """Stacked residual blocks with skip connections"""
    def __init__(self, channels, num_blocks=12, use_bn=False):
        super(StackedResidualUnits, self).__init__()
        self.blocks = nn.ModuleList([
            self._make_block(channels, use_bn) for _ in range(num_blocks)
        ])
    
    def _make_block(self, channels, use_bn):
        return nn.Sequential(
            ResidualBlock(channels, use_bn),
            ResidualBlock(channels, use_bn)
        )
    
    def forward(self, x):
        for block in self.blocks:
            identity = x
            x = block(x)
            x = x + identity
        return x
