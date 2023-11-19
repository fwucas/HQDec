# ！/usr/bin/python
# Create Time : 2022/4/22 下午3:41
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch.nn as nn
import torch
class Conv3x3(nn.Module):#
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True,groups=1):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3,groups=groups)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out

class ConvBlockGN(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels,groups=1):
        super(ConvBlockGN, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels,groups=groups)
        self.nonlin = nn.ELU(inplace=True)
        self.gn=nn.GroupNorm(num_groups=16,num_channels=out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        out=self.gn(out)
        return out
class Conv_withpad(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1,bias=True,padding=0):
        super(Conv_withpad, self).__init__()
        self.pad=nn.ReflectionPad2d(kernel_size//2)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,groups=groups,bias=bias)
    def forward(self,x):
        return self.conv(self.pad(x))
        
        

class Channel_attention_block_Param(nn.Module):
    def __init__(self,channels=None,gamma=2, beta=1,feat_height=None,feat_width=None):
        '''input:BCHW  kernel_size=3,
        out:BCHW'''
        super(Channel_attention_block_Param, self).__init__()
        if channels is not None:
            import math
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        else:
            kernel_size=3
        assert kernel_size % 2==1
        self.adaptive_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=(kernel_size-1)//2,bias=False)
        self.sigmoid=nn.Sigmoid()

        self.param = nn.Parameter(
            torch.ones((1, 1, feat_height, feat_width), dtype=torch.float32, requires_grad=True),
            requires_grad=True)
    def forward(self,x):
        '''x:bchw'''
        out=self.adaptive_pool(self.param*x)# bc11
        out=out.squeeze(-1) # bc1
        out=out.transpose(-1,-2) #  b1c
        out=self.conv(out) #  b1c
        out=out.transpose(-1,-2) # bc1
        out=out.unsqueeze(-1)# bc11
        out=self.sigmoid(out)
        out=x*out.expand_as(x)
        return out

