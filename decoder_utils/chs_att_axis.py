# ！/usr/bin/python
# Create Time : 2022/9/23 上午10:37
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch
import torch.nn as nn
from timm.models.layers.weight_init import trunc_normal_
from .general_blocks import ConvBlockGN,Channel_attention_block_Param
from .downsample_blocks import Conv2d_zero_pad

class SE(nn.Module):
    def __init__(self,in_chs,reduce_rate=0.25):
        super(SE, self).__init__()
        mid_chs=int(in_chs*reduce_rate)
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.conv_reduce=nn.Conv2d(in_channels=in_chs,out_channels=mid_chs,kernel_size=1,bias=True)
        self.act1=nn.SiLU(True)
        self.conv_expand=nn.Conv2d(in_channels=mid_chs,out_channels=in_chs,kernel_size=1,bias=True)
        self.act2=nn.Sigmoid()
    def forward(self,x):
        out=self.avg(x)
        out=self.act1(self.conv_reduce(out))
        out=self.act2(self.conv_expand(out))
        out=x*out
        return out

class DownPixelH_Pos_groups(nn.Module):
    def __init__(self,in_chs,stride=2,feat_height=None,feat_width=None):
        super(DownPixelH_Pos_groups, self).__init__()
        self.stride=stride #b,c,h,w)
        self.pos_h=nn.Parameter(torch.zeros(1,1,1,stride,1))
        trunc_normal_(self.pos_h,std=.02)

        self.conv1_h=ConvBlockGN(in_channels=in_chs*stride,out_channels=in_chs*stride,groups=in_chs*stride)#,groups=in_chs*stride
        self.se1_h = SE(in_chs=in_chs * stride , reduce_rate=1. / 16)
        self.conv1_hpw = Conv2d_zero_pad(in_channels=in_chs * stride, out_channels=in_chs * stride,
                                        kernel_size=1)


        self.conv2_h=ConvBlockGN(in_channels=in_chs*stride,out_channels=in_chs*stride,groups=in_chs*stride)
        self.se2_h = SE(in_chs=in_chs * stride , reduce_rate=1. / 16)

        self.conv2_hpw = Conv2d_zero_pad(in_channels=in_chs * stride , out_channels=in_chs * stride ,
                                        kernel_size=1)

        self.chs_atth = Channel_attention_block_Param(channels=None, feat_height=feat_height//stride, feat_width=feat_width)

        self.conv3_h=Conv2d_zero_pad(in_channels=in_chs*stride,out_channels=in_chs,kernel_size=3)




    def forward(self,x):
        b,c,h,w=x.shape
        out=x.contiguous().view(b, c, h//self.stride, self.stride, w)

        out = out+self.pos_h
        out=out.permute(0, 1,3,2,4 ).contiguous().view(b,c*self.stride,h//self.stride,w)

        out=self.conv1_h(out)
        out=self.se1_h(out)
        out=self.conv1_hpw(out)

        out=self.conv2_h(out)
        out=self.se2_h(out)
        out=self.conv2_hpw(out)

        out=self.chs_atth(out)

        out=self.conv3_h(out)

        return out

class DownPixelW_Pos_groups(nn.Module):
    def __init__(self,in_chs,stride=2,feat_height=None,feat_width=None):
        super(DownPixelW_Pos_groups, self).__init__()
        self.stride=stride
        self.pos_w=nn.Parameter(torch.zeros(1,1,1,1,stride))
        trunc_normal_(self.pos_w, std=.02)

        self.conv1_w = ConvBlockGN(in_channels=in_chs * stride, out_channels=in_chs * stride,groups=in_chs*stride)
        self.se1_w = SE(in_chs=in_chs * stride, reduce_rate=1. / 16)
        self.conv1_wpw = Conv2d_zero_pad(in_channels=in_chs * stride, out_channels=in_chs * stride,
                                         kernel_size=1)



        self.conv2_w = ConvBlockGN(in_channels=in_chs * stride, out_channels=in_chs * stride,groups=in_chs*stride)
        self.se2_w = SE(in_chs=in_chs * stride, reduce_rate=1. / 16)

        self.chs_attw = Channel_attention_block_Param(channels=None, feat_height=feat_height,
                                                      feat_width=feat_width//stride)
        self.conv2_wpw = Conv2d_zero_pad(in_channels=in_chs * stride, out_channels=in_chs * stride,
                                         kernel_size=1)

        self.conv3_w = Conv2d_zero_pad(in_channels=in_chs * stride, out_channels=in_chs, kernel_size=3)





    def forward(self,x):
        b,c,h,w=x.shape
        out=x.contiguous().view(b, c, h, w//self.stride, self.stride)
        out = out+ self.pos_w
        out = out.permute(0,1,4,2,3).contiguous().view(b,c*self.stride,h,w//self.stride)

        out=self.conv1_w(out)
        out=self.se1_w(out)
        out=self.conv1_wpw(out)

        out=self.conv2_w(out)
        out=self.se2_w(out)
        out=self.conv2_wpw(out)

        out=self.chs_attw(out)

        out=self.conv3_w(out)

        return out


class Chs_Att_Axis_Down_groups(nn.Module):
    def __init__(self,in_chs,stride=2,feat_height=None,feat_width=None):
        super(Chs_Att_Axis_Down_groups, self).__init__()
        self.axis_h=DownPixelH_Pos_groups(in_chs=in_chs,stride=stride,feat_height=feat_height,feat_width=feat_width) #b,c,h//stride,w
        self.axis_w=DownPixelW_Pos_groups(in_chs=in_chs,stride=stride,feat_height=feat_height//stride,feat_width=feat_width)# b,c,h//stride,w//stride

    def forward(self,x):
        out=self.axis_h(x)
        out=self.axis_w(out)
        return out


class AdaChs_Att_Axis_Down_groups(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,feat_height=None,feat_width=None):
        super(AdaChs_Att_Axis_Down_groups, self).__init__()
        bs=1
        self.param = nn.Parameter(
            torch.zeros((bs, 1, feat_height // stride, feat_width // stride), dtype=torch.float32,requires_grad=True),requires_grad=True)

        self.maxdownsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=in_chs)
        self.bn2 = nn.BatchNorm2d(num_features=in_chs)
        self.act = nn.ELU(True)

        self.chs_axis_down=Chs_Att_Axis_Down_groups(in_chs=in_chs,stride=stride,feat_height=feat_height,feat_width=feat_width)


    def forward(self,x):
        max_pool_x = self.maxdownsample(x)
        max_pool_x = self.bn1(max_pool_x)

        chs_axis=self.chs_axis_down(x)
        chs_axis=self.bn2(chs_axis)

        out = self.param * max_pool_x + chs_axis
        out = self.act(out)

        return out





