# ！/usr/bin/python
# Create Time : 2022/9/23 下午4:30
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch.nn as nn
import torch
from downsample_blocks import Conv2d_zero_pad
from timm.models.layers.weight_init import trunc_normal_
from general_blocks import Channel_attention_block_Param



class DownPixelH_Pos(nn.Module):
    def __init__(self,in_chs,stride=2,feat_height=None,feat_width=None):
        super(DownPixelH_Pos, self).__init__()
        self.stride=stride #b,c,h,w)
        self.pos_h=nn.Parameter(torch.zeros(1,1,1,stride,1))
        trunc_normal_(self.pos_h,std=.02)

        self.chs_atth = Channel_attention_block_Param(channels=None, feat_height=feat_height//stride, feat_width=feat_width)

        self.conv3_h=Conv2d_zero_pad(in_channels=in_chs*stride,out_channels=in_chs,kernel_size=1)

        self.conv4_h=Conv2d_zero_pad(in_channels=in_chs,out_channels=in_chs,kernel_size=3)
        self.conv5_h=Conv2d_zero_pad(in_channels=in_chs,out_channels=in_chs,kernel_size=3)


    def forward(self,x):
        b,c,h,w=x.shape
        out=x.contiguous().view(b, c, h//self.stride, self.stride, w)

        out = out+self.pos_h
        out=out.permute(0, 1,3,2,4 ).contiguous().view(b,c*self.stride,h//self.stride,w)

        out=self.chs_atth(out)

        out=self.conv3_h(out)
        out=self.conv4_h(out)

        out=self.conv5_h(out)
        return out

class DownPixelW_Pos(nn.Module):
    def __init__(self,in_chs,stride=2,feat_height=None,feat_width=None):
        super(DownPixelW_Pos, self).__init__()
        self.stride=stride
        self.pos_w=nn.Parameter(torch.zeros(1,1,1,1,stride))
        trunc_normal_(self.pos_w, std=.02)

        self.chs_attw = Channel_attention_block_Param(channels=None, feat_height=feat_height,
                                                      feat_width=feat_width//stride)

        self.conv3_w = Conv2d_zero_pad(in_channels=in_chs * stride, out_channels=in_chs, kernel_size=1)

        self.conv4_w = Conv2d_zero_pad(in_channels=in_chs, out_channels=in_chs, kernel_size=3)

        self.conv5_w=Conv2d_zero_pad(in_channels=in_chs, out_channels=in_chs, kernel_size=3)



    def forward(self,x):
        b,c,h,w=x.shape
        out=x.contiguous().view(b, c, h, w//self.stride, self.stride)
        out = out+ self.pos_w
        out = out.permute(0,1,4,2,3).contiguous().view(b,c*self.stride,h,w//self.stride)

        out=self.chs_attw(out)

        out=self.conv3_w(out)
        out=self.conv4_w(out)

        out=self.conv5_w(out)
        return out



class Patch_Block(nn.Module):
    def __init__(self,in_chs,embed_dim,patch_height,patch_width,feat_height,feat_width):
        super(Patch_Block, self).__init__()

        self.downpixelh=DownPixelH_Pos(in_chs=in_chs,stride=patch_height,feat_height=feat_height,feat_width=feat_width) #bc,h/s,w

        self.downpixelw=DownPixelW_Pos(in_chs=in_chs,stride=patch_width,feat_height=feat_height//patch_height,feat_width=feat_width)# b,c,h/s,w/s

        self.conv1=Conv2d_zero_pad(in_channels=in_chs,out_channels=embed_dim,kernel_size=3)

    def forward(self,x):
        out=self.downpixelh(x)
        out=self.downpixelw(out)
        out=self.conv1(out)

        out=out.flatten(2).transpose(1, 2)
        return out




class Patch_Embed(nn.Module):
    def __init__(self,in_chs,embed_dim,patch_height,patch_width,feat_height,feat_width):
        super(Patch_Embed, self).__init__()
        assert feat_height % patch_height==0 and feat_width % patch_width ==0

        self.num_patches=(feat_width//patch_width)*(feat_height//patch_height)
        self.proj_h = nn.Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=(patch_height, 1),
                              stride=(patch_height, 1),groups=in_chs)

        self.proj_w = nn.Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=(1, patch_width),
                                stride=(1, patch_width),groups=in_chs)

        self.act1=nn.ELU(True)
        self.act2=nn.ELU(True)


    def forward(self,x):
        x=self.act1(self.proj_h(x))
        x=self.act2(self.proj_w(x))
        x=x.flatten(2).transpose(1, 2)

        return x


