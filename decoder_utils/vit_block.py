# Author      :  wangfei
# Create time : 2021/8/12 上午8:55
# Email       : fei.wang2@siat.ac.cn

import torch
import torch.nn as nn
from functools import partial

from timm.models.layers.weight_init import trunc_normal_

from .downsample_blocks import Conv2d_zero_pad
class Channel_attention_block(nn.Module):
    def __init__(self,channels=None,gamma=2, beta=1):
        '''input:BCHW  kernel_size=3,
        out:BCHW'''
        super(Channel_attention_block, self).__init__()
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
    def forward(self,x):
        '''x:bchw'''
        out=self.adaptive_pool(x)# bc11
        out=out.squeeze(-1) # bc1
        out=out.transpose(-1,-2) #  b1c
        out=self.conv(out) #  b1c
        out=out.transpose(-1,-2) # bc1
        out=out.unsqueeze(-1)# bc11
        out=self.sigmoid(out)
        out=x*out.expand_as(x)
        return out
        

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor#
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DownPixelAlpha(nn.Module):
    '''
    input(b,c,h,w)
    return: (b,c*(stride**2),h/stride,w/stride)
    '''
    def __init__(self,stride_h=2,stride_w=2):
        super(DownPixelAlpha, self).__init__()
        self.stride_h=stride_h
        self.stride_w=stride_w
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.contiguous().view(b, c, h//self.stride_h, self.stride_h, w//self.stride_w, self.stride_w)
        x=x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c*(self.stride_h*self.stride_w), h//self.stride_h, w//self.stride_w)
        return x

class Patch_Embed(nn.Module):
    def __init__(self,in_chs,embed_dim,patch_height,patch_width,feat_height,feat_width):
        super(Patch_Embed, self).__init__()
        assert feat_height % patch_height==0 and feat_width % patch_width ==0

        self.num_patches=(feat_width//patch_width)*(feat_height//patch_height)
        self.proj=nn.Conv2d(in_channels=in_chs,out_channels=in_chs,kernel_size=(patch_height,patch_width),stride=(patch_height,patch_width),groups=in_chs)
        self.conv1x1=Conv2d_zero_pad(in_channels=in_chs,out_channels=embed_dim,kernel_size=1)

    def forward(self,x):
        x=self.proj(x)
        x=self.conv1x1(x)
        x=x.flatten(2).transpose(1, 2)  #

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   #
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GetFeatShape(nn.Module):
    def __init__(self):
        super(GetFeatShape, self).__init__()
    def forward(self,x):
        return x.shape


class TransToCNN(nn.Module):
    def __init__(self,in_chs,out_chs,before_embed_height,before_embed_width,patch_height,patch_width,
                 norm_layer=None,act_layer=None):
        super(TransToCNN, self).__init__()
        self.before_embed_height=before_embed_height
        self.before_embed_width=before_embed_width
        self.patch_height=patch_height
        self.patch_width=patch_width

        self.proj=nn.Conv2d(in_channels=in_chs,out_channels=out_chs,kernel_size=1)
        self.norm=norm_layer(out_chs) if norm_layer is not None else None
        self.act=act_layer() if act_layer is not None else None

        self.upsample=nn.ConvTranspose2d(in_channels=out_chs,out_channels=out_chs,
                                         kernel_size=(patch_height,patch_width),
                                         stride=(patch_height,patch_width),groups=out_chs)#,groups=out_chs
        self.conv1x1=Conv2d_zero_pad(in_channels=out_chs,out_channels=out_chs,kernel_size=1) #Conv2d_zero_pad
        
    def forward(self,x):
        b,_,c=x.shape
        x=x.transpose(1, 2).reshape(b, c,
                                    int(self.before_embed_height//self.patch_height),
                                    int(self.before_embed_width//self.patch_width))
        x=self.proj(x)
        if self.norm is not None:
            x=self.norm(x)
        if self.act is not None:
            x=self.act(x)

        x=self.upsample(x)
        x=self.conv1x1(x)
        
        return x



class TransBlock(nn.Module):
    def __init__(self,in_chs,embed_dim,patch_height,patch_width,feat_height,feat_width,
                 depth=2,num_heads=8,mlp_ratio=4,qkv_bias=True,drop_path_rate=0.1,
                 attn_drop_rate=0.1,drop_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,qk_scale=None):
        super(TransBlock, self).__init__()
        self.patch_embed=Patch_Embed(in_chs=in_chs,embed_dim=embed_dim,
                                     patch_height=patch_height,patch_width=patch_width,
                                     feat_height=feat_height,feat_width=feat_width)
        self.num_patches=self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))#
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)


        self.transtocnn=TransToCNN(in_chs=embed_dim,out_chs=in_chs,
                                   before_embed_height=feat_height,before_embed_width=feat_width,
                                   patch_height=patch_height,patch_width=patch_width,norm_layer=None,act_layer=None)

    def forward(self,x):
        x=self.patch_embed(x)
        x=self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x=self.transtocnn(x)
        return x

class Conv_withpad(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,groups=1,bias=True,padding=0):
        super(Conv_withpad, self).__init__()
        self.pad=nn.ReflectionPad2d(kernel_size//2)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,groups=groups,bias=bias)
    def forward(self,x):
        return self.conv(self.pad(x))

class Refine_layer_v3(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=8,patch_width=26,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,qkv_bias=True,
                 drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None):
        super(Refine_layer_v3, self).__init__()
        self.conv1 = conv_layer(in_channels=chs, out_channels=chs, kernel_size=3, padding=1, bias=True)
        self.conv2 = conv_layer(in_channels=chs, out_channels=chs, kernel_size=3, padding=1, bias=True)
        self.act1 = act_layer(True)
        self.act2 = act_layer(True)


        self.transblock=TransBlock(in_chs=chs,embed_dim=embed_dim,patch_height=patch_height,patch_width=patch_width,
                                   feat_height=feat_height,feat_width=feat_width,
                 depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop_path_rate=drop_path_rate,
                 attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,norm_layer=norm_layer,
                 act_layer=act_layer_trans,qk_scale=qk_scale)
        self.channel_att_trans = Channel_attention_block(channels=chs)
        self.channel_att_conv=Channel_attention_block(channels=chs)
    def forward(self,x):
        #b,c,h,w=x.shape
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out=self.channel_att_conv(out)

        transblock_out=self.transblock(x)
        transblock_out=self.channel_att_trans(transblock_out)

        return out+transblock_out+x

class Refine_layer_v4(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,qkv_bias=True,
                 drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None,expand_rate=4):
        super(Refine_layer_v4, self).__init__()
        expand_rate=expand_rate
        self.conv1 = conv_layer(in_channels=chs, out_channels=int(chs/expand_rate), kernel_size=1, bias=True)
        self.conv2 = conv_layer(in_channels=int(chs/expand_rate), out_channels=int(chs/expand_rate), kernel_size=3, padding=1, bias=True)
        self.conv3=conv_layer(in_channels=int(chs/expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.act1 = act_layer(True)
        self.act2 = act_layer(True)


        self.transblock=TransBlock(in_chs=int(chs/expand_rate),embed_dim=embed_dim,patch_height=patch_height,patch_width=patch_width,
                                   feat_height=feat_height,feat_width=feat_width,
                 depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop_path_rate=drop_path_rate,
                 attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,norm_layer=norm_layer,
                 act_layer=act_layer_trans,qk_scale=qk_scale)
        self.conv_transblock = conv_layer(in_channels=int(chs / expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.channel_att_trans = Channel_attention_block(channels=chs)
        self.channel_att_conv=Channel_attention_block(channels=chs)
    def forward(self,x):
        #b,c,h,w=x.shape
        out_conv1 = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out_conv1))
        out=self.act1(self.conv3(out))
        out=self.channel_att_conv(out)

        transblock_out=self.transblock(out_conv1)
        transblock_out=self.act1(self.conv_transblock(transblock_out))
        transblock_out=self.channel_att_trans(transblock_out)



        return out+transblock_out+x






