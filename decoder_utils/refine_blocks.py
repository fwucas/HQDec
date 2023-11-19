# ！/usr/bin/python
# Create Time : 2022/4/22 下午3:58
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch.nn as nn
from functools import partial
from .general_blocks import Conv_withpad
from .vit_block import TransBlock
from .general_blocks import Channel_attention_block_Param as Channel_attention_block
import torch
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

class Ablation_Refine_layer(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,qkv_bias=True,
                 drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None,expand_rate=4):
        super(Ablation_Refine_layer, self).__init__()

    def forward(self,x):

        return x


class ResBlock(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,expand_rate=4):
        super(ResBlock, self).__init__()
        self.conv1 = conv_layer(in_channels=chs, out_channels=int(chs / expand_rate), kernel_size=1, bias=True)
        self.conv2 = conv_layer(in_channels=int(chs / expand_rate), out_channels=int(chs / expand_rate), kernel_size=3,
                                padding=1, bias=True)
        self.conv3 = conv_layer(in_channels=int(chs / expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.act1 = act_layer(True)
        self.act2 = act_layer(True)
    def forward(self,x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.conv3(out)
        return x+out


class Ablation_Refine_layerv2(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,qkv_bias=True,
                 drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None,expand_rate=4):
        super(Ablation_Refine_layerv2, self).__init__()
        self.resblock1=ResBlock(chs,conv_layer=conv_layer,act_layer=act_layer,expand_rate=expand_rate)
        self.resblock2=ResBlock(chs,conv_layer=conv_layer,act_layer=act_layer,expand_rate=expand_rate)
        self.resblock3=ResBlock(chs,conv_layer=conv_layer,act_layer=act_layer,expand_rate=expand_rate)
        self.resblock4=ResBlock(chs,conv_layer=conv_layer,act_layer=act_layer,expand_rate=expand_rate)

    def forward(self,x):
        out=self.resblock1(x)
        out=self.resblock2(out)
        out=self.resblock3(out)
        out=self.resblock4(out)
        return out
        
        
#################
class AdaRefine_layerv2(nn.Module):
    def __init__(self, chs, conv_layer=Conv_withpad, act_layer=nn.ELU,
                 patch_height=16, patch_width=52, feat_height=None, feat_width=None,
                 embed_dim=192, depth=2, num_heads=8, mlp_ratio=2, qkv_bias=True,
                 drop_path_rate=0.5, attn_drop_rate=0.5, drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer_trans=nn.GELU, qk_scale=None, expand_rate=4):
        super(AdaRefine_layerv2, self).__init__()
        expand_rate = expand_rate
        self.conv1 = conv_layer(in_channels=chs, out_channels=int(chs / expand_rate), kernel_size=1, bias=True)
        self.conv2 = conv_layer(in_channels=int(chs / expand_rate), out_channels=int(chs / expand_rate), kernel_size=3,
                                padding=1, bias=True)
        self.conv3 = conv_layer(in_channels=int(chs / expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.act1 = act_layer(True)
        self.act2 = act_layer(True)

        self.transblock = TransBlock(in_chs=int(chs / expand_rate), embed_dim=embed_dim, patch_height=patch_height,
                                     patch_width=patch_width,
                                     feat_height=feat_height, feat_width=feat_width,
                                     depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                     drop_path_rate=drop_path_rate,
                                     attn_drop_rate=attn_drop_rate, drop_rate=drop_rate, norm_layer=norm_layer,
                                     act_layer=act_layer_trans, qk_scale=qk_scale)
        self.conv_transblock = conv_layer(in_channels=int(chs / expand_rate), out_channels=chs, kernel_size=1,
                                          bias=True)
        self.channel_att_trans = Channel_attention_block(channels=chs,feat_height=feat_height,feat_width=feat_width)
        self.channel_att_conv = Channel_attention_block(channels=chs,feat_height=feat_height,feat_width=feat_width)

        self.bn1=nn.BatchNorm2d(num_features=int(chs / expand_rate))
        self.bn2=nn.BatchNorm2d(num_features=int(chs / expand_rate))
        self.bn3=nn.BatchNorm2d(num_features=chs)

        self.bn_trans=nn.BatchNorm2d(num_features=int(chs / expand_rate))
        self.bn_trans_1=nn.BatchNorm2d(num_features=chs)

        self.bn_x=nn.BatchNorm2d(num_features=chs)
        bs=1

        self.param1= nn.Parameter(torch.zeros((bs,1,feat_height,feat_width),dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.param2 = nn.Parameter(torch.zeros((bs, 1, feat_height, feat_width), dtype=torch.float32, requires_grad=True),requires_grad=True)

         

    def forward(self, x):
        # b,c,h,w=x.shape
        out_conv1 = self.bn1(self.act1(self.conv1(x)))  #bn1
        out = self.bn2(self.act2(self.conv2(out_conv1)))#bn2
        out = self.bn3(self.conv3(out))#
        out = self.channel_att_conv(out)

        transblock_out = self.bn_trans(self.transblock(out_conv1))
        transblock_out = self.bn_trans_1(self.conv_transblock(transblock_out))
        transblock_out = self.channel_att_trans(transblock_out)

        final_out=self.act1(self.bn_x(x) + self.param1*out + self.param2*transblock_out)

        return final_out


