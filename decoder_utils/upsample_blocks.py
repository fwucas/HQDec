# ！/usr/bin/python
# Create Time : 2022/4/22 下午3:52
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch.nn as nn
from .vit_block import Refine_layer_v4
from functools import partial
import torch
from .general_blocks import Conv_withpad
from .refine_blocks import AdaRefine_layerv2


class Upsample_interpolate(nn.Module):
    def __init__(self,upscale_factor=2,mode='bilinear'):
        super(Upsample_interpolate, self).__init__()
        self.mode=mode
        self.upscale_factor=upscale_factor
    def forward(self,x):
        return torch.nn.functional.interpolate(x,scale_factor=self.upscale_factor,
                                             mode=self.mode,align_corners=False)

class Sub_Pixel_v4_1(nn.Module):
    def __init__(self,in_chs=None,upscale=2,patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5):
        super(Sub_Pixel_v4_1, self).__init__()
        #Refine_layer_v3
        self.inter_upsample_ratio=0.5
        self.in_chs=in_chs
        self.conv1 = nn.Conv2d(in_channels=int(in_chs*(1-self.inter_upsample_ratio)),
                               out_channels=int(in_chs*(1-self.inter_upsample_ratio)) * (upscale ** 2), kernel_size=1, stride=1,
                               padding=0,bias=False)
        self.refine=Refine_layer_v4(chs=int(in_chs*(1-self.inter_upsample_ratio)) * (upscale ** 2),conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=patch_height,patch_width=patch_width,feat_height=feat_height,feat_width=feat_width,
                 embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=True,
                 drop_path_rate=drop_path_rate,attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.act = nn.ReLU(True)
        self.interpolate=Upsample_interpolate()

    def forward(self,x):

        pixel_x,inter_x=torch.split(x,[int(self.in_chs*(1-self.inter_upsample_ratio)),self.in_chs-int(self.in_chs*(1-self.inter_upsample_ratio))],dim=1)
        interpolate_upsample=self.interpolate(inter_x)
        pixel_x=self.act(self.conv1(pixel_x))

        pixel_x=self.act(self.refine(pixel_x))

        pixel_x=self.pixel_shuffle(pixel_x)
        out=torch.cat((interpolate_upsample,pixel_x),dim=1)


        return out




class AdaSub_Pixel_v4_1_GN(nn.Module):
    def __init__(self,in_chs=None,upscale=2,patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5):
        super(AdaSub_Pixel_v4_1_GN, self).__init__()

        self.inter_upsample_ratio=0.5
        self.in_chs=in_chs
        self.conv1 = nn.Conv2d(in_channels=in_chs,
                               out_channels=in_chs * (upscale ** 2), kernel_size=1, stride=1,
                               padding=0,bias=False)
        self.refine=Refine_layer_v4(chs=in_chs * (upscale ** 2),conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=patch_height,patch_width=patch_width,feat_height=feat_height,feat_width=feat_width,
                 embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=True,
                 drop_path_rate=drop_path_rate,attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.act = nn.ReLU(True)
        self.interpolate=Upsample_interpolate()



        self.bn1=nn.BatchNorm2d(num_features=in_chs * (upscale ** 2))
        self.bn2 = nn.BatchNorm2d(num_features=in_chs * (upscale ** 2))
        self.bn3 = nn.BatchNorm2d(num_features=in_chs)
        self.bn4 = nn.BatchNorm2d(num_features=in_chs)


        bs = 1
        self.lambad_param=torch.nn.Parameter(torch.zeros((bs,1,feat_height*2,feat_width*2),
                                                         dtype=torch.float32, requires_grad=True, device='cuda:0'), requires_grad=True)


    def forward(self,x):
        interpolate_upsample=self.interpolate(x)
        interpolate_upsample=self.bn4(interpolate_upsample)


        pixel_x=self.act(self.conv1(x))
        pixel_x=self.bn1(pixel_x)

        pixel_x=self.act(self.refine(pixel_x))
        pixel_x=self.bn2(pixel_x)


        pixel_x=self.pixel_shuffle(pixel_x)
        pixel_x=self.bn3(pixel_x)

        out=interpolate_upsample+ self.lambad_param*pixel_x



        return out

class DAdaSub_pixel_v4_1_GN(nn.Module):
    def __init__(self,in_chs=None,upscale=2,patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,expand_rate=4):
        super(DAdaSub_pixel_v4_1_GN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_chs,
                               out_channels=in_chs * (upscale ** 2), kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.refine=AdaRefine_layerv2(chs=in_chs * (upscale ** 2), conv_layer=Conv_withpad, act_layer=nn.ELU,
                 patch_height=patch_height, patch_width=patch_width, feat_height=feat_height, feat_width=feat_width,
                 embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                 drop_path_rate=drop_path_rate, attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer_trans=nn.GELU, qk_scale=None, expand_rate=expand_rate)

        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.act = nn.ELU(True)
        self.interpolate = Upsample_interpolate()       

        self.bn1 = nn.BatchNorm2d(num_features=in_chs * (upscale ** 2))
        self.bn2 = nn.BatchNorm2d(num_features=in_chs * (upscale ** 2))
        self.bn3 = nn.BatchNorm2d(num_features=in_chs)
        self.bn4 = nn.BatchNorm2d(num_features=in_chs)

        bs = 1
        self.lambad_param = torch.nn.Parameter(torch.zeros((bs, 1, feat_height * 2, feat_width * 2),
                                                           dtype=torch.float32, requires_grad=True),
                                               requires_grad=True)#
        
    def forward(self,x):
        interpolate_upsample = self.interpolate(x)
        interpolate_upsample = self.bn4(interpolate_upsample)

        pixel_x = self.act(self.conv1(x))
        pixel_x = self.bn1(pixel_x)

        pixel_x = self.act(self.refine(pixel_x))
        pixel_x = self.bn2(pixel_x)

        pixel_x = self.pixel_shuffle(pixel_x)
        pixel_x = self.bn3(pixel_x)

        out = interpolate_upsample + self.lambad_param * pixel_x

        return out


class Ablation_Sub_pixel(nn.Module):
    def __init__(self,in_chs=None,upscale=2,patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5):
        super(Ablation_Sub_pixel, self).__init__()
        self.interpolate=Upsample_interpolate()
    def forward(self,x):
        return self.interpolate(x)
