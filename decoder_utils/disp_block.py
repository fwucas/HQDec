# Author      :  wangfei
# Create time : 2021/10/11 下午4:24
# Email       : fei.wang2@siat.ac.cn
import torch.nn as nn
from functools import partial
import torch
from timm.models.layers.weight_init import trunc_normal_
from .vit_block import Patch_Embed,Block,Conv_withpad
from .general_blocks import Channel_attention_block_Param as Channel_attention_block

class Conv3x3(nn.Module):#

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
class TransToCNN_Disp(nn.Module):
    def __init__(self,in_chs,out_chs,before_embed_height,before_embed_width,patch_height,patch_width,
                 norm_layer=None,act_layer=None):
        super(TransToCNN_Disp, self).__init__()
        self.before_embed_height=before_embed_height
        self.before_embed_width=before_embed_width
        self.patch_height=patch_height
        self.patch_width=patch_width

        self.proj=nn.Conv2d(in_channels=in_chs,out_channels=out_chs,kernel_size=1)
        self.norm=norm_layer(out_chs) if norm_layer is not None else None
        self.act=act_layer() if act_layer is not None else None

        self.upsample=nn.ConvTranspose2d(in_channels=out_chs,out_channels=1,
                                         kernel_size=(patch_height,patch_width),
                                         stride=(patch_height,patch_width))
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
        return x
class TransBlock_Disp(nn.Module):
    def __init__(self,in_chs,embed_dim,patch_height,patch_width,feat_height,feat_width,
                 depth=2,num_heads=8,mlp_ratio=4,qkv_bias=True,drop_path_rate=0.1,
                 attn_drop_rate=0.1,drop_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,qk_scale=None):
        super(TransBlock_Disp, self).__init__()
        self.patch_embed=Patch_Embed(in_chs=in_chs,embed_dim=embed_dim,
                                     patch_height=patch_height,patch_width=patch_width,
                                     feat_height=feat_height,feat_width=feat_width)
        self.num_patches=self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)

        self.transtocnn_disp=TransToCNN_Disp(in_chs=embed_dim,out_chs=in_chs,
                                   before_embed_height=feat_height,before_embed_width=feat_width,
                                   patch_height=patch_height,patch_width=patch_width,norm_layer=None,act_layer=None)

    def forward(self,x):
        x=self.patch_embed(x)
        x=self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x=self.transtocnn_disp(x)
        return x
class Refine_layer_v4(nn.Module):
    def __init__(self,chs,conv_layer=Conv_withpad,act_layer=nn.ReLU,
                 patch_height=16,patch_width=52,feat_height=None,feat_width=None,
                 embed_dim=192,depth=2,num_heads=8,mlp_ratio=2,qkv_bias=True,
                 drop_path_rate=0.5,attn_drop_rate=0.5,drop_rate=0.5,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),act_layer_trans=nn.GELU,qk_scale=None):
        super(Refine_layer_v4, self).__init__()
        expand_rate=4
        self.conv1 = conv_layer(in_channels=chs, out_channels=int(chs/expand_rate), kernel_size=1, bias=True)
        self.conv2 = conv_layer(in_channels=int(chs/expand_rate), out_channels=int(chs/expand_rate), kernel_size=3, padding=1, bias=True)
        self.conv3=conv_layer(in_channels=int(chs/expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.act1 = act_layer(True)
        self.act2 = act_layer(True)


        self.transblock=TransBlock_Disp(in_chs=int(chs/expand_rate),embed_dim=embed_dim,patch_height=patch_height,patch_width=patch_width,
                                   feat_height=feat_height,feat_width=feat_width,
                 depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,drop_path_rate=drop_path_rate,
                 attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,norm_layer=norm_layer,
                 act_layer=act_layer_trans,qk_scale=qk_scale)
        self.conv_transblock = conv_layer(in_channels=int(chs / expand_rate), out_channels=chs, kernel_size=1, bias=True)
        self.channel_att_trans = Channel_attention_block(channels=chs)
        self.channel_att_conv=Channel_attention_block(channels=chs)
    def forward(self,x):

        out_conv1 = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out_conv1))
        out=self.act1(self.conv3(out))
        out=self.channel_att_conv(out)

        transblock_out=self.transblock(out_conv1)
        transblock_out=self.act1(self.conv_transblock(transblock_out))
        transblock_out=self.channel_att_trans(transblock_out)


        return out+transblock_out+x
class Predict_disp(nn.Module):
    def __init__(self,in_channels,out_channels=1,alpha=10,beta=0.01):
        super(Predict_disp, self).__init__()
        self.alpha=alpha
        self.beta=beta

        self.conv=Conv3x3(in_channels=in_channels,out_channels=out_channels)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out=self.conv(x)
        out=self.sigmoid(out)
        out=self.alpha*out+self.beta
        return out
class Predict_dispv3(nn.Module):
    def __init__(self,in_channels,out_channels=1,alpha=10,beta=0.01,
                 patch_height=16,patch_width=52,feat_height=None,feat_width=None):
        super(Predict_dispv3, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.transblock_disp=TransBlock_Disp(in_chs=in_channels,embed_dim=192,patch_height=patch_height,patch_width=patch_width,
                feat_height=feat_height,feat_width=feat_width,depth=2,num_heads=8,mlp_ratio=4,qkv_bias=True,drop_path_rate=0.1,
                 attn_drop_rate=0.1,drop_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,qk_scale=None)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        conv_out=self.conv(x)
        conv_out=self.sigmoid(conv_out)

        transblockdisp=self.transblock_disp(x)
        transblockdisp=self.sigmoid(transblockdisp)
        final_disp=0.5*conv_out+0.5*transblockdisp
        final_disp = self.alpha * final_disp + self.beta
        return  final_disp


class Predict_dispv3_1(nn.Module):
    def __init__(self, in_channels, out_channels=1, alpha=10, beta=0.01,
                 patch_height=16, patch_width=52, feat_height=None, feat_width=None):
        super(Predict_dispv3_1, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.transblock_disp = TransBlock_Disp(in_chs=in_channels, embed_dim=192, patch_height=patch_height,
                                               patch_width=patch_width,
                                               feat_height=feat_height, feat_width=feat_width, depth=2, num_heads=8,
                                               mlp_ratio=4, qkv_bias=True, drop_path_rate=0.1,
                                               attn_drop_rate=0.1, drop_rate=0.1,
                                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                               act_layer=nn.GELU, qk_scale=None)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)

    def forward(self, x):
        conv_out = self.conv(x)

        transblockdisp = self.transblock_disp(x)
        transblockdisp = self.sigmoid(transblockdisp)
        conv_out = conv_out * transblockdisp
        conv_out = self.sigmoid(conv_out)
        final_disp = conv_out

        final_disp = self.alpha * final_disp + self.beta
        return final_disp


class Predict_dispv3_2(nn.Module):
    def __init__(self, in_channels, out_channels=1, alpha=10, beta=0.01,
                 patch_height=16, patch_width=52, feat_height=None, feat_width=None,embed_dim=192,
                 depth=2, num_heads=8,mlp_ratio=4,drop_path_rate=0.1,attn_drop_rate=0.1, drop_rate=0.1):
        super(Predict_dispv3_2, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.transblock_disp = TransBlock_Disp(in_chs=in_channels, embed_dim=embed_dim, patch_height=patch_height,
                                               patch_width=patch_width,
                                               feat_height=feat_height, feat_width=feat_width, depth=depth, num_heads=num_heads,
                                               mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=drop_path_rate,
                                               attn_drop_rate=attn_drop_rate, drop_rate=drop_rate,
                                               norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                               act_layer=nn.GELU, qk_scale=None)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.chs_att=Channel_attention_block(channels=in_channels,feat_height=feat_height,feat_width=feat_width)

    def forward(self, x):
        x=self.chs_att(x)
        conv_out = self.conv(x)

        transblockdisp = self.transblock_disp(x)
        transblockdisp = self.sigmoid(transblockdisp)
        conv_out = conv_out * transblockdisp
        conv_out = self.sigmoid(conv_out)
        final_disp = conv_out
        final_disp = self.alpha * final_disp + self.beta
        return final_disp
class Down_disp(nn.Module):
    def __init__(self,scale=2):
        super(Down_disp, self).__init__()
        self.scale=scale
        #self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU(True)
        if self.scale==2:
            self.conv1=nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=2,padding=1)
            self.conv1x1=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        if self.scale==4:
            self.conv1=nn.Conv2d(in_channels=1,out_channels=2,kernel_size=3,stride=2,padding=1)
            self.conv2=nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1)
            self.conv1x1=nn.Conv2d(in_channels=4,out_channels=1,kernel_size=3,padding=1)
    def forward(self,x):
        if self.scale==2:
            out=self.conv1(x)
            out=self.relu(out)
            out = self.conv1x1(out)
            #out = self.relu(out)
        elif self.scale==4:
            out=self.conv1(x)
            out=self.relu(out)
            out=self.conv2(out)
            out=self.relu(out)
            out=self.conv1x1(out)

        return out
class Up_disp(nn.Module):
    def __init__(self,scale=2):
        super(Up_disp, self).__init__()
        self.scale=scale
        self.relu=nn.ReLU(True)
        if self.scale==2:
            self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
        elif self.scale==4:
            self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.conv2=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,padding=1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)

    def forward(self,x):
        if self.scale==2:
            out=self.conv1(x)
            out=self.relu(out)
            out=self.pixel_shuffle1(out)
        elif self.scale==4:
            out=self.conv1(x)
            out=self.relu(out)
            out=self.pixel_shuffle1(out)
            out=self.conv2(out)
            out=self.relu(out)
            out=self.pixel_shuffle2(out)
        return out
class DownSamplev3(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2):
        super(DownSamplev3, self).__init__()
        self.maxpool_ratio = 0.5
        self.in_chs = in_chs
        self.stride = stride
        self.maxdownsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.relu=nn.ReLU(True)
        if self.stride==2:
            self.conv_stride1=nn.Conv2d(in_channels=int(in_chs*(1-self.maxpool_ratio)),
                                       out_channels=2*int(in_chs*(1-self.maxpool_ratio)),
                                       kernel_size=3,stride=2,padding=1)
            self.conv_stride_1x1=nn.Conv2d(in_channels=2*int(in_chs*(1-self.maxpool_ratio)),
                                           out_channels=int(in_chs*(1-self.maxpool_ratio)),
                                           kernel_size=1,stride=1)
        elif self.stride==4:
            self.conv_stride1=nn.Conv2d(in_channels=int(in_chs*(1-self.maxpool_ratio)),
                                       out_channels=2*int(in_chs*(1-self.maxpool_ratio)),
                                        kernel_size=3,stride=2,padding=1)
            self.conv_stride2=nn.Conv2d(in_channels=2*int(in_chs*(1-self.maxpool_ratio)),
                                        out_channels=2*2*int(in_chs*(1-self.maxpool_ratio)),
                                        kernel_size=3,stride=2,padding=1)
            self.conv_stride_1x1=nn.Conv2d(in_channels=2*2*int(in_chs*(1-self.maxpool_ratio)),
                                           out_channels=int(in_chs*(1-self.maxpool_ratio)),
                                           kernel_size=1,stride=1)
        elif self.stride==8:
            self.conv_stride1=nn.Conv2d(in_channels=int(in_chs*(1-self.maxpool_ratio)),
                                       out_channels=2*int(in_chs*(1-self.maxpool_ratio)),
                                        kernel_size=3,stride=2,padding=1)
            self.conv_stride2 = nn.Conv2d(in_channels=2 * int(in_chs * (1 - self.maxpool_ratio)),
                                          out_channels=2 * 2 * int(in_chs * (1 - self.maxpool_ratio)),
                                          kernel_size=3, stride=2, padding=1)
            self.conv_stride3 = nn.Conv2d(in_channels=2*2 * int(in_chs * (1 - self.maxpool_ratio)),
                                          out_channels=2*2 * 2 * int(in_chs * (1 - self.maxpool_ratio)),
                                          kernel_size=3, stride=2, padding=1)
            self.conv_stride_1x1 = nn.Conv2d(in_channels=2*2 * 2 * int(in_chs * (1 - self.maxpool_ratio)),
                                             out_channels=int(in_chs * (1 - self.maxpool_ratio)),
                                             kernel_size=1, stride=1)


    def forward(self,x):
        maxpool_x, conv_x = torch.split(x, [self.in_chs - int(self.in_chs * (1 - self.maxpool_ratio)),
                                            int(self.in_chs * (1 - self.maxpool_ratio))], dim=1)
        max_down = self.maxdownsample(maxpool_x)
        if self.stride==2:
            conv_out=self.conv_stride1(conv_x)
            conv_out=self.relu(conv_out)
            conv_out=self.conv_stride_1x1(conv_out)
            conv_out=self.relu(conv_out)
        elif self.stride==4:
            conv_out=self.conv_stride1(conv_x)
            conv_out=self.relu(conv_out)
            conv_out=self.conv_stride2(conv_out)
            conv_out=self.relu(conv_out)
            conv_out=self.conv_stride_1x1(conv_out)
            conv_out=self.relu(conv_out)
        elif self.stride==8:
            conv_out = self.conv_stride1(conv_x)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_stride2(conv_out)
            conv_out = self.relu(conv_out)
            conv_out=self.conv_stride3(conv_out)
            conv_out=self.relu(conv_out)
            conv_out = self.conv_stride_1x1(conv_out)
            conv_out = self.relu(conv_out)
        out = torch.cat((max_down, conv_out), dim=1)
        return out



class Ablation_Predict_disp(nn.Module):
    def __init__(self,in_channels, out_channels=1, alpha=10, beta=0.01,
                 patch_height=16, patch_width=52, feat_height=None, feat_width=None,embed_dim=192,
                 depth=2, num_heads=8,mlp_ratio=4,drop_path_rate=0.1,attn_drop_rate=0.1, drop_rate=0.1):
        super(Ablation_Predict_disp, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.conv = Conv3x3(in_channels=in_channels, out_channels=out_channels)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out=self.sigmoid(self.conv(x))*self.alpha +self.beta
        return out

if __name__ == '__main__':
    input_tensor=torch.randn(2,16,32,104)
    out=TransBlock_Disp(in_chs=16,embed_dim=192,patch_height=16,patch_width=52,feat_height=32,feat_width=104,
                 depth=2,num_heads=8,mlp_ratio=4,qkv_bias=True,drop_path_rate=0.1,
                 attn_drop_rate=0.1,drop_rate=0.1,norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,qk_scale=None)
