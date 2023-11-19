# ！/usr/bin/python
# Create Time : 2022/4/22 下午3:29
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch.nn as nn
import numpy as np
import torch
from .decoder_utils.general_blocks import ConvBlock
from .decoder_utils.upsample_blocks import DAdaSub_pixel_v4_1_GN
from .decoder_utils.chs_att_axis import AdaChs_Att_Axis_Down_groups
from .decoder_utils.refine_blocks import AdaRefine_layerv2
from .decoder_utils.disp_block import Predict_dispv3_2


class HQDec(nn.Module):
    def __init__(self,chs_transfors_layer=ConvBlock,
                 up_layer=DAdaSub_pixel_v4_1_GN,
                 down_layer=AdaChs_Att_Axis_Down_groups,
                 refine_layer=AdaRefine_layerv2,
                 Predict_disp=Predict_dispv3_2,
                 encoder_chs_num=np.array([24, 48, 64, 160, 256]),
                 decoder_chs_num=np.array([24, 48, 64, 160, 256]),
                 patch_size_list = [(16, 52), (16, 52), (16, 52), (16, 52), (8, 26)],
                 feat_size_list = [(128, 416), (64, 208), (32, 104), (16, 52), (8, 26)],
                 disp_size_list=[(256,832),(128,416),(64,208)],
                 embed_dim_list=[24, 48, 64, 160, 256],
                 disp_embed_dim_list=[12, 24, 32],
                 depth=[2, 2, 2, 2, 2],
                 disp_depth=[2, 2, 2],
                 num_heads=[2, 4, 4, 8, 8],
                 disp_num_heads= [2, 4, 4],
                 mlp_ratio=[2, 2, 2, 2, 2],
                 disp_mlp_ratio=[2, 2, 2],
                 drop_path_rate=[0., 0., 0., 0., 0.],
                 disp_drop_path_rate=[0., 0., 0.],
                 attn_drop_rate=[0., 0., 0., 0., 0.],
                 disp_attn_drop_rate=[0., 0., 0.],
                 drop_rate=[0., 0., 0., 0., 0.],
                 disp_drop_rate=[0., 0., 0.],
                 refine_expand_rate=[2, 4, 4, 8, 8],
                 use_dilation=False,out_stride=16,
                 use_mutiscale_connection=True,final_constraint_dispout_flag=True,**kwargs):

        super(HQDec, self).__init__()
        self.embed_dim_list =embed_dim_list
        self.disp_embed_dim_list=disp_embed_dim_list
        self.depth = depth
        self.disp_depth=disp_depth
        self.num_heads=num_heads
        self.disp_num_heads=disp_num_heads
        self.mlp_ratio =mlp_ratio
        self.disp_mlp_ratio =disp_mlp_ratio
        self.drop_path_rate=drop_path_rate
        self.disp_drop_path_rate =disp_drop_path_rate
        self.attn_drop_rate=attn_drop_rate
        self.disp_attn_drop_rate =disp_attn_drop_rate
        self.drop_rate = drop_rate
        self.disp_drop_rate =disp_drop_rate
        self.refine_expand_rate=refine_expand_rate

        self.use_mutiscale_connection=use_mutiscale_connection
        self.use_dilation=use_dilation
        self.out_stride=out_stride



        self.patch_size_list = patch_size_list
        self.feat_size_list = feat_size_list
        self.disp_size_list=disp_size_list



        self.encoder_chs_num=encoder_chs_num
        self.decoder_chs_num=decoder_chs_num
        if final_constraint_dispout_flag:
            alpha = 1
            beta = 0
        else:
            alpha = 10
            beta = 0.01

        self.feat0_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[0],out_channels=decoder_chs_num[0])
        self.feat0_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[0],patch_height=self.patch_size_list[0][0],patch_width=self.patch_size_list[0][1],
                                                    feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1],
                                                    embed_dim=self.embed_dim_list[0],depth=self.depth[0],num_heads=self.num_heads[0],
                                                    mlp_ratio=self.mlp_ratio[0],drop_path_rate=self.drop_path_rate[0],
                                                    attn_drop_rate=self.attn_drop_rate[0],drop_rate=self.drop_rate[0],expand_rate=self.refine_expand_rate[0])

        if self.use_mutiscale_connection:
            self.feat0_down_scale2=down_layer(in_chs=encoder_chs_num[0],kernel_size=2,stride=2,feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1])
            self.feat0_down_scale2_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[0],out_channels=decoder_chs_num[1])
            self.feat0_down_scale2_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[1],patch_height=self.patch_size_list[1][0],patch_width=self.patch_size_list[1][1],
                                                                    feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1],
                                                                    embed_dim=self.embed_dim_list[1],depth=self.depth[1],num_heads=self.num_heads[1],
                                                                    mlp_ratio=self.mlp_ratio[1],drop_path_rate=self.drop_path_rate[1],
                                                                    attn_drop_rate=self.attn_drop_rate[1],drop_rate=self.drop_rate[1],expand_rate=self.refine_expand_rate[1])

            self.feat0_down_scale4=down_layer(in_chs=encoder_chs_num[0],kernel_size=4,stride=4,feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1])
            self.feat0_down_scale4_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[0],out_channels=decoder_chs_num[2])
            self.feat0_down_scale4_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[2],patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                                                    feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1],
                                                                    embed_dim=self.embed_dim_list[2],depth=self.depth[2],num_heads=self.num_heads[2],
                                                                    mlp_ratio=self.mlp_ratio[2],drop_path_rate=self.drop_path_rate[2],
                                                                    attn_drop_rate=self.attn_drop_rate[2],drop_rate=self.drop_rate[2],expand_rate=self.refine_expand_rate[2])


            if self.use_dilation and self.out_stride==8:
                pass
            else:
                self.feat0_down_scale8=down_layer(in_chs=encoder_chs_num[0],kernel_size=8,stride=8,feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1])
            self.feat0_down_scale8_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[0],out_channels=decoder_chs_num[3])
            self.feat0_down_scale8_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[3],patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                                    feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                                    embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                                    mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                                    attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3],expand_rate=self.refine_expand_rate[3])

        self.feat1_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[1],out_channels=decoder_chs_num[1])
        self.feat1_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[1],patch_height=self.patch_size_list[1][0],patch_width=self.patch_size_list[1][1],
                                                    feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1],
                                                    embed_dim=self.embed_dim_list[1],depth=self.depth[1],num_heads=self.num_heads[1],
                                                    mlp_ratio=self.mlp_ratio[1],drop_path_rate=self.drop_path_rate[1],
                                                    attn_drop_rate=self.attn_drop_rate[1],drop_rate=self.drop_rate[1],expand_rate=self.refine_expand_rate[1])


        if self.use_mutiscale_connection:
            self.feat1_down_scale2=down_layer(in_chs=encoder_chs_num[1],kernel_size=2,stride=2,feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1])
            self.feat1_down_scale2_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[1],out_channels=decoder_chs_num[2])
            self.feat1_down_scale2_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[2],patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                                                    feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1],
                                                                    embed_dim=self.embed_dim_list[2],depth=self.depth[2],num_heads=self.num_heads[2],
                                                                    mlp_ratio=self.mlp_ratio[2],drop_path_rate=self.drop_path_rate[2],
                                                                    attn_drop_rate=self.attn_drop_rate[2],drop_rate=self.drop_rate[2],expand_rate=self.refine_expand_rate[2])

            if self.use_dilation and self.out_stride==8:
                pass
            else:
                self.feat1_down_scale4=down_layer(in_chs=encoder_chs_num[1],kernel_size=4,stride=4,feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1])
            self.feat1_down_scale4_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[1],out_channels=decoder_chs_num[3])
            self.feat1_down_scale4_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[3],patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                                    feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                                    embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                                    mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                                    attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3],expand_rate=self.refine_expand_rate[3])

        self.feat2_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[2],out_channels=decoder_chs_num[2])
        self.feat2_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[2],patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                                    feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1],
                                                    embed_dim=self.embed_dim_list[2],depth=self.depth[2],num_heads=self.num_heads[2],
                                                    mlp_ratio=self.mlp_ratio[2],drop_path_rate=self.drop_path_rate[2],
                                                    attn_drop_rate=self.attn_drop_rate[2],drop_rate=self.drop_rate[2],expand_rate=self.refine_expand_rate[2])

        if self.use_mutiscale_connection:
            if self.use_dilation and self.out_stride==8:
                pass
            else:
                self.feat2_down_scale2=down_layer(in_chs=encoder_chs_num[2],kernel_size=2,stride=2,feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1])
            self.feat2_down_scale2_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[2],out_channels=decoder_chs_num[3])
            self.feat2_down_scale2_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[3],patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                                    feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                                    embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                                    mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                                    attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3],expand_rate=self.refine_expand_rate[3])

        self.feat3_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[3],out_channels=decoder_chs_num[3])
        self.feat3_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[3],patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                    feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                    embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                    mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                    attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3],expand_rate=self.refine_expand_rate[3])

        self.feat4_chs_tl=chs_transfors_layer(in_channels=encoder_chs_num[4],out_channels=decoder_chs_num[3])
        self.feat4_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[3],patch_height=self.patch_size_list[4][0],patch_width=self.patch_size_list[4][1],
                                                    feat_height=self.feat_size_list[4][0],feat_width=self.feat_size_list[4][1],
                                                    embed_dim=self.embed_dim_list[4],depth=self.depth[4],num_heads=self.num_heads[4],
                                                    mlp_ratio=self.mlp_ratio[4],drop_path_rate=self.drop_path_rate[4],
                                                    attn_drop_rate=self.attn_drop_rate[4],drop_rate=self.drop_rate[4],expand_rate=self.refine_expand_rate[4])

        if self.use_dilation and (self.out_stride==16 or self.out_stride==8):
            pass
        else:
            self.feat4_chs_tl_refine_layer_up=up_layer(in_chs=decoder_chs_num[3],upscale=2,patch_height=self.patch_size_list[4][0],patch_width=self.patch_size_list[4][1],
                                                       feat_height=self.feat_size_list[4][0],feat_width=self.feat_size_list[4][1],
                                                       embed_dim=self.embed_dim_list[4],depth=self.depth[4],num_heads=self.num_heads[4],
                                                       mlp_ratio=self.mlp_ratio[4],drop_path_rate=self.drop_path_rate[4],
                                                       attn_drop_rate=self.attn_drop_rate[4],drop_rate=self.drop_rate[4])

        self.fuse3_chs_tl=chs_transfors_layer(in_channels=decoder_chs_num[3],out_channels=decoder_chs_num[2])
        self.fuse3_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[2],patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                    feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                    embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                    mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                    attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3],expand_rate=self.refine_expand_rate[3])

        if self.use_dilation and self.out_stride==8:
            pass
        else:
            self.fuse3_chs_tl_refine_layer_up=up_layer(in_chs=decoder_chs_num[2],upscale=2,patch_height=self.patch_size_list[3][0],patch_width=self.patch_size_list[3][1],
                                                       feat_height=self.feat_size_list[3][0],feat_width=self.feat_size_list[3][1],
                                                       embed_dim=self.embed_dim_list[3],depth=self.depth[3],num_heads=self.num_heads[3],
                                                       mlp_ratio=self.mlp_ratio[3],drop_path_rate=self.drop_path_rate[3],
                                                       attn_drop_rate=self.attn_drop_rate[3],drop_rate=self.drop_rate[3])

        self.fuse2_chs_tl=chs_transfors_layer(in_channels=decoder_chs_num[2],out_channels=decoder_chs_num[1])
        self.fuse2_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[1],patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                                    feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1],
                                                    embed_dim=self.embed_dim_list[2],depth=self.depth[2],num_heads=self.num_heads[2],
                                                    mlp_ratio=self.mlp_ratio[2],drop_path_rate=self.drop_path_rate[2],
                                                    attn_drop_rate=self.attn_drop_rate[2],drop_rate=self.drop_rate[2],expand_rate=self.refine_expand_rate[2])
        self.fuse2_chs_tl_refine_layer_up=up_layer(in_chs=decoder_chs_num[1],upscale=2,patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                                   feat_height=self.feat_size_list[2][0],feat_width=self.feat_size_list[2][1],
                                                   embed_dim=self.embed_dim_list[2],depth=self.depth[2],num_heads=self.num_heads[2],
                                                   mlp_ratio=self.mlp_ratio[2],drop_path_rate=self.drop_path_rate[2],
                                                   attn_drop_rate=self.attn_drop_rate[2],drop_rate=self.drop_rate[2])

        self.fuse1_chs_tl=chs_transfors_layer(in_channels=decoder_chs_num[1],out_channels=decoder_chs_num[0])
        self.fuse1_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[0],patch_height=self.patch_size_list[1][0],patch_width=self.patch_size_list[1][1],
                                                    feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1],
                                                    embed_dim=self.embed_dim_list[1],depth=self.depth[1],num_heads=self.num_heads[1],
                                                    mlp_ratio=self.mlp_ratio[1],drop_path_rate=self.drop_path_rate[1],
                                                    attn_drop_rate=self.attn_drop_rate[1],drop_rate=self.drop_rate[1],expand_rate=self.refine_expand_rate[1])
        self.fuse1_chs_tl_refine_layer_up=up_layer(in_chs=decoder_chs_num[0],upscale=2,patch_height=self.patch_size_list[1][0],patch_width=self.patch_size_list[1][1],
                                                   feat_height=self.feat_size_list[1][0],feat_width=self.feat_size_list[1][1],
                                                   embed_dim=self.embed_dim_list[1],depth=self.depth[1],num_heads=self.num_heads[1],
                                                   mlp_ratio=self.mlp_ratio[1],drop_path_rate=self.drop_path_rate[1],
                                                   attn_drop_rate=self.attn_drop_rate[1],drop_rate=self.drop_rate[1])

        self.fuse0_chs_tl=chs_transfors_layer(in_channels=decoder_chs_num[0],out_channels=decoder_chs_num[0])
        self.fuse0_chs_tl_refine_layer=refine_layer(chs=decoder_chs_num[0],patch_height=self.patch_size_list[0][0],patch_width=self.patch_size_list[0][1],
                                                    feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1],
                                                    embed_dim=self.embed_dim_list[0],depth=self.depth[0],num_heads=self.num_heads[0],
                                                    mlp_ratio=self.mlp_ratio[0],drop_path_rate=self.drop_path_rate[0],
                                                    attn_drop_rate=self.attn_drop_rate[0],drop_rate=self.drop_rate[0],expand_rate=self.refine_expand_rate[0])
        self.fuse0_chs_tl_refine_layer_up=up_layer(in_chs=decoder_chs_num[0],upscale=2,patch_height=self.patch_size_list[0][0],patch_width=self.patch_size_list[0][1],
                                                   feat_height=self.feat_size_list[0][0],feat_width=self.feat_size_list[0][1],
                                                   embed_dim=self.embed_dim_list[0],depth=self.depth[0],num_heads=self.num_heads[0],
                                                   mlp_ratio=self.mlp_ratio[0],drop_path_rate=self.drop_path_rate[0],
                                                   attn_drop_rate=self.attn_drop_rate[0],drop_rate=self.drop_rate[0])

        self.disp0=Predict_disp(in_channels=decoder_chs_num[0],out_channels=1, alpha=alpha, beta=beta,
                                patch_height=self.patch_size_list[0][0],patch_width=self.patch_size_list[0][1],
                                feat_height=self.disp_size_list[0][0],feat_width=self.disp_size_list[0][1],
                                embed_dim=self.disp_embed_dim_list[0],depth=self.disp_depth[0],num_heads=self.disp_num_heads[0],
                                mlp_ratio=self.disp_mlp_ratio[0],drop_path_rate=self.disp_drop_path_rate[0],
                                attn_drop_rate=self.disp_attn_drop_rate[0],drop_rate=self.disp_drop_rate[0])
        self.disp1=Predict_disp(in_channels=decoder_chs_num[0],out_channels=1, alpha=alpha, beta=beta,
                                patch_height=self.patch_size_list[1][0],patch_width=self.patch_size_list[1][1],
                                feat_height=self.disp_size_list[1][0],feat_width=self.disp_size_list[1][1],
                                embed_dim=self.disp_embed_dim_list[1],depth=self.disp_depth[1],num_heads=self.disp_num_heads[1],
                                mlp_ratio=self.disp_mlp_ratio[1],drop_path_rate=self.disp_drop_path_rate[1],
                                attn_drop_rate=self.disp_attn_drop_rate[1],drop_rate=self.disp_drop_rate[1])
        self.disp2=Predict_disp(in_channels=decoder_chs_num[1],out_channels=1, alpha=alpha, beta=beta,
                                patch_height=self.patch_size_list[2][0],patch_width=self.patch_size_list[2][1],
                                feat_height=self.disp_size_list[2][0],feat_width=self.disp_size_list[2][1],
                                embed_dim=self.disp_embed_dim_list[2],depth=self.disp_depth[2],num_heads=self.disp_num_heads[2],
                                mlp_ratio=self.disp_mlp_ratio[2],drop_path_rate=self.disp_drop_path_rate[2],
                                attn_drop_rate=self.disp_attn_drop_rate[2],drop_rate=self.disp_drop_rate[2])


        self.rec_img0=Predict_disp(in_channels=decoder_chs_num[0], out_channels=3, alpha=1, beta=0,
                                   patch_height=self.patch_size_list[0][0], patch_width=self.patch_size_list[0][1],
                                   feat_height=self.disp_size_list[0][0], feat_width=self.disp_size_list[0][1],
                                   embed_dim=self.disp_embed_dim_list[0],depth=self.disp_depth[0],num_heads=self.disp_num_heads[0],
                                   mlp_ratio=self.disp_mlp_ratio[0],drop_path_rate=self.disp_drop_path_rate[0],
                                   attn_drop_rate=self.disp_attn_drop_rate[0],drop_rate=self.disp_drop_rate[0])

        self.rec_img1 = Predict_disp(in_channels=decoder_chs_num[0], out_channels=3, alpha=1, beta=0,
                                     patch_height=self.patch_size_list[1][0], patch_width=self.patch_size_list[1][1],
                                     feat_height=self.disp_size_list[1][0], feat_width=self.disp_size_list[1][1],
                                     embed_dim=self.disp_embed_dim_list[1],depth=self.disp_depth[1],num_heads=self.disp_num_heads[1],
                                     mlp_ratio=self.disp_mlp_ratio[1],drop_path_rate=self.disp_drop_path_rate[1],
                                     attn_drop_rate=self.disp_attn_drop_rate[1],drop_rate=self.disp_drop_rate[1])
        self.rec_img2 = Predict_disp(in_channels=decoder_chs_num[1], out_channels=3, alpha=1, beta=0,
                                     patch_height=self.patch_size_list[2][0], patch_width=self.patch_size_list[2][1],
                                     feat_height=self.disp_size_list[2][0], feat_width=self.disp_size_list[2][1],
                                     embed_dim=self.disp_embed_dim_list[2],depth=self.disp_depth[2],num_heads=self.disp_num_heads[2],
                                     mlp_ratio=self.disp_mlp_ratio[2],drop_path_rate=self.disp_drop_path_rate[2],
                                     attn_drop_rate=self.disp_attn_drop_rate[2],drop_rate=self.disp_drop_rate[2])

        bs = 1#


        self.fuse3_lambda0 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-2], self.feat_size_list[-2][0], self.feat_size_list[-2][1]), dtype=torch.float32,
            requires_grad=True),requires_grad=True)
        self.fuse3_lambda1 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-2], self.feat_size_list[-2][0], self.feat_size_list[-2][1]),
            dtype=torch.float32, requires_grad=True),requires_grad=True)
        self.fuse3_lambda2 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-2], self.feat_size_list[-2][0], self.feat_size_list[-2][1]),
            dtype=torch.float32, requires_grad=True),requires_grad=True)

        self.fuse2_lambda0 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-3], self.feat_size_list[-3][0], self.feat_size_list[-3][1]),
            dtype=torch.float32, requires_grad=True),requires_grad=True)
        self.fuse2_lambda1 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-3], self.feat_size_list[-3][0], self.feat_size_list[-3][1]),
            dtype=torch.float32, requires_grad=True),requires_grad=True)

        self.fuse1_lambda0 = torch.nn.Parameter(torch.zeros(
            (bs, decoder_chs_num[-4], self.feat_size_list[-4][0], self.feat_size_list[-4][1]),
            dtype=torch.float32, requires_grad=True),requires_grad=True)

    def forward(self,features_list):
        feat0,feat1,feat2,feat3,feat4=features_list
        feat4=self.feat4_chs_tl(feat4)
        feat4=self.feat4_chs_tl_refine_layer(feat4)
        if self.use_dilation and (self.out_stride == 16 or self.out_stride == 8):
            feat4_up=feat4
        else:
            feat4_up=self.feat4_chs_tl_refine_layer_up(feat4)

        feat3_down_scale0=self.feat3_chs_tl(feat3)
        feat3_down_scale0=self.feat3_chs_tl_refine_layer(feat3_down_scale0)#

        feat2_down_scale0=self.feat2_chs_tl(feat2)
        feat2_down_scale0=self.feat2_chs_tl_refine_layer(feat2_down_scale0)

        if self.use_mutiscale_connection:

            if self.use_dilation and self.out_stride == 8:
                feat2_down_scale2=self.feat2_down_scale2_chs_tl(feat2)
            else:
                feat2_down_scale2=self.feat2_down_scale2(feat2)
                feat2_down_scale2=self.feat2_down_scale2_chs_tl(feat2_down_scale2)
            feat2_down_scale2=self.feat2_down_scale2_chs_tl_refine_layer(feat2_down_scale2)

        feat1_down_scale0=self.feat1_chs_tl(feat1)
        feat1_down_scale0=self.feat1_chs_tl_refine_layer(feat1_down_scale0)

        if self.use_mutiscale_connection:
            feat1_down_scale2_tmp=self.feat1_down_scale2(feat1)
            feat1_down_scale2=self.feat1_down_scale2_chs_tl(feat1_down_scale2_tmp)
            feat1_down_scale2=self.feat1_down_scale2_chs_tl_refine_layer(feat1_down_scale2)

            if self.use_dilation and self.out_stride == 8:
                feat1_down_scale4=self.feat1_down_scale4_chs_tl(feat1_down_scale2_tmp)
            else:
                feat1_down_scale4=self.feat1_down_scale4(feat1)
                feat1_down_scale4=self.feat1_down_scale4_chs_tl(feat1_down_scale4)
            feat1_down_scale4=self.feat1_down_scale4_chs_tl_refine_layer(feat1_down_scale4)

        feat0_down_scale0=self.feat0_chs_tl(feat0)
        feat0_down_scale0=self.feat0_chs_tl_refine_layer(feat0_down_scale0)

        if self.use_mutiscale_connection:
            feat0_down_scale2=self.feat0_down_scale2(feat0)
            feat0_down_scale2=self.feat0_down_scale2_chs_tl(feat0_down_scale2)
            feat0_down_scale2=self.feat0_down_scale2_chs_tl_refine_layer(feat0_down_scale2)

            feat0_down_scale4_tmp=self.feat0_down_scale4(feat0)
            feat0_down_scale4=self.feat0_down_scale4_chs_tl(feat0_down_scale4_tmp)
            feat0_down_scale4=self.feat0_down_scale4_chs_tl_refine_layer(feat0_down_scale4)

            if self.use_dilation and self.out_stride == 8:
                feat0_down_scale8 = self.feat0_down_scale8_chs_tl(feat0_down_scale4_tmp)
            else:
                feat0_down_scale8=self.feat0_down_scale8(feat0)
                feat0_down_scale8=self.feat0_down_scale8_chs_tl(feat0_down_scale8)
            feat0_down_scale8=self.feat0_down_scale8_chs_tl_refine_layer(feat0_down_scale8)


        if self.use_mutiscale_connection:
            fuse3 = self.fuse3_lambda0*feat0_down_scale8 + self.fuse3_lambda1*feat1_down_scale4 + self.fuse3_lambda2*feat2_down_scale2 + feat3_down_scale0 + feat4_up
        else:
            fuse3=feat3_down_scale0 + feat4_up
        fuse3=self.fuse3_chs_tl(fuse3)
        fuse3=self.fuse3_chs_tl_refine_layer(fuse3)
        if self.use_dilation and self.out_stride == 8:
            fuse3_up=fuse3
        else:
            fuse3_up=self.fuse3_chs_tl_refine_layer_up(fuse3)


        if self.use_mutiscale_connection:
            fuse2=self.fuse2_lambda0*feat0_down_scale4 + self.fuse2_lambda1*feat1_down_scale2 + feat2_down_scale0 + fuse3_up
        else:
            fuse2=feat2_down_scale0 + fuse3_up
        fuse2=self.fuse2_chs_tl(fuse2)
        fuse2=self.fuse2_chs_tl_refine_layer(fuse2)
        fuse2_up=self.fuse2_chs_tl_refine_layer_up(fuse2)
        disp2=self.disp2(fuse2_up)
        rec_img2=self.rec_img2(fuse2_up)

        if self.use_mutiscale_connection:
            fuse1=self.fuse1_lambda0*feat0_down_scale2 + feat1_down_scale0 + fuse2_up
        else:
            fuse1=feat1_down_scale0 + fuse2_up
        fuse1=self.fuse1_chs_tl(fuse1)
        fuse1=self.fuse1_chs_tl_refine_layer(fuse1) #64x64x208
        fuse1_up=self.fuse1_chs_tl_refine_layer_up(fuse1)
        disp1=self.disp1(fuse1_up)
        rec_img1=self.rec_img1(fuse1_up)


        fuse0=feat0_down_scale0 + fuse1_up
        fuse0=self.fuse0_chs_tl(fuse0)
        fuse0=self.fuse0_chs_tl_refine_layer(fuse0)#32x128x416
        fuse0_up=self.fuse0_chs_tl_refine_layer_up(fuse0)
        disp0=self.disp0(fuse0_up)

        rec_img0=self.rec_img0(fuse0_up)

        return {'disps':[disp0,disp1,disp2],'reduce_feat0':feat0_down_scale0,'reduce_feat_fuse0':fuse0,'feat4_reduce':feat4,'feat0':feat0,
                'rec_img':[rec_img0,rec_img1,rec_img2]}
if __name__ == '__main__':
    hqdec=HQDec(patch_size_list = [(16, 52), (16, 52), (16, 52), (16, 52), (8, 26)],
                 feat_size_list = [(128, 416), (64, 208), (32, 104), (16, 52), (8, 26)],
                 disp_size_list=[(256,832),(128,416),(64,208)],use_dilation=False,out_stride=8,use_mutiscale_connection=False)

    encoder_chs_num = np.array([32, 128, 256, 512, 512])
    f0 = torch.randn(2, encoder_chs_num[0], 128, 416)
    f1 = torch.randn(2, encoder_chs_num[1], 64, 208)
    f2 = torch.randn(2, encoder_chs_num[2], 32, 104)
    f3 = torch.randn(2, encoder_chs_num[3], 16,52)
    f4 = torch.randn(2, encoder_chs_num[4], 8,26)#
    out=hqdec([f0,f1,f2,f3,f4])

