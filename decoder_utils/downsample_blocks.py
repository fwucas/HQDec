# ！/usr/bin/python
# Create Time : 2022/4/22 下午3:57
# Author : FeiWang
# Email: fei.wang2@siat.ac.cn
import torch
import torch.nn as nn

from .general_blocks import ConvBlockGN,Channel_attention_block_Param
from timm.models.layers.weight_init import trunc_normal_
class DownSamplev3(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,feat_height=None,feat_width=None):
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


class Conv2d_zero_pad(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(Conv2d_zero_pad, self).__init__()
        if out_channels%16==0:
            num_groups=16
        elif out_channels%8==0:
            num_groups=8
        elif out_channels%4==0:
            num_groups=4
        elif out_channels%2==0:
            num_groups = 2
        else:
            num_groups=1
        self.conv2d=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size)
        self.norm=nn.GroupNorm(num_groups=num_groups,num_channels=out_channels)
        self.pad=nn.ConstantPad2d([kernel_size//2]*4,value=0)
        self.act=nn.ELU(inplace=True)

    def forward(self,x):
        x=self.pad(x)
        x=self.conv2d(x)
        x=self.norm(x)
        x=self.act(x)

        return x

class DownPixel(nn.Module):
    '''
    input(b,c,h,w)
    return: (b,c*(stride**2),h/stride,w/stride)
    '''
    def __init__(self,stride=2):
        super(DownPixel, self).__init__()
        self.stride=stride
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.contiguous().view(b, c, h//self.stride, self.stride, w//self.stride, self.stride)
        x=x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c*(self.stride **2), h//self.stride, w//self.stride)
        return x

class DownPixel_pos(nn.Module):
    '''
    input(b,c,h,w)
    return: (b,c*(stride**2),h/stride,w/stride)
     add position information
    '''
    def __init__(self,stride=2):
        super(DownPixel_pos, self).__init__()
        self.stride=stride
        self.pos = nn.Parameter(torch.zeros(1, 1, 1,stride, 1, stride))
        trunc_normal_(self.pos, std=.02)
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.contiguous().view(b, c, h//self.stride, self.stride, w//self.stride, self.stride)
        x=x+self.pos
        x=x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c*(self.stride **2), h//self.stride, w//self.stride)
        return x
class DownPixelH(nn.Module):
    def __init__(self,stride=2):
        super(DownPixelH, self).__init__()
        self.stride=stride
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.contiguous().view(b, c, h//self.stride, self.stride, w)
        x=x.permute(0, 1,3,2,4 ).contiguous().view(b,c*self.stride,h//self.stride,w)

        return x

class DownPixelW(nn.Module):
    def __init__(self,stride=2):
        super(DownPixelW, self).__init__()
        self.stride=stride
    def forward(self,x):
        b,c,h,w=x.shape
        x=x.contiguous().view(b, c, h, w//self.stride, self.stride)
        x = x.permute(0,1,4,2,3).contiguous().view(b,c*self.stride,h,w//self.stride)
        return x

class DownPixel_2DLayer(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,d=8):
        super(DownPixel_2DLayer, self).__init__()
        self.stride=stride
        self.conv=Conv2d_zero_pad(in_channels=in_chs*(self.stride ** 2)*d,out_channels=in_chs,kernel_size=3)
        self.downpixel=DownPixel(stride=self.stride)
        self.conv2d=nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(3,3),padding=(1,1))
    def forward(self,x):
        x=self.downpixel(x)
        b,c,h,w=x.shape
        x=x.view(b*c,h,w)
        x=x.unsqueeze(1)
        x=self.conv2d(x)
        x=x.view(b,-1,h,w)
        x=self.conv(x)
        return x
class DownPixel_3DLayer(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,d=8,feat_height=None,feat_width=None):
        super(DownPixel_3DLayer, self).__init__()
        self.stride=stride
        self.downpixel=DownPixel(stride=self.stride)
        self.conv2d=Conv2d_zero_pad(in_channels=in_chs*(self.stride**2 )*d,out_channels=in_chs,kernel_size=3)
        self.conv3d=nn.Conv3d(1,d,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1))
    def forward(self,x):
        x=self.downpixel(x)
        x=x.unsqueeze(1)
        x=self.conv3d(x)
        b,c,d,h,w=x.shape
        x = x.view(b, c * d, h, w)
        x = self.conv2d(x)
        return x



class DownPixel_2DLayer_Axis(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,d=8):
        super(DownPixel_2DLayer_Axis, self).__init__()
        self.downpixel_h=DownPixelH(stride=stride)
        self.downpixel_w=DownPixelW(stride=stride)

        self.conv_zero_pad_w=Conv2d_zero_pad(in_channels=in_chs*stride*d,out_channels=in_chs,kernel_size=3)
        self.conv_zero_pad_h=Conv2d_zero_pad(in_channels=in_chs*stride*d,out_channels=in_chs,kernel_size=3)
        self.conv_h=nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(3,3),padding=(1,1))
        self.conv_w=nn.Conv2d(in_channels=1,out_channels=d,kernel_size=(3,3),padding=(1,1))
    def forward(self,x):
        x=self.downpixel_w(x)
        b,c,h,w=x.shape
        x=x.view(b*c,h,w)
        x=x.unsqueeze(1)
        x=self.conv_w(x)# b2dhw
        x=x.view(b,-1,h,w)
        x=self.conv_zero_pad_w(x)

        x=self.downpixel_h(x)
        b1,c1,h1,w1=x.shape
        x=x.view(b1*c1,h1,w1)
        x=x.unsqueeze(1)
        x=self.conv_h(x)
        x=x.view(b1,-1,h1,w1)
        x = self.conv_zero_pad_w(x)

        return x
class DownPixel_3DLayer_Axis(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,d=8):
        super(DownPixel_3DLayer_Axis, self).__init__()
        self.downpixel_h = DownPixelH(stride=stride)
        self.downpixel_w = DownPixelW(stride=stride)

        self.conv_zero_pad_w = Conv2d_zero_pad(in_channels=in_chs * stride * d, out_channels=in_chs, kernel_size=3)
        self.conv_zero_pad_h = Conv2d_zero_pad(in_channels=in_chs * stride * d, out_channels=in_chs, kernel_size=3)
        self.conv3d_h = nn.Conv3d(in_channels=1, out_channels=d, kernel_size=(3, 3,3), padding=(1, 1,1))
        self.conv3d_w = nn.Conv3d(in_channels=1, out_channels=d, kernel_size=(3, 3,3), padding=(1, 1,1))

    def forward(self,x):
        x=self.downpixel_w(x)
        b,c,h,w=x.shape
        x = x.unsqueeze(1)
        x = self.conv3d_w(x)
        x=x.view(b,-1,h,w)
        x=self.conv_zero_pad_w(x)

        x=self.downpixel_h(x)
        b1,c1,h1,w1=x.shape
        x = x.unsqueeze(1)
        x=self.conv3d_h(x)
        x=x.view(b1,-1,h1,w1)
        x=self.conv_zero_pad_h(x)
        return x
        
        

class Down_Channel_Attv3_Param_groups_pos(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,feat_height=None,feat_width=None):
        super(Down_Channel_Attv3_Param_groups_pos, self).__init__()
        self.chs_att=Channel_attention_block_Param(channels=None,feat_height=feat_height,feat_width=feat_width)
        self.conv1=ConvBlockGN(in_channels=in_chs*(stride**2),out_channels=in_chs*(stride**2),groups=in_chs*(stride**2))
        self.conv2=ConvBlockGN(in_channels=in_chs*(stride**2),out_channels=in_chs*(stride**2),groups=in_chs*(stride**2))


        self.se1=SE(in_chs=in_chs*(stride**2),reduce_rate=1./16)
        self.conv1_pw=Conv2d_zero_pad(in_channels=in_chs*(stride**2),out_channels=in_chs*(stride**2),kernel_size=1)

        self.se2=SE(in_chs=in_chs*(stride**2),reduce_rate=1./16)
        self.conv2_pw=Conv2d_zero_pad(in_channels=in_chs*(stride**2),out_channels=in_chs*(stride**2),kernel_size=1)

        self.downpixel=DownPixel_pos(stride=stride)
        self.conv1x1=Conv2d_zero_pad(in_channels=in_chs*(stride**2),
                               out_channels=in_chs,kernel_size=3)
    def forward(self,x):
        out=self.downpixel(x)
        out=self.conv1(out)
        out=self.conv1_pw(self.se1(out))


        out=self.conv2(out)
        out=self.conv2_pw(self.se2(out))


        out=self.chs_att(out)
        out=self.conv1x1(out)
        return out
class AdaDown_Channel_Attv4_Param(nn.Module):
    def __init__(self,in_chs,kernel_size=2,stride=2,feat_height=None,feat_width=None):
        super(AdaDown_Channel_Attv4_Param, self).__init__()

        bs = 1
        self.param = nn.Parameter(
            torch.zeros((bs, in_chs, feat_height//stride, feat_width//stride), dtype=torch.float32, requires_grad=True, device='cuda:0'),
            requires_grad=True)

        self.maxdownsample = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        self.bn1=nn.BatchNorm2d(num_features=in_chs)
        self.bn2=nn.BatchNorm2d(num_features=in_chs)
        self.chs_att_down=Down_Channel_Attv3_Param_groups_pos(in_chs=in_chs,kernel_size=kernel_size,stride=stride,feat_height=feat_height//stride,feat_width=feat_width//stride)
        self.act=nn.ELU(True)
    def forward(self,x):

        max_pool_x=self.maxdownsample(x)
        max_pool_x=self.bn1(max_pool_x)



        att_chs_x=self.chs_att_down(x)
        att_chs_x=self.bn2(att_chs_x)

        out=self.param*max_pool_x+att_chs_x
        out=self.act(out)


        return out


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

if __name__ == '__main__':

    tmp=torch.randn(2,16,256,832)
    downsample=DownSamplev3(in_chs=16,kernel_size=4,stride=4)
    print(downsample(tmp).shape)
    down=DownPixel_2DLayer_Axis(in_chs=16,kernel_size=2,stride=4,d=8)#



