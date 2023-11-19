import torch.nn as nn
import torch

def dpc_conv(in_dim, reduction_dim, dil, separable):
    if separable:
        groups = reduction_dim
    else:
        groups = 1

    return nn.Sequential(
        nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=dil,
                  padding=dil, bias=False, groups=groups),
        nn.BatchNorm2d(reduction_dim),
        nn.ReLU(inplace=True)
    )

class DPC(nn.Module):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16,
                 rates=[(1, 6), (18, 15), (6, 21), (1, 1), (6, 3)],
                 dropout=False, separable=False):
        super(DPC, self).__init__()

        self.dropout = dropout
        if output_stride == 8:
            rates = [(2 * r[0], 2 * r[1]) for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.a = dpc_conv(in_dim, reduction_dim, rates[0], separable)
        self.b = dpc_conv(reduction_dim, reduction_dim, rates[1], separable)
        self.c = dpc_conv(reduction_dim, reduction_dim, rates[2], separable)
        self.d = dpc_conv(reduction_dim, reduction_dim, rates[3], separable)
        self.e = dpc_conv(reduction_dim, reduction_dim, rates[4], separable)

        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        a = self.a(x)
        b = self.b(a)
        c = self.c(a)
        d = self.d(a)
        e = self.e(b)
        out = torch.cat((a, b, c, d, e), 1)
        if self.dropout:
            out = self.drop(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)
    def forward(self, x):
        out = self.conv(x)
        return out


class CRPBlock(nn.Module):  #
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i + 1, 'pointwise'), Conv1x1(in_planes if (i == 0) else out_planes, out_planes, False))
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x

        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'pointwise'))(top)
            x = top + x
        return x

def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()





class AttModule(nn.Module):
    def __init__(self,feat0_chs,scale_num):
        super(AttModule, self).__init__()
        self.attnet=nn.Sequential(nn.Conv2d(in_channels=feat0_chs*scale_num,out_channels=256,kernel_size=3,padding=1,bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=256,out_channels=scale_num,kernel_size=1,padding=0,bias=False),
                                  nn.Sigmoid())

    def forward(self,x):
        out=self.attnet(x)
        out=out/torch.sum(out,dim=1,keepdim=True)
        return out


class AttModuleShare(nn.Module):
    def __init__(self,feat0_chs):
        super(AttModuleShare, self).__init__()
        self.attnet=nn.Sequential(nn.Conv2d(in_channels=feat0_chs*2,out_channels=256,kernel_size=3,padding=1,bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels=256,out_channels=2,kernel_size=1,padding=0,bias=False),
                                  nn.Sigmoid())

    def forward(self,x):
        out=self.attnet(x)
        out=out/torch.sum(out,dim=1,keepdim=True)
        return out

