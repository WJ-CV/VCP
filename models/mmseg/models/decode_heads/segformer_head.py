import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
from torch.nn import Conv2d, Parameter, Softmax
from torch.nn import functional as F

from IPython import embed
def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU()
    )
def convblock2(in_ch, out_ch, rate):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, dilation=rate, padding=rate),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class EnLayer(nn.Module):
    def __init__(self, in_channel=64, mid_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x

class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=64):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x

class GlobalInfo(nn.Module):
    def __init__(self, in_1):
        super(GlobalInfo, self).__init__()
        self.convd1 = convblock2(in_1, 64, 1)
        self.convd2 = convblock2(in_1, 64, 2)
        self.convd3 = convblock2(in_1, 64, 4)
        self.convd4 = convblock2(in_1, 64, 6)
        self.convd5 = convblock(in_1, 64, 1, 1, 0)
        self.fus = convblock(64*5, 64, 3, 1, 1)
        # self.gfm = GlobalFM(64)

    def forward(self, rgb):
        out1 = self.convd1(rgb)
        out2 = self.convd2(rgb)
        out3 = self.convd3(rgb)
        out4 = self.convd4(rgb)
        out5 = F.interpolate(self.convd5(F.adaptive_avg_pool2d(rgb, 2)), rgb.size()[2:], mode='bilinear', align_corners=True)
        out = self.fus(torch.cat((out1, out2, out3, out4, out5),1))
        return out

@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        in_c_list = [c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels]

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']   #768

        linear_layers = []
        for idx in range(4):
            linear_layers.append(MLP(input_dim=in_c_list[idx], embed_dim=embedding_dim))
        self.linear_layers = nn.ModuleList(linear_layers)

        self.glo_ASPP = GlobalInfo(embedding_dim)
        # self.glo_ASPP = LatLayer(in_channel=embedding_dim, mid_channel=64)

        CA_layers = []
        for idx in range(4):
            CA_layers.append(CA(embedding_dim))
        self.CA_layers = nn.ModuleList(CA_layers)

        lat_layers = []
        for idx in range(4):
            lat_layers.append(LatLayer(in_channel=embedding_dim, mid_channel=64))
        self.lat_layers = nn.ModuleList(lat_layers)

        dec_layers = []
        for idx in range(4):
            dec_layers.append(EnLayer(in_channel=64, mid_channel=64))
        self.dec_layers = nn.ModuleList(dec_layers)

        self.sig = nn.Sigmoid()
        self.conv_pre = convblock(64, 64, 3, 1, 1)
        self.pre_out = convblock(64, 1, 1, 1, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 291 + 78)  # DUTS_class + COCO9k has 291+65 classes

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape
        rgb_f = [c1, c2, c3, c4]

        for idx in range(4):
            rgb_f[idx] = self.linear_layers[idx](rgb_f[idx]).permute(0,2,1).reshape(n, -1, rgb_f[idx].shape[2], rgb_f[idx].shape[3])

        for idx in range(4):
            rgb_f[idx] = self.CA_layers[idx](rgb_f[idx])

        # feat_ISP = self.glo_ASPP(rgb_f[-1])
        feat_ISP = self.glo_ASPP(rgb_f[-1])

        feat_down = []
        for idx in range(4):
            p = self.lat_layers[idx](rgb_f[idx])
            feat_down.append(self.upsample_add_sig(p, feat_ISP))

        up_3 = self.dec_layers[3](feat_down[3])
        up_2 = self.dec_layers[2](self.upsample_add(feat_down[2], up_3))
        up_1 = self.dec_layers[1](self.upsample_add(feat_down[1], up_2))
        up_0 = self.dec_layers[0](self.upsample_add(feat_down[0], up_1))

        pre_class = self.classifier(self.avgpool(up_3).view(n, -1))

        prediction_map = self.pre_out(self.conv_pre(up_0))

        return prediction_map, pre_class

    def upsample_add_sig(self, x, y):
        up_y = F.interpolate(y, x.size()[2:], mode='bilinear')
        return x + x * self.sig(up_y)

    def upsample_add(self, x, y):
        up_y = F.interpolate(y, x.size()[2:], mode='bilinear')
        return x + up_y