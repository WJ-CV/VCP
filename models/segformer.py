import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import models
from models import register
from mmcv.runner import build_runner
import math
import numpy as np
from torch.autograd import Variable
import mmcv
from .mmseg.models import build_segmentor
from mmseg.models import backbones
from mmseg.models.builder import BACKBONES, SEGMENTORS
import os
import logging
logger = logging.getLogger(__name__)
from .iou_loss import IOU
import random
from torch.nn import init
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import thop

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    count = (weit > 1).sum().item()
    sum_of_values = torch.sum(weit[weit > 1])
    foreground_weight = sum_of_values // count
    weit = torch.where(mask == 1, foreground_weight * weit, weit)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def structure_loss1(pred_list, mask_ori):
    structure_loss = 0
    for i in range(4):
        pred = pred_list[i]
        mask = F.interpolate(mask_ori, pred.size()[2:], mode='bilinear', align_corners=True)
        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        count = (weit > 1).sum().item()
        sum_of_values = torch.sum(weit[weit > 1])
        foreground_weight = sum_of_values // count
        weit = torch.where(mask == 1, foreground_weight * weit, weit)

        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)
        structure_loss += (wbce + wiou).mean()
    return structure_loss

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

@register('segformer')
class SegFormer(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if encoder_mode['name'] == 'vpt_deep':
            backbone = dict(
                type=BACKBONES.get('mit_b4_vpt'),
                img_size=inp_size,
                prompt_cfg=encoder_mode['name'],)
        elif encoder_mode['name'] == 'fp':
            backbone = dict(
                type=BACKBONES.get('mit_b4_fp'),
                img_size=inp_size,
                scale_factor=encoder_mode['scale_factor'],
                tuning_stage=encoder_mode['tuning_stage'],
                frequency_tune=encoder_mode['frequency_tune'],
                embedding_tune=encoder_mode['embedding_tune'],
                adaptor=encoder_mode['adaptor'])
        elif encoder_mode['name'] == 'vcp':
            backbone = dict(
                type=BACKBONES.get('mit_b4_vcp'),
                img_size=inp_size,
                scale_factor=encoder_mode['scale_factor'],
                prompt_type=encoder_mode['prompt_type'],
                tuning_stage=encoder_mode['tuning_stage'],
                input_type=encoder_mode['input_type'],
                freq_nums=encoder_mode['freq_nums'],
                handcrafted_tune=encoder_mode['handcrafted_tune'],
                embedding_tune=encoder_mode['embedding_tune'],
                adaptor=encoder_mode['adaptor'])
        elif encoder_mode['name'] == 'linear':
            backbone = dict(
                type=BACKBONES.get('mit_b4'),
                img_size=inp_size)
        elif encoder_mode['name'] == 'adaptformer':
            backbone = dict(
                type=BACKBONES.get('mit_b4_adaptformer'),
                img_size=inp_size)
        else:
            backbone = dict(
                type=BACKBONES.get('mit_b4'),
                img_size=inp_size)

        model_config = dict(
            type='EncoderDecoder',
            pretrained=encoder_mode['pretrained'],
            backbone=backbone,
            decode_head=dict(
                type='SegFormerHead',
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                num_classes=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                align_corners=False,
                decoder_params=dict(embed_dim=128),
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
            # model training and testing settings
            train_cfg=dict(),
            test_cfg=dict(mode='whole'))

        print('loading segformer weigths...')
        model = build_segmentor(
            model_config,
            # train_cfg=dict(),
            # test_cfg=dict(mode='whole')
        )

        self.encoder = model

        if encoder_mode['name'] == 'vcp':
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "decode_head" not in k:
                    p.requires_grad = False
        if encoder_mode['name'] == 'vpt_deep':
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "decode_head" not in k:
                    p.requires_grad = False
        if encoder_mode['name'] == 'adaptformer':
            for k, p in self.encoder.named_parameters():
                if "adaptmlp" not in k and "decode_head" not in k:
                    p.requires_grad = False
        if encoder_mode['name'] == 'linear':
            for k, p in self.encoder.named_parameters():
                if "decode_head" not in k:
                    p.requires_grad = False

        self.STR_w = encoder_mode['STR_w']
        self.BCE_w = encoder_mode['BCE_w']
        self.IOU_w = encoder_mode['IOU_w']
        self.Class_w1 = encoder_mode['Class_w1']
        self.Class_w2 = encoder_mode['Class_w2']

        model_total_params = sum(p.numel() for p in self.encoder.parameters())
        model_grad_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params),
              '\nmodel_total_params:' + str(model_total_params))

        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()


    def set_input(self, input, gt_mask, class_gt):              ##############1
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.class_gt = class_gt.to(self.device)

    def forward(self):
        self.pred_mask, self.pred_class, self.initial_prediction = self.encoder.forward_dummy(self.input, train=True)

    def backward_G(self):
        self.loss_STR = self.STR_w * structure_loss(self.pred_mask, self.gt_mask)
        self.loss_BCE = self.BCE_w * self.criterionBCE(self.pred_mask, self.gt_mask) + self.IOU_w * _iou_loss(self.pred_mask, self.gt_mask)
        self.loss_iou = 2 * structure_loss1(self.initial_prediction, self.gt_mask)
        self.cls_loss =  0.5 * F.cross_entropy(self.pred_class, self.class_gt)

        # self.cls_loss = self.Class_w1 * F.cross_entropy(self.pred_class, self.class_gt) + self.Class_w2 * F.cross_entropy(self.handcrafted4_class, self.class_gt)

        self.loss_G = self.loss_STR + self.loss_BCE + self.loss_iou + self.cls_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
