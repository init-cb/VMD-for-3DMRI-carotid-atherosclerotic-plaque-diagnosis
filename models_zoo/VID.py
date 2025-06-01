from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Model.ResNet3D import ResNetEncoder3D


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""

    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, input, target):
        # pool for dimentsion match
        s_H, t_H = input.shape[2], target.shape[2]
        if s_H > t_H:
            input = F.adaptive_avg_pool2d(input, (t_H, t_H))
        elif s_H < t_H:
            target = F.adaptive_avg_pool2d(target, (s_H, s_H))
        else:
            pass
        pred_mean = self.regressor(input)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)
        neg_log_prob = 0.5 * (
                (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )
        loss = torch.mean(neg_log_prob)
        return loss


def load_dict(model1, model_pre_path, devices):
    model_pre = torch.load(model_pre_path, map_location=devices)
    dict = {"ResNetEncoder3D" + ".".join(str(k).split('.')): v for k, v in model_pre.items() if "img_res." in str(k)}
    model_state = model1.state_dict()
    model_state.update(dict)
    model1.load_state_dict(model_state)
    # print(model1)
    return model1


class VID(nn.Module):
    def __init__(self, devices):
        super(VID, self).__init__()
        self.res_x = ResNetEncoder3D()
        pretrained_unseg = "/data2_vision9/cb/VKD/trained/" \
                         "2024_02_12_00_05_09-SUA_res50_pure_/vkd_MRI_epoch_189_time_2024_02_12--01_15.pth"
        pretrained_seg = "/data2_vision9/cb/VKD/trained/" \
                         "2024_02_11_18_48_12-SSA_res50_pure_/vkd_MRI_epoch_88_time_2024_02_11--19_40.pth"
        self.res_x = load_dict(self.res_x, pretrained_unseg, devices)
        self.res_seg_x = ResNetEncoder3D()
        self.res_seg_x = load_dict(self.res_seg_x, pretrained_seg, devices)
        self.adapter_x = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.adapter_seg_x = nn.Sequential(
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.adapter_text = nn.Sequential(
            nn.Linear(768, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.cls = nn.Linear(512, 2)
        self.cls_seg_x = nn.Linear(512, 2)
        self.cls_t = nn.Linear(512, 2)

        self.vid_rs = VIDLoss(1, 1, 1)
        self.vid_ts = VIDLoss(1, 1, 1)

    def forward(self, x, seg_x, t):
        x_feature = self.res_x(x).view(-1, 2048, 2048)
        x = self.adapter_x(torch.squeeze(self.pool(x_feature)))
        # seg_x = self.res_seg_x(seg_x)
        seg_x_feature = self.res_seg_x(seg_x).view(-1, 2048, 2048)
        seg_x = self.adapter_seg_x(torch.squeeze(self.pool(seg_x_feature)))
        t_feature = t
        t = self.adapter_text(torch.squeeze(self.pool(t)))

        cls_x = self.cls(x)
        cls_seg_x = self.cls_seg_x(seg_x)
        cls_t = self.cls_t(t)

        return x_feature, seg_x_feature, t_feature, cls_x, cls_seg_x, cls_t

    def eval_x(self, x, y):
        x1 = self.adapter_x(torch.squeeze(self.pool(self.res_x(x).view(-1, 2048, 2048))))
        x1 = self.cls(x1)
        loss_cls = self.BCEloss(x1, y)
        return x1, loss_cls

    def losses(self, x, seg_x, t, cls_x, cls_seg_x, cls_t, y):
        loss_cls_x = self.BCEloss(cls_x, y)
        loss_cls_seg_x = self.BCEloss(cls_seg_x, y)
        loss_cls_t = self.BCEloss(cls_t, y)

        x = torch.unsqueeze(x, 1)
        seg_x = torch.unsqueeze(seg_x, 1)
        t = torch.unsqueeze(t, 1)
        loss_vid_rs = self.vid_rs(x, t)
        loss_vid_ts = self.vid_ts(x, seg_x)

        loss = loss_vid_rs + loss_cls_x + loss_cls_t + loss_cls_seg_x + loss_vid_ts
        return {"loss": loss, "Reconstruction_Loss": loss_cls_x,
                "seg_cls_loss": loss_cls_x, "t_cls_loss": loss_cls_t,
                "loss_vid_rs": loss_vid_rs, "loss_vid_ts": loss_vid_ts}
