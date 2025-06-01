'''
3 key points
1:composition
2:contrastive nce
3:JSD


'''
import torch
import torch.nn as nn
from Model.ResNet3D import ResNetEncoder3D


class Composition(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        half_input_dim = int(input_dim) // 2
        self.mlp = nn.Linear(input_dim, half_input_dim)
        self.fc = nn.Linear(half_input_dim, num_class)

    def forward(self, x1, x2):
        '''
        :param x1: student
        :param x2: teacher
        :return:
        '''
        # x2 = x2.to(x1.device)
        # print(x2.device)
        residual_x = torch.cat((x2, x1), 1)
        # print(residual_x.device)
        residual_x = self.mlp(residual_x)
        feature_x = x2 + residual_x
        out_x = self.fc(feature_x)
        return out_x, feature_x


def load_dict(model1, model_pre_path, devices):
    model_pre = torch.load(model_pre_path, map_location=devices)
    dict = {"ResNetEncoder3D" + ".".join(str(k).split('.')): v for k, v in model_pre.items() if "img_res." in str(k)}
    model_state = model1.state_dict()
    model_state.update(dict)
    model1.load_state_dict(model_state)
    # print(model1)
    return model1


class CCL(nn.Module):
    def __init__(self, devices):
        super(CCL, self).__init__()
        self.res_x = ResNetEncoder3D()
        pretrained_SUA = "/home/bocao/plaque/vkd_segpre_multiclassification/trained/2024_02_13_21_22_32-SUA_res50_with_linear_freez_7e_5/vkd_MRI_epoch_96_time_2024_02_13--21_45.pth"
        pretrained_SSK = "/home/bocao/plaque/vkd_segpre_multiclassification/trained/2024_01_27_10_14_04-SSK_res50/vkd_MRI_epoch_148_time_2024_01_27--12_40.pth"
        self.res_x = load_dict(self.res_x, pretrained_SUA, devices)
        self.res_seg_x = ResNetEncoder3D()
        self.res_seg_x = load_dict(self.res_seg_x, pretrained_SSK, devices)
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
        # self.CCL_loss = CCL_loss()
        self.com_feature_x = Composition(input_dim=1024, num_class=2)
        self.com_feature_t = Composition(input_dim=1024, num_class=2)

    def forward(self, x1, x2, t, y):
        x1 = self.res_x(x1)
        x1 = x1.view(-1, 2048, 2048)
        x1 = self.adapter_x(torch.squeeze(self.pool(x1)))
        # print(x1.size())
        x2 = self.adapter_seg_x(torch.squeeze(self.pool(self.res_seg_x(x2).view(-1, 2048, 2048))))
        t = self.adapter_text(torch.squeeze(self.pool(t)))
        # loss = self.CCL_loss(x1, x2, t, y)
        out_x, feature_x = self.com_feature_x(x1, x2)
        out_t, feature_t = self.com_feature_t(x1, t)

        y_pred = self.cls(x1)
        y_pred_seg = self.cls_seg_x(x2)
        y_pred_t = self.cls_t(t)

        # return loss + loss_cls
        # print(" forward: ", loss)
        # return {"loss": loss, "Reconstruction_Loss": loss_cls}
        return y_pred, y_pred_seg, y_pred_t, y, out_x, out_t

    '''
     must divide the forward and loss 
    '''

    def losses(self, y_pred, y_pred_segx, y_pred_t, y, out_x, out_t):
        # print("losses ")
        # loss_jsd = self.JSDloss(out_x, out_t)
        loss_nce = self.Nce(out_x, out_t, y)
        loss_cls = self.BCEloss(y_pred, y)
        loss_cls_seg_x = self.BCEloss(y_pred_segx, y)
        loss_cls_t = self.BCEloss(y_pred_t, y)
        # print(loss, loss_cls)
        loss = loss_cls + loss_nce * 0.05 + loss_cls_t + loss_cls_seg_x
        return {"loss": loss, "Reconstruction_Loss": loss_cls, "nce": loss_nce}

    def eval_x(self, x, y):
        x1 = self.adapter_x(torch.squeeze(self.pool(self.res_x(x).view(-1, 2048, 2048))))
        x1 = self.cls(x1)
        loss_cls = self.BCEloss(x1, y)
        return x1, loss_cls

    def Nce(self, f1, f2, targets):
        import torch.nn.functional as F

        EPISILON = 1e-10
        temperature = 0.5
        softmax = nn.Softmax(dim=1)

        # def where(self, cond, x_1, x_2):
        #     cond = cond.type(torch.float32)
        #     return (cond * x_1) + ((1 - cond) * x_2)

        ### cuda implementation
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        ## set distances of the same label to zeros
        targets = torch.argmax(targets, 1)
        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)

        ## convert l2 distance to cos distance
        cos = 1 - 0.5 * dist

        ## convert cos distance to exponential space
        pred_softmax = softmax(cos / temperature)  ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + EPISILON) * ~self_mask.int()
        log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.int()
        log_softmax = log_pos_softmax.sum(1) / ~self_mask.sum(1).int() + log_neg_softmax.sum(
            1) / self_mask.sum(1).int()
        loss = log_softmax

        return loss.mean()

    def JSDloss(self, p, q, weight=1.0):
        eps = 1e-10
        loss1 = (q * torch.log(q / p + eps)).sum(1)
        loss2 = (p * torch.log(p / q + eps)).sum(1)
        loss = (loss2 + loss1).mean() * weight
        return loss


# class CCL_loss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.com_feature_x = Composition(input_dim=1024, num_class=2)
#         self.com_feature_t = Composition(input_dim=1024, num_class=2)
#
#     def forward(self, x1, x2, t, y):
#         '''
#
#         :param x1: student
#         :param x2: teacher img
#         :param t: teacher text
#         :param y:
#         :return:
#         '''
#
#         out_x, feature_x = self.com_feature_x(x1, x2)
#         out_t, feature_t = self.com_feature_t(x1, t)
#         loss_jsd = self.JSDloss(out_x, out_t)
#         loss_nce = self.Nce(out_x, out_t, y)
#         # print("jsd,nce", loss_nce, loss_nce)
#         return loss_jsd + loss_nce
#
#     def Nce(self, f1, f2, targets):
#         import torch.nn.functional as F
#
#         EPISILON = 1e-10
#         temperature = 0.5
#         softmax = nn.Softmax(dim=1)
#
#         # def where(self, cond, x_1, x_2):
#         #     cond = cond.type(torch.float32)
#         #     return (cond * x_1) + ((1 - cond) * x_2)
#
#         ### cuda implementation
#         f1 = F.normalize(f1, dim=1)
#         f2 = F.normalize(f2, dim=1)
#
#         ## set distances of the same label to zeros
#         targets = torch.argmax(targets, 1)
#         mask = targets.unsqueeze(1) - targets
#         self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
#         dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)
#
#         ## convert l2 distance to cos distance
#         cos = 1 - 0.5 * dist
#
#         ## convert cos distance to exponential space
#         pred_softmax = softmax(cos / temperature)  ### convert to multi-class prediction scores
#
#         log_pos_softmax = - torch.log(pred_softmax + EPISILON) * ~self_mask.int()
#         log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.int()
#         log_softmax = log_pos_softmax.sum(1) / ~self_mask.sum(1).int() + log_neg_softmax.sum(
#             1) / self_mask.sum(1).int()
#         loss = log_softmax
#
#         return loss.mean()
#
#     def JSDloss(self, p, q, weight=1.0):
#         eps = 1e-10
#         loss1 = (q * torch.log(q / p + eps)).sum(1)
#         loss2 = (p * torch.log(p / q + eps)).sum(1)
#         loss = (loss2 + loss1).mean() * weight
#         return loss

if __name__ == '__main__':
    c = CCL()
