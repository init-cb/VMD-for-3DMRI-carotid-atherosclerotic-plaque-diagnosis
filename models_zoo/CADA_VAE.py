import torch.nn
import torch.nn as nn
from Model.ResNet3D import ResNetEncoder3D


def load_dict(model1, model_pre_path, devices):
    model_pre = torch.load(model_pre_path, map_location=devices)
    dict = {"ResNetEncoder3D" + ".".join(str(k).split('.')): v for k, v in model_pre.items() if "img_res." in str(k)}
    model_state = model1.state_dict()
    model_state.update(dict)
    model1.load_state_dict(model_state)
    # print(model1)
    return model1


class EncoderTemplate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EncoderTemplate, self).__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.var = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        mu = self.mu(x)
        var = self.var(x)
        return mu, var


class DecoderTemplate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecoderTemplate, self).__init__()

        self.cls = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, z):
        return self.cls(z)


class CADAVAE(torch.nn.Module):
    def __init__(self, devices):
        super(CADAVAE, self).__init__()
        self.res_x = ResNetEncoder3D()
        pretrained_unseg = "/home/bocao/plaque/vkd_segpre_multiclassification/trained/" \
                           "2024_02_12_00_05_09-SUA_res50_pure_/vkd_MRI_epoch_189_time_2024_02_12--01_15.pth"
        pretrained_seg = "/home/bocao/plaque/vkd_segpre_multiclassification/trained/" \
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
        self.cls_loss = nn.BCEWithLogitsLoss()
        # self.cls_loss = nn.L1Loss(size_average=False)
        self.cls = nn.Linear(512, 2)
        self.cls_seg_x = nn.Linear(512, 2)
        self.cls_t = nn.Linear(512, 2)

        self.encoder_s = EncoderTemplate(2048, 512)
        self.encoder_t = EncoderTemplate(2048, 512)
        self.encoder_r = EncoderTemplate(768, 512)

        self.decoder_s = DecoderTemplate(512, 2)
        self.decoder_t = DecoderTemplate(512, 2)
        self.decoder_r = DecoderTemplate(512, 2)

    def forward(self, x, seg_x, report, y):
        x = torch.squeeze(self.pool(self.res_x(x).view(-1, 2048, 2048)))
        seg_x = torch.squeeze(self.pool(self.res_seg_x(seg_x).view(-1, 2048, 2048)))
        report = torch.squeeze(self.pool(report))

        mu_s, var_s = self.encoder_s(x)
        mu_t, var_t = self.encoder_t(seg_x)
        mu_r, var_r = self.encoder_r(report)
        z_s = self.reparameterize(mu_s, var_s)
        z_t = self.reparameterize(mu_t, var_t)
        z_r = self.reparameterize(mu_r, var_r)
        s_cls_from_s = self.decoder_s(z_s)
        t_cls_from_t = self.decoder_t(z_t)
        r_cls_from_r = self.decoder_r(z_r)

        t_cls_from_s = self.decoder_s(z_t)
        s_cls_from_t = self.decoder_t(z_s)

        r_cls_from_s = self.decoder_s(z_r)
        s_cls_from_r = self.decoder_r(z_s)

        params = [mu_s, var_s, mu_t, var_t, mu_r, var_r, z_s, z_t, z_r, s_cls_from_s, t_cls_from_t, r_cls_from_r,
                  t_cls_from_s, s_cls_from_t, r_cls_from_s, s_cls_from_r]
        return params

    def eval_x(self, x, y):
        x = torch.squeeze(self.pool(self.res_x(x).view(-1, 2048, 2048)))
        mu_s, var_s = self.encoder_s(x)
        z_s = self.reparameterize(mu_s, var_s)
        s_cls_from_s = self.decoder_s(z_s)
        loss_recon_s = self.cls_loss(s_cls_from_s, y)
        return s_cls_from_s, loss_recon_s

    def reparameterize(self, mu, logvar):
        # if self.reparameterize_with_noise:
        sigma = torch.exp(logvar)
        eps = torch.cuda.FloatTensor(logvar.size()[0], 1).normal_(0, 1)
        eps = eps.expand(sigma.size())
        return mu + sigma * eps
        # else:
        #     return mu

    def losses(self, param, y):
        mu_s, var_s, mu_t, var_t, mu_r, var_r, z_s, z_t, z_r, s_cls_from_s, t_cls_from_t, r_cls_from_r, \
        t_cls_from_s, s_cls_from_t, r_cls_from_s, s_cls_from_r = param
        loss_recon_s = self.cls_loss(s_cls_from_s, y)
        loss_recon_t = self.cls_loss(t_cls_from_t, y)
        loss_recon_r = self.cls_loss(r_cls_from_r, y)

        loss_recon = loss_recon_t + loss_recon_r + loss_recon_s
        ##############################################
        # Cross Reconstruction Loss
        ##############################################

        cross_loss_recon_st = self.cls_loss(s_cls_from_t, y)
        cross_loss_recon_sr = self.cls_loss(s_cls_from_r, y)
        cross_loss_recon_ts = self.cls_loss(t_cls_from_s, y)
        cross_loss_recon_rs = self.cls_loss(r_cls_from_s, y)

        loss_cross_recon = cross_loss_recon_st + cross_loss_recon_ts + cross_loss_recon_rs + cross_loss_recon_sr

        ##############################################
        # KLD Loss
        ##############################################
        KLD_ts = (0.5 * torch.sum(1 + var_t - mu_t.pow(2) - var_t.exp())) \
                 + (0.5 * torch.sum(1 + var_s - mu_s.pow(2) - var_s.exp()))

        KLD_rs = (0.5 * torch.sum(1 + var_r - mu_r.pow(2) - var_r.exp())) \
                 + (0.5 * torch.sum(1 + var_s - mu_s.pow(2) - var_s.exp()))

        loss_kld = KLD_rs + KLD_ts

        ##############################################
        # distance Loss
        ##############################################

        distance_ts = torch.sqrt(torch.sum((mu_s - mu_t) ** 2, dim=1) + \
                                 torch.sum((torch.sqrt(var_s.exp()) - torch.sqrt(var_t.exp())) ** 2, dim=1)).sum()
        distance_rs = torch.sqrt(torch.sum((mu_s - mu_r) ** 2, dim=1) + \
                                 torch.sum((torch.sqrt(var_s.exp()) - torch.sqrt(var_r.exp())) ** 2, dim=1)).sum()

        loss_distance = distance_rs + distance_ts
        loss = loss_recon + loss_cross_recon - loss_kld

        if loss_cross_recon > 0:
            loss += loss_cross_recon
        if loss_distance > 0:
            loss += loss_distance

        return {"loss": loss, "Reconstruction_Loss": loss_recon_s, "seg_cls_loss": loss_recon_t,
                "t_cls_loss": loss_recon_r, "loss_recon": loss_recon, "loss_cross_recon": loss_cross_recon,
                "loss_kld": loss_kld, "loss_distance": loss_distance}
