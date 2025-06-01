import torch.nn as nn
import torch
from utils.attention import PositionalEncoding, BertPooler, TransformerEncoderLayer
from utils.generate_3DresNet import generate_model as get_res


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class Block_ResNet3D(nn.Module):
    def __init__(self,
                 IS_BRATS,
                 latent_dim,
                 n_feature_maps,
                 feature_map_size,
                 devices,
                 dropout_rate=0.5,
                 batch_size=1,
                 is_eval=False, IS_test=False):
        super(Block_ResNet3D, self).__init__()
        self.IS_BRATS = IS_BRATS
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.BCEloss = nn.BCEWithLogitsLoss()

        if IS_BRATS:
            self.pool3d = nn.AdaptiveAvgPool3d((16, 16, 8))
        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size
        self.n_feature_maps = n_feature_maps
        self.img_res = get_res(
            pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
            no_seg=True, train=True, map_devices=devices)

        self.net_linear = nn.Sequential(
            nn.Linear(self.feature_map_size, self.feature_map_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_map_size, self.feature_map_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fcp_mu = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fcp_var = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        modules_p = []
        dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
        in_channels = int(self.latent_dim)
        for d_dim in dense_dims:
            modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU()
                )
            )
            in_channels = d_dim
        modules_p.append(
            nn.Sequential(
                nn.Linear(in_channels, 2)
            ))
        self.generation_p = nn.Sequential(*modules_p)
        self.IS_test = IS_test

    def forward(self, x):
        t1 = torch.unsqueeze(x, 1)
        t1 = self.img_res(t1)
        if self.IS_BRATS:
            t1 = self.pool3d(t1)
        t1 = t1.view(-1, 2048, 2048)
        t1 = self.net_linear(t1)
        out_t1 = torch.squeeze(self.pool(t1))
        mu_t1 = self.fcp_mu(out_t1)
        logvar_t1 = self.fcp_var(out_t1)
        z_t1 = self.reparameterize(mu_t1, logvar_t1)
        y_pred_t1 = self.generation_p(z_t1)
        if self.IS_test:
            return [y_pred_t1, z_t1]
        paras = [y_pred_t1, z_t1, mu_t1, logvar_t1]
        return paras

    def loss_MI_img(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001
        tau = 0.5
        y_pred_t1 = paras[0]
        z_t1 = paras[1]
        mu_t1 = paras[2]
        logvar_t1 = paras[3]
        recons_img_loss = self.BCEloss(y_pred_t1, y)
        std_img = torch.exp(0.5 * logvar_t1)

        recons_img_loss = self.BCEloss(y_pred_t1, y)
        kld_loss = self.kl_loss_single(mu_t1, logvar_t1)

        loss = recons_img_loss * recon_weight + kld_loss * kld_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_img_loss, 'KLD': kld_loss}

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            eps_mu = torch.randn_like(mu)
            z += eps * std + mu
        return z * 0.01

    def kl_loss_single(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


class Block_Report(nn.Module):
    def __init__(self,
                 IS_BRATS,
                 latent_dim,
                 bert_embed_size,
                 max_num_words,
                 dropout_rate=0.5,
                 batch_size=1,
                 is_eval=False, IS_test=False):
        super(Block_Report, self).__init__()
        self.IS_BRATS = IS_BRATS
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.latent_dim = latent_dim
        self.token_dim = bert_embed_size  # 768
        self.max_num_words = max_num_words
        self.pe_t = PositionalEncoding(self.token_dim, max_len=self.max_num_words)  # 768 * 77
        self.transformer_t = TransformerEncoderLayer(d_model=self.token_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mu = nn.Sequential(
            nn.Linear(self.token_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.token_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate))
        modules = []
        dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
        in_channels = self.latent_dim
        for d_dim in dense_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU()
                )
            )
            in_channels = d_dim
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, 2)
            ))

        self.generation = nn.Sequential(*modules)
        self.IS_test = IS_test

    def forward(self, x):
        text_features = x
        text_features = self.transformer_t(self.pe_t(x))
        out_r = torch.squeeze(self.pool(text_features))
        mu_r = self.fc_mu(out_r)
        logvar_r = self.fc_var(out_r)
        z_r = self.reparameterize(mu_r, logvar_r)
        y_predr = self.generation(z_r)
        if self.IS_test:
            return [y_predr, z_r]
        paras = [y_predr, z_r, mu_r, logvar_r]
        return paras

    def loss_MI_text(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1e-3
        w_weight = 0.001
        tau = 0.5
        y_predr = paras[0]
        z_t1 = paras[1]
        mu = paras[2]
        logvar_r = paras[3]

        # std_text = torch.exp(0.5 * logvar_r)
        recons_text_loss = self.BCEloss(y_predr, y)
        kld_loss = self.kl_loss_single(mu, logvar_r)

        loss = kld_loss * kld_weight + recons_text_loss * recon_weight_r
        return {'loss': loss, 'KLD': kld_loss, 'Recons_r': recons_text_loss}

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            eps_mu = torch.randn_like(mu)
            z += eps * std + mu
        return z * 0.01

    def kl_loss_single(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


class Block_SegResNet3D(nn.Module):
    def __init__(self,
                 IS_BRATS,
                 latent_dim,
                 n_feature_maps,
                 feature_map_size,
                 devices,
                 dropout_rate=0.5,
                 batch_size=1,
                 is_eval=False, IS_test=False):
        super(Block_SegResNet3D, self).__init__()
        self.BCEloss = nn.BCEWithLogitsLoss()

        self.IS_BRATS = IS_BRATS
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        if IS_BRATS:
            self.pool3d = nn.AdaptiveAvgPool3d((16, 16, 8))
        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size
        self.n_feature_maps = n_feature_maps
        self.SSK_img_img_res = get_res(
            pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
            no_seg=True, train=True, map_devices=devices)

        self.SSK_img_net_linear = nn.Sequential(
            nn.Linear(self.feature_map_size, self.feature_map_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_map_size, self.feature_map_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fcp_mu = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fcp_var = nn.Sequential(
            nn.Linear(self.n_feature_maps, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate))

        SSK_img_modules_p = []
        dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
        in_channels = int(self.latent_dim)
        for d_dim in dense_dims:
            SSK_img_modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, d_dim),
                    nn.BatchNorm1d(d_dim),
                    nn.Dropout(dropout_rate),
                    nn.LeakyReLU()
                )
            )
            in_channels = d_dim
        SSK_img_modules_p.append(
            nn.Sequential(
                nn.Linear(in_channels, 2)
            ))
        self.generation_p = nn.Sequential(*SSK_img_modules_p)
        self.IS_test = IS_test

    def forward(self, x):
        seg_t1 = self.SSK_img_img_res(torch.unsqueeze(x, 1))
        if self.IS_BRATS:
            seg_t1 = self.pool3d(seg_t1)
        seg_t1 = seg_t1.view(-1, 2048, 2048)
        seg_t1 = self.SSK_img_net_linear(seg_t1)

        out_seg_t1 = torch.squeeze(self.pool(seg_t1))
        mu_seg_t1 = self.fcp_mu(out_seg_t1)
        logvar_seg_t1 = self.fcp_var(out_seg_t1)
        z_seg_t1 = self.reparameterize(mu_seg_t1, logvar_seg_t1)
        y_pred_seg_t1 = self.generation_p(z_seg_t1)
        if self.IS_test:
            return [y_pred_seg_t1, z_seg_t1]
        paras = [y_pred_seg_t1, z_seg_t1, mu_seg_t1, logvar_seg_t1]
        return paras

    def loss_MI_img(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001
        tau = 0.5
        y_pred_t1 = paras[0]
        z_t1 = paras[1]
        mu_t1 = paras[2]
        logvar_t1 = paras[3]
        recons_img_loss = self.BCEloss(y_pred_t1, y)
        std_img = torch.exp(0.5 * logvar_t1)

        recons_img_loss = self.BCEloss(y_pred_t1, y)
        kld_loss = self.kl_loss_single(mu_t1, logvar_t1)

        loss = recons_img_loss * recon_weight + kld_loss * kld_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_img_loss, 'KLD': kld_loss}

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            eps_mu = torch.randn_like(mu)
            z += eps * std + mu
        return z * 0.01

    def kl_loss_single(self, mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)


class Block_MultiSequencesResNet3D(nn.Module):
    def __init__(self, dropout_rate, IS_test, devices):
        super(Block_MultiSequencesResNet3D, self).__init__()
        self.t1_resNet3D = Block_ResNet3D()
        self.t2_resNet3D = Block_ResNet3D()
        self.t1c_resNet3D = Block_ResNet3D()
        self.tof_resNet3D = Block_ResNet3D()
        self.segResNet3D = Block_SegResNet3D()
        self.reportNet = Block_Report()
    # def forward(self,t1,seg_t1,text):


class VMD_Net(nn.Module):
    def __init__(self, latent_dim, bert_embed_size, n_feature_maps, feature_map_size, max_num_words, class_weights=None,
                 dropout_rate=0.5, batch_size=1, is_eval=False,
                 res_path="/home/bocao/plaque/vkd_segpre_multiclassification/trained/"
                          "2024_01_27_10_14_04-SSK_res50/vkd_MRI_epoch_148_time_2024_01_27--12_40.pth",
                 IS_BRATS=False,
                 devices=torch.device("cuda"),
                 IS_test=False,
                 **kwargs):
        super(VMD_Net, self).__init__()
        self.IS_test = IS_test
        self.is_eval = is_eval
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        if IS_BRATS:
            self.pool3d = nn.AdaptiveAvgPool3d((16, 16, 8))
        self.BCEloss = nn.BCEWithLogitsLoss()

        self.resNet3D = Block_ResNet3D(IS_BRATS, latent_dim, n_feature_maps,
                                       feature_map_size, devices,
                                       dropout_rate=dropout_rate,
                                       batch_size=batch_size,
                                       is_eval=is_eval, IS_test=IS_test)
        self.segResNet3D = Block_SegResNet3D(IS_BRATS, latent_dim, n_feature_maps,
                                             feature_map_size, devices,
                                             dropout_rate=dropout_rate,
                                             batch_size=batch_size,
                                             is_eval=is_eval, IS_test=IS_test)
        self.reportNet = Block_Report(IS_BRATS, latent_dim, bert_embed_size, max_num_words, dropout_rate, batch_size,
                                      is_eval, IS_test)

    def forward(self, img_input, text_input, seg_t1):

        resNet_paras = self.resNet3D(img_input)
        if self.IS_test:
            return resNet_paras
        segResNet_paras = self.segResNet3D(seg_t1)
        report_paras = self.reportNet(text_input)

        return resNet_paras + report_paras + segResNet_paras

    def loss_MI(self, f1, f2, targets):
        import torch
        import torch.nn.functional as F
        from torch import nn

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

    def kl_loss_multi(self, mu_p, logvar_p, mu_r, logvar_r):

        p = torch.distributions.Normal(mu_p, logvar_p)
        r = torch.distributions.Normal(mu_r, logvar_r)

        return torch.distributions.kl_divergence(p, r).mean()

    def loss_MI_VMD(self, paras, y):
        # 输入：y_pred, y_batch, mu, logvar, y_predr, mur, logvarr,kl_ann_factor[2]
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001
        tau = 0.5
        y_pred_t1 = paras[0]
        z_t1 = paras[1]
        mu_t1 = paras[2]
        logvar_t1 = paras[3]
        y_predr = paras[4]
        z_r = paras[5]
        mu_r = paras[6]
        logvar_r = paras[7]
        y_pred_seg_t1 = paras[8]
        z_seg_t1 = paras[9]
        mu_seg_t1 = paras[10]
        logvar_seg_t1 = paras[11]
        std_img = torch.exp(0.5 * logvar_t1)
        std_text = torch.exp(0.5 * logvar_r)

        recons_img_loss = self.BCEloss(y_pred_t1, y)
        recons_text_loss = self.BCEloss(y_predr, y)
        # kld_loss = self.kl_loss_single(mu, logvar)
        # kld_loss = self.kl_loss_multi(mu, logvar, mu_r, logvar_r)
        kld_loss = self.kl_loss_multi(mu_t1, std_img, mu_r, std_text)

        # simi_segt1_t1 = nn.functional.cosine_similarity(y_pred_seg_t1, y_pred_t1, dim=-1)
        # simi_segt1_text = nn.functional.cosine_similarity(y_pred_seg_t1, y_predr, dim=-1)

        mi_simi_MS = self.loss_MI(y_pred_seg_t1, y_pred_t1, y)
        mi_simi_MA = self.loss_MI(y_pred_seg_t1, y_predr, y)
        ms_weight = 0.05
        ma_weight = 0.05

        loss = recons_img_loss * recon_weight + kld_loss * kld_weight + recons_text_loss * recon_weight_r + mi_simi_MS * ms_weight + mi_simi_MA * ma_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_img_loss, 'KLD': kld_loss, 'Recons_r': recons_text_loss,
                'mi_simi_MS': mi_simi_MS, 'mi_simi_MA': mi_simi_MA}


class VMD(nn.Module):
    def __init__(self,
                 latent_dim,
                 bert_embed_size,
                 n_feature_maps,
                 feature_map_size,
                 max_num_words,
                 class_weights=None,
                 dropout_rate=0.5,
                 batch_size=1,
                 is_eval=False,
                 res_path="/home/bocao/plaque/vkd_segpre_multiclassification/trained/"
                          "2024_01_27_10_14_04-SSK_res50/vkd_MRI_epoch_148_time_2024_01_27--12_40.pth",
                 IS_Mutual_Information=False,
                 is_vae_resNet=False,
                 is_vae_seg_resNet=False,
                 is_vae_text=False,
                 IS_BRATS=False,
                 devices=torch.device("cuda"),
                 **kwargs):
        super(VMD, self).__init__()

        self.devices = devices
        self.leakyRelu = nn.LeakyReLU()
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.token_dim = bert_embed_size  # 768
        self.n_feature_maps = n_feature_maps
        self.feature_map_size = feature_map_size
        self.max_num_words = max_num_words
        if class_weights != None:
            self.BCElossde = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights))
        else:
            self.BCElossde = nn.BCEWithLogitsLoss()
        self.pretrained_path = res_path
        '''
        pure img:
        pure text:
        pure seged img:
        mutual information:
        multi sequences:
        mutual information+multi sequences:
        '''
        self.is_eval = is_eval
        self.IS_Mutual_Information = IS_Mutual_Information
        self.IS_BRATS = IS_BRATS
        self.is_vae_resNet = is_vae_resNet
        self.is_vae_seg_resNet = is_vae_seg_resNet
        self.is_vae_text = is_vae_text
        '''
                        self.img_res
        '''
        if self.is_vae_text:
            self.pe_t = PositionalEncoding(self.token_dim, max_len=self.max_num_words)  # 768 * 77
            self.transformer_t = TransformerEncoderLayer(d_model=self.token_dim)
            self.fc_mu = nn.Sequential(
                nn.Linear(self.token_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.token_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            modules = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = self.latent_dim
            for d_dim in dense_dims:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))

            self.generation = nn.Sequential(*modules)
        elif self.is_vae_resNet:
            self.img_res = get_res(
                pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
                no_seg=True, train=True, map_devices=self.devices)

            self.net_linear = nn.Sequential(
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fcp_mu = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)

            )
            self.fcp_var = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            modules_p = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = int(self.latent_dim)
            for d_dim in dense_dims:
                modules_p.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))
            self.generation_p = nn.Sequential(*modules_p)
        elif self.is_vae_seg_resNet:
            self.SSK_img_img_res = get_res(
                pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
                no_seg=True, train=True, map_devices=devices)

            self.SSK_img_net_linear = nn.Sequential(
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fcp_mu = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)

            )
            self.fcp_var = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))

            SSK_img_modules_p = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = int(self.latent_dim)
            for d_dim in dense_dims:
                SSK_img_modules_p.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            SSK_img_modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))
            self.generation_p = nn.Sequential(*SSK_img_modules_p)
        elif self.IS_Mutual_Information:
            self.SSK_img_img_res = get_res(
                pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
                no_seg=True, map_devices=devices)

            self.SSK_img_net_linear = nn.Sequential(
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fcp_mu = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)

            )
            self.fcp_var = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))

            SSK_img_modules_p = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = int(self.latent_dim)
            for d_dim in dense_dims:
                SSK_img_modules_p.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            SSK_img_modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))
            self.generation_p = nn.Sequential(*SSK_img_modules_p)
            self.img_res = get_res(
                pretrain_path="/home/bocao/plaque/vkd_segpre_multiclassification/Model/resnet_50_23dataset.pth",
                no_seg=True, map_devices=devices)

            self.net_linear = nn.Sequential(
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_map_size, self.feature_map_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fcp_mu = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)

            )
            self.fcp_var = nn.Sequential(
                nn.Linear(self.n_feature_maps, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            modules_p = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = int(self.latent_dim)
            for d_dim in dense_dims:
                modules_p.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            modules_p.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))
            self.generation_p = nn.Sequential(*modules_p)
            self.pe_t = PositionalEncoding(self.token_dim, max_len=self.max_num_words)  # 768 * 77
            self.transformer_t = TransformerEncoderLayer(d_model=self.token_dim)
            self.fc_mu = nn.Sequential(
                nn.Linear(self.token_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate)
            )
            self.fc_var = nn.Sequential(
                nn.Linear(self.token_dim, self.latent_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            modules = []
            dense_dims = [int(self.latent_dim / 2), int(self.latent_dim / 4)]
            in_channels = self.latent_dim
            for d_dim in dense_dims:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, d_dim),
                        nn.BatchNorm1d(d_dim),
                        nn.Dropout(dropout_rate),
                        nn.LeakyReLU()
                    )
                )
                in_channels = d_dim
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, 2)
                ))

            self.generation = nn.Sequential(*modules)
        else:
            raise Exception("wrong setting")

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1, 1))
        if IS_BRATS:
            self.pool3d = nn.AdaptiveAvgPool3d((16, 16, 8))

    def loss_MI_img(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001
        tau = 0.5
        y_pred_t1 = paras[0]
        z_t1 = paras[1]
        mu_t1 = paras[2]
        logvar_t1 = paras[3]
        recons_img_loss = self.BCEloss(y_pred_t1, y)
        std_img = torch.exp(0.5 * logvar_t1)

        recons_img_loss = self.BCEloss(y_pred_t1, y)
        kld_loss = self.kl_loss_single(mu_t1, logvar_t1)

        loss = recons_img_loss * recon_weight + kld_loss * kld_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_img_loss, 'KLD': kld_loss}

    def loss_MI_text(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1e-3
        w_weight = 0.001
        tau = 0.5
        y_predr = paras[0]
        z_t1 = paras[1]
        mu = paras[2]
        logvar_r = paras[3]

        recons_text_loss = self.BCEloss(y_predr, y)
        kld_loss = self.kl_loss_single(mu, logvar_r)

        loss = kld_loss * kld_weight + recons_text_loss * recon_weight_r
        return {'loss': loss, 'KLD': kld_loss, 'Recons_r': recons_text_loss}

    def loss_MI_VKD(self, paras, y):
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001
        tau = 0.5
        y_pred_t1 = paras[0]
        z_t1 = paras[1]
        mu_t1 = paras[2]
        logvar_t1 = paras[3]
        y_predr = paras[4]
        z_r = paras[5]
        mu_r = paras[6]
        logvar_r = paras[7]
        y_pred_seg_t1 = paras[8]
        z_seg_t1 = paras[9]
        mu_seg_t1 = paras[10]
        logvar_seg_t1 = paras[11]
        std_img = torch.exp(0.5 * logvar_t1)
        std_text = torch.exp(0.5 * logvar_r)

        recons_img_loss = self.BCEloss(y_pred_t1, y)
        recons_text_loss = self.BCEloss(y_predr, y)
        kld_loss = self.kl_loss_multi(mu_t1, std_img, mu_r, std_text)

        mi_simi_MS = self.loss_MI(y_pred_seg_t1, y_pred_t1, y)
        mi_simi_MA = self.loss_MI(y_pred_seg_t1, y_predr, y)
        ms_weight = 0.05
        ma_weight = 0.05

        loss = recons_img_loss * recon_weight + kld_loss * kld_weight + recons_text_loss * recon_weight_r + mi_simi_MS * ms_weight + mi_simi_MA * ma_weight
        return {'loss': loss, 'Reconstruction_Loss': recons_img_loss, 'KLD': kld_loss, 'Recons_r': recons_text_loss,
                'mi_simi_MS': mi_simi_MS, 'mi_simi_MA': mi_simi_MA}

    def loss_MI(self, f1, f2, targets):
        import torch
        import torch.nn.functional as F
        from torch import nn

        EPISILON = 1e-10
        temperature = 0.5
        softmax = nn.Softmax(dim=1)

        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        targets = torch.argmax(targets, 1)
        mask = targets.unsqueeze(1) - targets
        self_mask = (torch.zeros_like(mask) != mask)  ### where the negative samples are labeled as 1
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)
        cos = 1 - 0.5 * dist

        pred_softmax = softmax(cos / temperature)  ### convert to multi-class prediction scores

        log_pos_softmax = - torch.log(pred_softmax + EPISILON) * ~self_mask.int()
        log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.int()
        log_softmax = log_pos_softmax.sum(1) / ~self_mask.sum(1).int() + log_neg_softmax.sum(
            1) / self_mask.sum(1).int()
        loss = log_softmax

        return loss.mean()

    def forward(self, img_input, img_input_t2, img_input_t1c, img_input_tof, text_input, seg_t1, eval=False,
                test=False):

        '''
                img feature
                res50:features_map_size = 2048
                res10:features_map_size = 1024

                因为vae部分的也是对应的，所以同步修改了

                pad 1 : 2312
                pad 5 : 2916
                pad 10 : 3610
        '''
        # if eval:
        #     self.is_using_reports = False
        if self.is_vae_text:
            text_features = text_input
            out_r = torch.squeeze(self.pool(text_features))
            mu_r = self.fc_mu(out_r)
            logvar_r = self.fc_var(out_r)
            z_r = self.reparameterize(mu_r, logvar_r)
            y_predr = self.generation(z_r)
            if test:
                return y_predr, z_r
            paras = [y_predr, z_r, mu_r, logvar_r]
            return paras
        elif self.is_vae_resNet:
            t1 = torch.unsqueeze(img_input, 1)

            t1 = self.img_res(t1)
            if self.IS_BRATS:
                t1 = self.pool3d(t1)
            t1 = t1.view(-1, 2048, 2048)
            t1 = self.net_linear(t1)
            out_t1 = torch.squeeze(self.pool(t1))
            mu_t1 = self.fcp_mu(out_t1)
            logvar_t1 = self.fcp_var(out_t1)
            z_t1 = self.reparameterize(mu_t1, logvar_t1)
            y_pred_t1 = self.generation_p(z_t1)
            if test:
                return y_pred_t1, z_t1
            paras = [y_pred_t1, z_t1, mu_t1, logvar_t1]
            return paras
        elif self.is_vae_seg_resNet:
            seg_t1 = self.SSK_img_img_res(torch.unsqueeze(seg_t1, 1))
            if self.IS_BRATS:
                seg_t1 = self.pool3d(seg_t1)
            seg_t1 = seg_t1.view(-1, 2048, 2048)
            seg_t1 = self.SSK_img_net_linear(seg_t1)

            out_seg_t1 = torch.squeeze(self.pool(seg_t1))
            mu_seg_t1 = self.fcp_mu(out_seg_t1)
            logvar_seg_t1 = self.fcp_var(out_seg_t1)
            z_seg_t1 = self.reparameterize(mu_seg_t1, logvar_seg_t1)
            y_pred_seg_t1 = self.generation_p(z_seg_t1)
            if test:
                return y_pred_seg_t1, z_seg_t1
            paras = [y_pred_seg_t1, z_seg_t1, mu_seg_t1, logvar_seg_t1]
            return paras
        elif self.IS_Mutual_Information:
            t1 = torch.unsqueeze(img_input, 1)
            t1 = self.img_res(t1)
            if self.IS_BRATS:
                t1 = self.pool3d(t1)
            t1 = t1.view(-1, 2048, 2048)
            t1 = self.net_linear(t1)
            out_t1 = torch.squeeze(self.pool(t1))
            mu_t1 = self.fcp_mu(out_t1)
            logvar_t1 = self.fcp_var(out_t1)
            z_t1 = self.reparameterize(mu_t1, logvar_t1)
            y_pred_t1 = self.generation_p(z_t1)
            if test:
                return y_pred_t1, z_t1
            else:
                # text_features = self.transformer_t(self.pe_t(text))
                text_features = text_input
                out_r = torch.squeeze(self.pool(text_features))
                mu_r = self.fc_mu(out_r)
                logvar_r = self.fc_var(out_r)
                z_r = self.reparameterize(mu_r, logvar_r)
                y_predr = self.generation(z_r)

                seg_t1 = self.SSK_img_img_res(torch.unsqueeze(seg_t1, 1))
                if self.IS_BRATS:
                    seg_t1 = self.pool3d(seg_t1)
                seg_t1 = seg_t1.view(-1, 2048, 2048)
                seg_t1 = self.SSK_img_net_linear(seg_t1)

                out_seg_t1 = torch.squeeze(self.pool(seg_t1))

                mu_seg_t1 = self.fcp_mu(out_seg_t1)
                logvar_seg_t1 = self.fcp_var(out_seg_t1)
                z_seg_t1 = self.reparameterize(mu_seg_t1, logvar_seg_t1)
                y_pred_seg_t1 = self.generation_p(z_seg_t1)
                paras = [y_pred_t1, z_t1, mu_t1, logvar_t1, y_predr, z_r, mu_r, logvar_r, y_pred_seg_t1, z_seg_t1,
                         mu_seg_t1, logvar_seg_t1]
                return paras

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        z = torch.zeros_like(mu)
        std = torch.exp(0.5 * logvar)
        for i in range(100):
            eps = torch.randn_like(std)
            eps_mu = torch.randn_like(mu)
            z += eps * std + mu
        return z * 0.01

        # eps = torch.randn(mu.shape[0], mu.shape[1]).to(logvar)
        # return mu + eps * torch.exp(0.5 * logvar)


    def kl_loss_single(self, mu, log_var):
        # mu1 = mu
        # var1 = log_var
        #
        # mu2 = (0.5*mu).exp()
        # var2 = (0.5*log_var).exp()
        #
        # mu0 = 0
        # var0 = 1
        #
        # var_ratio = (var1 / 1).pow(2)
        # t1 = ((mu1 - 0) / 1).pow(2)
        # kld_logvar_in =  -0.5 * (-log_var.pow(2) - mu.pow(2) + 1 + log_var.log())
        # 结果说明
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    def kl_loss_multi(self, mu_p, logvar_p, mu_r, logvar_r):

        p = torch.distributions.Normal(mu_p, logvar_p)
        r = torch.distributions.Normal(mu_r, logvar_r)

        return torch.distributions.kl_divergence(p, r).mean()


    def loss_img(self, *args,
                 **kwargs):
        epsilon = 1e-8
        recons = args[0]  # y_pred
        label = args[1]  # y_batch
        recons_loss = self.BCEloss(recons, label)
        if not self.is_pure_resNet:
            mu = args[2] + epsilon
            logvar = args[3]
            annealing_factor = args[4]
            kld_loss = self.kl_loss_single(mu, logvar)
            loss = recons_loss + kld_loss * 0.01
            return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
        else:
            loss = recons_loss
            return {'loss': loss, 'Reconstruction_Loss': recons_loss}



    def loss_img_text(self,
                      *args,
                      **kwargs):

        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        # 输入：y_pred, y_batch, mu, logvar, y_predr, mur, logvarr,kl_ann_factor[2]
        epsilon = 1e-8
        recon_weight = 1
        recon_weight_r = 1
        kld_weight = 1
        w_weight = 0.001

        recons = args[0]  # y_pred
        label = args[1]  # y_batch
        mu = args[2] + epsilon
        logvar = args[3]
        recons_r = args[4] + epsilon  # y_predr
        mu_r = args[5] + epsilon
        logvar_r = args[6]
        annealing_factor = args[7]
        std_img = torch.exp(0.5 * logvar)
        std_text = torch.exp(0.5 * logvar_r)

        recons_loss = self.BCEloss(recons, label)
        recons_loss_r = self.BCEloss(recons_r, label)
        # kld_loss = self.kl_loss_single(mu, logvar)
        # kld_loss = self.kl_loss_multi(mu, logvar, mu_r, logvar_r)
        kld_loss = self.kl_loss_multi(mu, std_img, mu_r, std_text)

        loss = recons_loss * recon_weight + kld_loss * kld_weight + recons_loss_r * recon_weight_r
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss, 'Recons_r': recons_loss_r}
