import torch
from torch import nn
import numpy as np
from Probabilistic_Unet_Pytorch.utils import l2_regularisation
from pionono_models.model_headless import UnetHeadless
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LatentVariable(nn.Module):
    """
    This module defines the random latent variable z with distribution q(z|r) with r being the rater.
    """
    def __init__(self, num_annotators, latent_dims=2, prior_mu_value=0.0, prior_sigma_value=1.0, z_posterior_init_sigma=0.0):
        super(LatentVariable, self).__init__()
        self.latent_dims = latent_dims
        self.no_annotators = num_annotators
        prior_mu, prior_cov = self._init_distributions(prior_mu=prior_mu_value, prior_sigma=prior_sigma_value)
        self.prior_mu = torch.nn.Parameter(prior_mu)
        self.prior_covtril = torch.nn.Parameter(prior_cov)
        self.prior_mu.requires_grad = False
        self.prior_covtril.requires_grad = False
        post_mu_value = np.random.standard_normal(size=[num_annotators, latent_dims])*z_posterior_init_sigma + prior_mu_value
        post_sigma_value = prior_sigma_value
        posterior_mu, posterior_cov = self._init_distributions(prior_mu=post_mu_value, prior_sigma=post_sigma_value)
        self.posterior_mu = torch.nn.Parameter(posterior_mu)
        self.posterior_covtril = torch.nn.Parameter(posterior_cov)
        self.name = 'LatentVariable'

    def _init_distributions(self, prior_mu=0.0, prior_sigma=1.0):
        mu_list = []
        cov_list = []
        prior_mu = np.array(prior_mu)
        prior_sigma = np.array(prior_sigma)
        for a in range(self.no_annotators):
            if prior_mu.size > 1:
                mu = prior_mu[a]
            else:
                mu = prior_mu
            if prior_sigma.size > 1:
                sigma = prior_sigma[a]
            else:
                sigma = prior_sigma
            mu_a = np.ones(self.latent_dims)*mu
            # we use sigma values (instead of sigma+sigma) because we pass this matrix as tril matrix L
            # this makes the sigma value of the cov matrix squared
            cov_a = np.eye(self.latent_dims) * (sigma)
            mu_list.append(mu_a)
            cov_list.append(cov_a)
        mu_list = torch.tensor(np.array(mu_list))
        covtril_list = torch.tensor(np.array(cov_list))
        return mu_list, covtril_list

    def forward(self, annotator, sample=True):
        z = torch.zeros([len(annotator), self.latent_dims]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                scale_tril=torch.tril(self.posterior_covtril[a]))

            if sample:
                z_i = dist_a.rsample()
            else:
                z_i = dist_a.loc
            z[i] = z_i
        return z

    def get_kl_loss(self, annotator):
        kl_loss = torch.zeros([len(annotator)]).to(device)
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a_posterior = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                          scale_tril=torch.tril(self.posterior_covtril[a]))
            dist_a_prior = torch.distributions.multivariate_normal.MultivariateNormal(self.prior_mu[a],
                                                                                      scale_tril=torch.tril(self.prior_covtril[a]))

            kl_loss[i] = torch.distributions.kl_divergence(dist_a_posterior, dist_a_prior)
        kl_mean = torch.mean(kl_loss)
        return kl_mean


class PiononoHead(nn.Module):
    """
    The Segmentation head combines the sample taken from the latent space,
    and feature map by concatenating them along their channel axis.
    """
    def __init__(self, num_filters_last_layer, latent_dim, num_output_channels, num_classes, no_convs_fcomb,
                 head_kernelsize, head_dilation, use_tile=True):
        super(PiononoHead, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters_last_layer = num_filters_last_layer
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.head_kernelsize = head_kernelsize
        self.name = 'PiononoHead'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters_last_layer+self.latent_dim, self.num_filters_last_layer,
                                    kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters_last_layer, self.num_filters_last_layer,
                                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters_last_layer, self.num_classes, kernel_size=self.head_kernelsize,
                                        dilation=head_dilation, padding='same')
            self.activation = torch.nn.Softmax(dim=1)

            self.layers.apply(self.initialize_weights)
            self.last_layer.apply(self.initialize_weights)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=True):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            x = self.layers(feature_map)
            y = self.last_layer(x)
            if use_softmax: 
                y = self.activation(y)
            return y

class PiononoModel(nn.Module):
    """
    The implementation of the Pionono Model. It consists of a segmentation backbone, probabilistic latent variable and
    segmentation head.
    """

    def __init__(self, input_channels=3, num_classes=1, annotators=6, gold_annotators=[0], latent_dim=8,
                 z_prior_mu=0.0, z_prior_sigma=2.0, z_posterior_init_sigma=8.0, no_head_layers=3, head_kernelsize=1,
                 head_dilation=1, kl_factor=1.0, reg_factor=0.1, mc_samples=5):
        super(PiononoModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.annotators = annotators
        self.gold_annotators = gold_annotators
        self.no_head_layers = no_head_layers
        self.head_kernelsize = head_kernelsize
        self.head_dilation = head_dilation
        self.kl_factor = kl_factor
        self.reg_factor = reg_factor
        self.train_mc_samples = mc_samples
        self.test_mc_samples = 20
        self.unet = UnetHeadless(input_channels).to(device)
        self.z = LatentVariable(len(annotators), latent_dim, prior_mu_value=z_prior_mu, prior_sigma_value=z_prior_sigma,
                                z_posterior_init_sigma=z_posterior_init_sigma).to(device)
        self.head = PiononoHead(16, self.latent_dim, self.input_channels, self.num_classes,
                                self.no_head_layers, self.head_kernelsize, self.head_dilation, use_tile=True).to(device)
        self.phase = 'segmentation'
        self.name = 'PiononoModel'

    def forward(self, patch):
        """
        Get feature maps.
        """
        self.unet_features = self.unet.forward(patch)

    def map_annotators_to_correct_id(self, annotator_ids: torch.tensor, annotator_list:list = None):
        new_ids = torch.zeros_like(annotator_ids).to(device)
        for a in range(len(annotator_ids)):
            id_corresponds = (annotator_list[int(annotator_ids[a])] == np.array(self.annotators))
            if not np.any(id_corresponds):
                raise Exception('Annotator has no corresponding distribution. Annotator: ' + str(annotator_list[int(annotator_ids[a])]))
            new_ids[a] = torch.nonzero(torch.tensor(annotator_list[int(annotator_ids[a])] == np.array(self.annotators)))[0][0]
        return new_ids

    def sample(self, use_z_mean: bool, annotator_ids: torch.tensor, annotator_list: list = None, use_softmax=True):
        """
        Get sample of output distribution. Annotator list defines the distributions (q|r) that are used.
        """
        if annotator_list is not None:
            annotator_ids = self.map_annotators_to_correct_id(annotator_ids, annotator_list)

        if use_z_mean == False:
            z = self.z.forward(annotator_ids, sample=True)
        else:
            z = self.z.forward(annotator_ids, sample=False)
        pred = self.head.forward(self.unet_features, z, use_softmax)

        return pred

    def get_gold_predictions(self):
        """
        Get gold predictions (based on the gold distribution).
        """
        if len(self.gold_annotators) == 1:
            annotator = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[0]
            mean, std = self.mc_sampling(annotator, use_softmax=True)
        else:
            shape = [self.train_mc_samples * len(self.gold_annotators), self.unet_features.shape[0], self.num_classes,
                     self.unet_features.shape[-2], self.unet_features.shape[-1]]
            samples = torch.zeros(shape).to(device)
            for a in range(len(self.gold_annotators)):
                for i in range(self.train_mc_samples):
                    annotator_ids = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[a]
                    samples[(a * self.train_mc_samples) + i] = self.sample(use_z_mean=False,
                                                                           annotator_ids=annotator_ids,
                                                                           use_softmax=True)
            mean = torch.mean(samples, dim=0)
            std = torch.std(samples, dim=0)
        return mean, std

    def mc_sampling(self, annotator: torch.tensor = None, use_softmax=True):
        """
        Monte-Carlo sampling to get mean and std of output distribution.
        """
        if self.training:
            mc_samples = self.train_mc_samples
        else:
            mc_samples = self.test_mc_samples
        shape = [mc_samples, annotator.shape[0], self.num_classes, self.unet_features.shape[-2], self.unet_features.shape[-1]]
        samples = torch.zeros(shape).to(device)
        for i in range(mc_samples):
            samples[i] = self.sample(use_z_mean=False, annotator_ids=annotator, use_softmax=use_softmax)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        return mean, std

    def elbo(self, labels: torch.tensor, loss_fct, annotator: torch.tensor):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        # self.preds = self.sample(use_z_mean=False, annotator=annotator)
        self.preds, _ = self.mc_sampling(annotator=annotator, use_softmax=False)
        self.log_likelihood_loss = loss_fct(self.preds, labels)
        self.kl_loss = self.z.get_kl_loss(annotator) * self.kl_factor

        return -(self.log_likelihood_loss + self.kl_loss)

    def combined_loss(self, labels, loss_fct, annotator):
        """
        Combine ELBO with regularization of deep network weights.
        """
        elbo = self.elbo(labels, loss_fct=loss_fct, annotator=annotator)
        self.reg_loss = l2_regularisation(self.head.layers) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def train_step(self, images, labels, loss_fct, ann_ids):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        loss = self.combined_loss(labels, loss_fct, ann_ids)
        y_pred = self.preds

        return loss, y_pred
    
    def val_step(self, images):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        y_pred = []
        meta = torch.ones([images.shape[0]])
        for i in range(4):
            y_pred_case, _ = self.mc_sampling(annotator=meta*i, use_softmax=False)
            y_pred.append(y_pred_case)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred
