#This code is based on: https://github.com/SimonKohl/probabilistic_unet
import torch
from torch import nn
from Probabilistic_Unet_Pytorch.unet_blocks import *
from Probabilistic_Unet_Pytorch.unet import Unet
from Probabilistic_Unet_Pytorch.utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from pionono_models.model_headless import UnetHeadless

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv_block(nn.Module):
    def __init__(self, input_channels, num_filters, padding=True):
        super(Conv_block, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = []
        for i in range(len(self.num_filters)):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(0.1))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class Projection(nn.Module):
    def __init__(self, num_experts, latent_dim):
        super(Projection, self).__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.multi_expert_heads =nn.ModuleList([Conv_block(16, [32, 16, 6], 3)
                                        for _ in range(self.num_experts)])
        self.pooling_layer = nn.AdaptiveAvgPool2d([1,1])
        self.activation = torch.nn.Softmax(dim=2)

    def forward(self, feature_map, z_set, idx):
        feats =  self.multi_expert_heads[idx](feature_map)
        bs = feature_map.shape[0]
        global_z = self.pooling_layer(feats).view(bs, self.latent_dim, -1).permute(0,2,1)
        z_expert =  global_z
        similarity = torch.bmm(z_expert, z_set.permute(0,2,1))
        similarity = self.activation(similarity)
        output = torch.bmm(similarity, z_set)
        return output

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 4

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            # segm = torch.unsqueeze(segm, 1)
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(16+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)
        self.activation = torch.nn.Softmax(dim=1)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=True):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            output = self.last_layer(output)
            if use_softmax:
                output = self.activation(output)
            return output


class DPersona(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, num_filters=[16,32,64,128,256],
                 latent_dim=6, no_convs_fcomb=4, num_experts=4, reg_factor=1.0, original_backbone=False):
        super(DPersona, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.num_experts = num_experts
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.reg_factor = reg_factor
        self.original_backbone = original_backbone
        if self.original_backbone:
            self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        else:
            self.unet = UnetHeadless(input_channels).to(device)

        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)
    
        self.proj_heads = Projection(self.num_experts, self.latent_dim).to(device)

    def forward(self, patch, segm=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        if self.original_backbone:
            self.unet_features = self.unet.forward(patch, False)
        else:
            self.unet_features = self.unet.forward(patch)

    def minmax(self, post_masks):
        gt_max_union = torch.max(post_masks, dim=1, keepdim=True).values
        gt_min_union = torch.min(post_masks, dim=1, keepdim=True).values
        masks = torch.cat([gt_max_union, gt_min_union],dim=1)
        return masks

    def get_z_set(self, dist, sample_num=400):
        z_set = []
        for _ in range(sample_num):
            z_case = dist.sample()
            z_set.append(z_case.unsqueeze(1))
        z_set = torch.cat(z_set, dim=1)
        return z_set

    def z_mapping(self, z_set):
        num_z = z_set.size(1)
        cases = [] #
        for idx in range(num_z):
            z_case = z_set[:,idx]
            case = self.fcomb.forward(self.unet_features, z_case, use_softmax=False)
            cases.append(case)
        samples = torch.cat(cases, dim=1)
        sample = torch.mean(samples, dim=1).unsqueeze(1)
        return sample

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def prior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            if training:
                z_prior = self.prior_latent_space.rsample()
            else:
                z_prior = self.prior_latent_space.sample()
            sample = self.fcomb.forward(self.unet_features, z_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def posterior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            if training:
                z_posterior = self.posterior_latent_space.rsample()
            else:
                z_posterior = self.posterior_latent_space.sample()
            sample = self.fcomb.forward(self.unet_features, z_posterior, use_softmax=False)
            samples.append(sample)
        return samples

    def elbo(self, args, segm, criterion, analytic_kl=True):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
  
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False))

        #Here we use the posterior sample sampled above
        self.re_post = self.posterior_sampling(sample_num=1, training=True)[0]

        #Here we use the prior sample sampled above
        self.re_prior = torch.cat(self.prior_sampling(sample_num=args.prior_sample_num, training=True), dim=1)

        prior_bound = self.minmax(segm)
        prior_pred_max = torch.max(self.re_prior, dim=1, keepdim=True).values
        prior_pred_min = torch.min(self.re_prior, dim=1, keepdim=True).values
        pred_bound = torch.cat([prior_pred_max, prior_pred_min],dim=1)
        bound_loss = criterion(pred_bound, prior_bound)

        random_idx = np.random.randint(0, args.mask_num)
        random_label = segm[:,random_idx].unsqueeze(1)

        reconstruction_loss = criterion(self.re_post, random_label)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.bound_loss = torch.sum(bound_loss)
        return -(self.reconstruction_loss + self.kl + args.beta * self.bound_loss)

    def combined_loss(self, args, labels, loss_fct):
        elbo = self.elbo(args, labels, criterion=loss_fct)
        self.reg_loss = (l2_regularisation(self.posterior) + l2_regularisation(self.prior) + l2_regularisation(self.fcomb.layers)) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def train_step(self, args, images, masks, loss_fct, stage = 1):
        if stage == 1:
            self.forward(images, masks, training=True)
            loss = self.combined_loss(args, masks, loss_fct)
            y_preds = self.re_post
        elif stage == 2:
            y_preds = self.test_step(images)
            loss = loss_fct(y_preds, masks)
        return loss, y_preds

    def val_step(self, images, sample_num = 50):
        self.forward(images, None, training=False)
        y_pred = torch.cat(self.prior_sampling(sample_num=sample_num, training=False), dim=1)
        return y_pred

    def test_step(self, images, sample_num = 100):
        self.forward(images, None, training=False)
        z_set = self.get_z_set(self.prior_latent_space, sample_num=sample_num)
        zs_expert, samples_expert = [], []
        for idx in range(self.num_experts):
            z_expert = self.proj_heads(self.unet_features, z_set, idx)
            sample_expert = self.z_mapping(z_expert)
            zs_expert.append(z_expert)
            samples_expert.append(sample_expert)
        y_preds = torch.cat(samples_expert, dim=1)
        return y_preds
