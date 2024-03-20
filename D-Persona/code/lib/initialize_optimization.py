import torch
from torch import nn as nn

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        num_classes = input.shape[1]
        # target = torch.movedim(F.one_hot(target, num_classes), -1, 1)
        # compute per channel Dice coefficient
        # target = torch.movedim(target, -1, 1)
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect / denominator)

def init_optimization(model, args):

    learning_rate = 0.0001

    if args.model_name == 'pionono':
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.head.parameters()},
            {'params': model.z.parameters(), 'lr': 0.02}
        ]
    elif 'cm' in args.model_name:
        opt_params = [
            {'params': model.seg_model.parameters()},
            {'params': model.cm_head.parameters(), 'lr': 0.01}
        ]
    elif args.model_name == 'DPersona' and args.stage == 1:
        opt_params = [
            {'params': model.unet.parameters()},
            {'params': model.prior.parameters()},
            {'params': model.posterior.parameters()},
            {'params': model.fcomb.parameters()}
        ]
    elif args.model_name == 'DPersona' and args.stage == 2:
        opt_params = [
            {'params': model.proj_heads.parameters()}
        ]
    elif args.model_name == 'prob_unet':
        opt_params = [
            {'params': model.parameters()}
        ]

    optimizer = torch.optim.Adam(opt_params, lr=learning_rate)

    loss_fct = GeneralizedDiceLoss(normalization='sigmoid')
    return optimizer, loss_fct