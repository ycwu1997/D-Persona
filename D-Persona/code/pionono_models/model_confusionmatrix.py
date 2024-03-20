import torch
from pionono_models.segmentation_backbone import create_segmentation_backbone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'bn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.BatchNorm2d(out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'ln':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels, out_channels, affine=True),
            torch.nn.PReLU()
        )
    elif norm == 'gn':
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            torch.nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            torch.nn.PReLU()
        )


class gcm_layers(torch.nn.Module):
    """ This defines the global confusion matrix layer. It defines a (class_no x class_no) confusion matrix, we then use unsqueeze function to match the
    size with the original pixel-wise confusion matrix layer, this is due to convenience to be compact with the existing loss function and pipeline.
    """

    def __init__(self, class_no, input_height, input_width):
        super(gcm_layers, self).__init__()
        self.class_no = class_no
        self.input_height = input_height
        self.input_width = input_width
        self.global_weights = torch.nn.Parameter(torch.eye(class_no))
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        all_weights = self.global_weights.unsqueeze(0).repeat(x.size(0), 1, 1)
        all_weights = all_weights.unsqueeze(3).unsqueeze(4).repeat(1, 1, 1, self.input_height, self.input_width)
        y = self.softplus(all_weights)
        # y = all_weights

        return y


class cm_layers(torch.nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """

    def __init__(self, in_channels, norm, class_no):
        super(cm_layers, self).__init__()
        self.conv_1 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_2 = double_conv(in_channels=in_channels, out_channels=in_channels, norm=norm, step=1)
        self.conv_last = torch.nn.Conv2d(in_channels, class_no ** 2, 1, bias=True)
        self.softplus = torch.nn.Softplus()

    def forward(self, x):
        y = self.softplus(self.conv_last(self.conv_2(self.conv_1(x)))) # matrix has to be normalized per row!

        return y


class ConfusionMatrixModel(torch.nn.Module):
    def __init__(self, input_channels, num_classes, num_annotators, level, image_res, learning_rate, alpha, min_trace):
        super().__init__()
        self.seg_model = create_segmentation_backbone(input_channels).to(device)
        self.num_annotators = num_annotators
        self.num_classes = num_classes
        self.level = level
        self.image_res = image_res
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.min_trace = min_trace
        print("Number of annotators (model): ", self.num_annotators)
        self.cm_head = torch.nn.ModuleList()
        if level == 'global':
            print("Global crowdsourcing")
            for i in range(self.num_annotators):
                self.cm_head.append(
                    gcm_layers(num_classes, image_res, image_res))  # TODO: arrange inputwidht and height
        else:
            for i in range(self.num_annotators):
                self.cm_head.append(cm_layers(in_channels=16, norm='in',
                                              class_no=num_classes))  # TODO: arrange in_channels
        self.activation = torch.nn.Softmax(dim=1)


    def forward(self, x, use_softmax=True):
        y = self.seg_model(x)
        if use_softmax:
            y = self.activation(y)
        return y

    def forward_with_cms(self, x, use_softmax=True):
        x = self.seg_model.encoder(x)
        x = self.seg_model.decoder(*x)
        cms = []
        for i in range(self.num_annotators):
            cm = self.cm_head[i](x)  # BxCxCxWxH
            #cm_ = cm[0, :, :, 0, 0]
            #print("CM! ", cm_ / cm_.sum(0, keepdim=True))
            cms.append(cm)
        y = self.seg_model.segmentation_head(x)
        if use_softmax:
            y = self.activation(y)
        return y, cms

    def get_used_cms(self, cms, ann_ids):
        cm_shape = cms[0].size()
        used_cms = torch.zeros_like(cms[0])
        for i in range(len(ann_ids)):
            used_cms[i] = cms[int(ann_ids[i].long().to('cpu'))][i]
        return used_cms

    def get_noisy_pred(self, pred_gold, cm):
        b, c, h, w = pred_gold.size()

        # normalise the segmentation output tensor along dimension 1
        pred_norm = pred_gold

        # b x c x h x w ---> b*h*w x c x 1
        pred_norm = pred_norm.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)

        # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
        # label_noisy: noisy label, b x h x w

        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        cm = cm / cm.sum(1, keepdim=True)  # normalization
        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        # print(cm.shape, pred_norm.shape)
        pred_noisy = torch.bmm(cm, pred_norm).view(b * h * w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        return pred_noisy, cm

    def train_step(self, images, labels, loss_fct, ann_ids):
        # for training don't use softmax because it is integrated in loss function
        y_pred, cms = self.forward_with_cms(images, use_softmax=False)
        cms_used = self.get_used_cms(cms, ann_ids)
        pred_noisy, cms_used = self.get_noisy_pred(y_pred, cms_used)
        # log_likelihood_loss = loss_fct(pred_noisy, labels.view(b, h, w).long())
        log_likelihood_loss = loss_fct(pred_noisy, labels)

        b, c, h, w = y_pred.size()
        regularisation = torch.trace(torch.transpose(torch.sum(cms_used, dim=0), 0, 1)).sum() / (b * h * w)
        regularisation = self.alpha * regularisation
        # print(cms_used)
        if self.min_trace:
            loss = log_likelihood_loss + regularisation
        else:
            loss = log_likelihood_loss - regularisation
        return loss, y_pred

    def activate_min_trace(self):
        print("Minimize trace activated!")
        self.min_trace = True
        print("Alpha updated", self.alpha)
        optimizer = torch.optim.Adam([
            {'params': self.seg_model.parameters()},
            {'params': self.cm_head.parameters(), 'lr': 1e-4}
        ], lr=self.learning_rate)
        return optimizer

    def val_step(self, images):
        # for training don't use softmax because it is integrated in loss function
        pred, cms = self.forward_with_cms(images, use_softmax=False)
        
        y_pred = []
        meta = torch.ones([images.shape[0]])
        for i in range(4):
            cms_used = self.get_used_cms(cms, meta*i)
            y_pred_case, _ = self.get_noisy_pred(pred, cms_used)
            y_pred.append(y_pred_case)
        y_pred = torch.cat(y_pred, dim=1)
        return y_pred