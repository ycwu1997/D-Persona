import torch
from pionono_models.segmentation_backbone import create_segmentation_backbone

class SupervisedSegmentationModel(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.seg_model = create_segmentation_backbone(num_channels)
        self.activation = torch.nn.Softmax(dim=1)

    def forward(self, x, use_softmax=True):
        x = self.seg_model(x)
        if use_softmax:
            y = self.activation(x)
        else:
            y = x
        return y

    def train_step(self, images, labels, loss_fct, ann_ids):
        y_pred = self.forward(images, use_softmax=False)
        loss = loss_fct(y_pred, labels)
        return loss, y_pred
