import torch
from pionono_models.segmentation_backbone import create_segmentation_backbone


class UnetHeadless(torch.nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        seg_model = create_segmentation_backbone(input_channels)
        self.seg_encoder = seg_model.encoder
        self.seg_decoder = seg_model.decoder

    def forward(self, x):
        x = self.seg_encoder(x)
        x = self.seg_decoder(*x)
        # x = self.seg_model(x)
        return x