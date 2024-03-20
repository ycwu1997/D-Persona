import torch
from segmentation_models_pytorch import Unet

def create_segmentation_backbone(input_channels):

    seg_model = Unet(
        encoder_name='resnet34',  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights='imagenet',
        # use `imagenet` pre-trained weights for encoder initialization
        in_channels=input_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1  # model output channels (number of classes in your dataset)
    )
    
    return seg_model

