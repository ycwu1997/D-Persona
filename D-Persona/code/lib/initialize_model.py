import torch
from pionono_models.model_supervised import SupervisedSegmentationModel
from pionono_models.model_confusionmatrix import ConfusionMatrixModel
from lib.DPersona import DPersona
from Probabilistic_Unet_Pytorch.probabilistic_unet import ProbabilisticUnet
from pionono_models.model_pionono import PiononoModel

def init_model(args, opt):

    if args.model_name == 'prob_unet':
        model = ProbabilisticUnet(input_channels=opt.INPUT_CHANNEL, num_classes=opt.OUTPUT_CHANNEL,
                                       latent_dim=6,
                                       no_convs_fcomb=4, 
                                       alpha=1.0,
                                       reg_factor=0.00001,
                                       original_backbone=False)
    elif 'DPersona' in args.model_name:
        model = DPersona(input_channels=opt.INPUT_CHANNEL, num_classes=opt.OUTPUT_CHANNEL,
                                       latent_dim=6,
                                       no_convs_fcomb=4,
                                       num_experts=args.mask_num,
                                       reg_factor=0.00001,
                                       original_backbone=False)
    elif args.model_name == 'pionono':
        model = PiononoModel(input_channels=opt.INPUT_CHANNEL, num_classes=opt.OUTPUT_CHANNEL,
                             annotators=[0,1,2,3],
                             gold_annotators=0,
                             latent_dim=8,
                             no_head_layers=3,
                             head_kernelsize=1,
                             head_dilation=1,
                             kl_factor=0.0005,
                             reg_factor=0.00001,
                             mc_samples=5,
                             z_prior_sigma=2.0,
                             z_posterior_init_sigma=8.0,
                             )
    elif args.model_name == 'cm_global':
        model = ConfusionMatrixModel(input_channels=opt.INPUT_CHANNEL, num_classes=opt.OUTPUT_CHANNEL, num_annotators=args.mask_num,
                                     level='global',
                                     image_res=opt.INPUT_SIZE,
                                     learning_rate=0.001,
                                     alpha=1.0,
                                     min_trace=False)
    elif args.model_name == 'cm_pixel':
        model = ConfusionMatrixModel(input_channels=opt.INPUT_CHANNEL, num_classes=opt.OUTPUT_CHANNEL, num_annotators=args.mask_num,
                                     level='pixel',
                                     image_res=opt.INPUT_SIZE,
                                     learning_rate=0.001,
                                     alpha=1.0,
                                     min_trace=False)
    else:
        model = SupervisedSegmentationModel(opt.INPUT_CHANNEL)

    return model
