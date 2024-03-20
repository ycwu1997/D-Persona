import torch
from tensorboardX import SummaryWriter

import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import shutil

from configs.config import *
from dataloader.data_loader import DatasetSpliter
from evaluate import validate
from utils.logger import Logger
from utils.utils import rand_seed
from dataloader.utils import data_preprocess
from lib.initialize_model import init_model
from lib.initialize_optimization import init_optimization

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/params_lidc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, help="save path", default='')
    parser.add_argument("--model_name", type=str, default='pionono')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--RESUME_FROM", type=int, default=0)

    args = parser.parse_args()
    opt = Config(config_path=args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    rand_seed(opt.RANDOM_SEED)

    # log & model folder
    if args.save_path == '':
        opt.MODEL_DIR += args.model_name + '_{}_{}'.format(opt.DATASET, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        opt.MODEL_DIR = args.save_path

    if not os.path.exists(opt.MODEL_DIR):
        os.mkdir(opt.MODEL_DIR)

    logger = Logger(args.model_name, path=opt.MODEL_DIR)
    writer = SummaryWriter(opt.MODEL_DIR)

    shutil.copytree('../code/', opt.MODEL_DIR + '/code/', shutil.ignore_patterns(['.git','__pycache__']))

    # dataset
    data_spliter = DatasetSpliter(opt=opt, input_size=opt.INPUT_SIZE)

    for fold_idx in range(opt.KFOLD):
        print('### {} of {} FOLD ###'.format(fold_idx + 1, opt.KFOLD))
        train_loader, test_loader = data_spliter.get_datasets(fold_idx=fold_idx)
        rand_seed(opt.RANDOM_SEED)

        # Training Config
        epochs = args.epochs
        epoch_start = 0

        net = init_model(args, opt)
        optimizer, loss_fct = init_optimization(net, args)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Resume
        if args.RESUME_FROM > 0:
            ckpt = torch.load(os.path.join(opt.MODEL_DIR, '{}_{}_{}_{}.pth'.format(args.model_name, opt.DATASET, fold_idx, args.RESUME_FROM)))
            net.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt.keys():
                scheduler.load_state_dict(ckpt['scheduler'])
            epoch_start = args.RESUME_FROM

        net.cuda()
        # Training
        best_metric = 0
        for epoch in range(epoch_start, epochs):
            net.train()
            print_str = '-------epoch {}/{}-------'.format(epoch+1, epochs)
            logger.write_and_print(print_str)

            for step, (patch, masks, _) in enumerate(tqdm(train_loader)):

                patch, mask = data_preprocess(patch, masks, training=True)
                patch = patch.cuda()

                # prepare data
                batches_done = len(train_loader) * epoch + step
                optimizer.zero_grad()

                ann_array = []
                labels = []
                for i in range(patch.shape[0]):
                    random_idx = np.random.randint(0,4)
                    ann_array.append(random_idx)
                    labels.append(mask[i, random_idx].unsqueeze(0).unsqueeze(0))
                labels = torch.cat(labels, dim=0).cuda()
                ann_array = torch.Tensor(ann_array).cuda().float()

                loss, _ = net.train_step(patch, labels, loss_fct, ann_array)
                if torch.isnan(loss):
                    logger.write_and_print('***** Warning: loss is NaN *****')
                    loss = torch.tensor(10000).cuda()

                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/train fold_idx-{}'.format(fold_idx), loss.item(), batches_done)

            # log learning_rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('LR/fold_idx-{}'.format(fold_idx), current_lr, epoch)
            scheduler.step()

            # # save model
            # if epoch % 20 == 0:
            #     ckpt = {'model': net.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict()
            #             }
            #     torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_{}.pth'.format(args.model_name, opt.DATASET, fold_idx, epoch)))

            metrics_dict = validate(net, test_loader, opt, writer, epoch)
            print_str = ''
            for key in metrics_dict.keys():
                print_str += key + ': {:.4f}  '.format(metrics_dict[key])
                writer.add_scalar('Metrics/'+key, metrics_dict[key], epoch)
            logger.write_and_print(print_str)

            metric_instance_ = metrics_dict['Dice_max'] + metrics_dict['Dice_soft']
            if metric_instance_ >= best_metric:
                best_metric = metric_instance_
                logger.write_and_print("Best Dice Max and Mean: {}".format(best_metric))
                ckpt = {'model': net.state_dict()}
                torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_{}_best.pth'.format(args.model_name, opt.DATASET, fold_idx, epoch)))
                torch.save(ckpt, os.path.join(opt.MODEL_DIR, '{}_{}_{}_best.pth'.format(args.model_name, opt.DATASET, fold_idx)))


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()