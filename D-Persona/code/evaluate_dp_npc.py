import torch
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import nibabel as nib
from configs.config import *
from utils.utils import rand_seed, show_img
from lib.metrics_set import *
from dataloader.dataset import BaseDataSets, ZoomGenerator
from torch.utils.data import DataLoader
from lib.initialize_model import init_model

def validate(args, net, val_loader, opt, writer=None, times_step = 0):
    GED_global, Dice_max_reverse, Dice_soft, Dice_match, Dice_each_mean = 0.0, 0.0, 0.0, 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for val_step, sample in enumerate(tqdm(val_loader)):

            patch = sample['image'].cuda()
            masks = sample['label'].float()

            preds = []
            for idx in range(patch.shape[2]):
                if args.stage == 1:
                    output_slice = net.val_step(patch[:,:,idx], sample_num = args.val_num)
                elif args.stage == 2:
                    output_slice = net.test_step(patch[:,:,idx], sample_num = args.val_num).unsqueeze(2)
                output_slice = torch.sigmoid(output_slice).cpu()
                preds.append(output_slice)
            preds = torch.cat(preds, 2)
            # Dice score
            GED_iter = generalized_energy_distance(masks, preds)
            _, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(masks, preds, thresh=0.5)
            dice_soft_iter = dice_at_thresh(masks, preds)

            GED_global += GED_iter
            Dice_max_reverse += dice_max_reverse_iter
            Dice_soft += dice_soft_iter
            Dice_match += dice_match_iter
            Dice_each_mean += np.mean(dice_each_iter)

            index_z = preds.shape[2] // 2
            if opt.VISUALIZE:
                concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
                cv2.imshow('predictions', concat_pred)
                cv2.waitKey(0)
            
            if writer is not None and val_step == len(val_loader) // 2:
                concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
                writer.add_image('Images', concat_pred, times_step, dataformats='HW')

    # store in dict
    metrics_dict = {'GED': GED_global / len(val_loader),
                    'Dice_max_reserse': Dice_max_reverse / len(val_loader),
                    'Dice_soft': Dice_soft / len(val_loader),
                    'Dice_match': Dice_match / len(val_loader),
                    'Dice_each_mean': Dice_each_mean / len(val_loader)}

    return metrics_dict

def evaluate(net, test_loader, opt, args, result_path):
    GED_global, Dice_max, Dice_max_reverse, Dice_soft, Dice_match, Dice_each = 0.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0] * 4)

    net.eval()
    with torch.no_grad():
        for test_step, sample in enumerate(tqdm(test_loader)):

            patch = sample['image'].cuda()
            masks = sample['label'].float()

            preds = []
            for idx in range(patch.shape[2]):
                if args.stage == 1:
                    output_slice = net.val_step(patch[:,:,idx], sample_num = args.test_num)
                elif args.stage == 2:
                    output_slice = net.test_step(patch[:,:,idx], sample_num = args.test_num).unsqueeze(2)
                output_slice = output_slice.unsqueeze(2)
                output_slice = torch.sigmoid(output_slice).cpu()
                preds.append(output_slice)
            preds = torch.cat(preds, 2)
            
            GED_iter = generalized_energy_distance(masks, preds)
            # Dice score
            dice_max_iter, dice_max_reverse_iter, dice_match_iter, dice_each_iter= dice_at_all(masks, preds, thresh=0.5)
            dice_soft_iter = dice_at_thresh(masks, preds)

            GED_global += GED_iter
            Dice_match += dice_match_iter
            Dice_max += dice_max_iter
            Dice_max_reverse += dice_max_reverse_iter
            Dice_soft += dice_soft_iter
            Dice_each += np.array(dice_each_iter)

            index_z = preds.shape[2] // 2
            if opt.VISUALIZE:
                concat_pred = show_img(patch[:,:,index_z], preds[:,:,index_z], masks[:,:,index_z])
                cv2.imshow('predictions', concat_pred)
                cv2.waitKey(0)
            
            if opt.TEST_SAVE:
                patch = patch.cpu().numpy()
                masks = masks.numpy()
                preds = preds.numpy()
                nib.save(nib.Nifti1Image(patch[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t1.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(patch[0,1].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t1c.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(patch[0,2].astype(np.float32), np.eye(4)), result_path +  "%02d_image_t2.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,0].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a1.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,1].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a2.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,2].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a3.nii.gz" % test_step)
                nib.save(nib.Nifti1Image(masks[0,3].astype(np.float32), np.eye(4)), result_path +  "%02d_label_a4.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,0]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s1.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,1]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s2.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,2]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s3.nii.gz" % test_step)
                nib.save(nib.Nifti1Image((preds[0,3]>0.5).astype(np.float32), np.eye(4)), result_path +  "%02d_pred_s4.nii.gz" % test_step)

    # store in dict
    metrics_dict = {'GED': GED_global / len(test_loader),
                    'Dice_max': Dice_max / len(test_loader),
                    'Dice_max_reverse': Dice_max_reverse / len(test_loader),
                    'Dice_max_mean': (Dice_max_reverse + Dice_max) / (2 * len(test_loader)),
                    'Dice_match': Dice_match / len(test_loader),
                    'Dice_soft': Dice_soft / len(test_loader),
                    'Dice_each': Dice_each / len(test_loader),
                    'Dice_each_mean': np.mean(Dice_each) / len(test_loader)}

    return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/params_npc.yaml', help="config path (*.yaml)")
    parser.add_argument("--save_path", type=str, default='../models/DPersona2_NPC_20240319-152059//', help="save path")
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--model_name", type=str, default='DPersona')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--mask_num", type=int, default=4)
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    opt = Config(config_path=args.config)
    rand_seed(opt.RANDOM_SEED)

    evaluate_records = []
    db_test = BaseDataSets(
        base_dir=opt.DATA_PATH,
        split="test",
        transform=ZoomGenerator(opt.PATCH_SIZE)
    )
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    net = init_model(args, opt)

    ckpt = torch.load(os.path.join(args.save_path, '{}{}_{}_best.pth'.format(args.model_name, args.stage, opt.DATASET)))
    net.load_state_dict(ckpt['model'])
    net.cuda()

    result_path = args.save_path + 'results/'
    os.makedirs(result_path, exist_ok=True)

    metrics_dict = evaluate(net, test_loader, opt, args, result_path)
    evaluate_records.append(metrics_dict)
    for key in metrics_dict.keys():
        print(key, ': ', metrics_dict[key])

    print(args.save_path)
    with open(args.save_path + 'performance.txt', 'w') as f:
        for key in evaluate_records[0].keys():
            temp = []
            for record in evaluate_records:
                temp.append(record[key])
            print('{}: {}±{}'.format(key, np.mean(temp, axis=0), np.std(temp, axis=0, ddof=0)))
            f.writelines('{}: {}±{} \n'.format(key, np.mean(temp, axis=0), np.std(temp, axis=0, ddof=0))) 