import torch
import numpy as np
import random

def norm_img(x, eps=1e-8):
    x = (x - x.min()) / ((x.max() - x.min()+eps))
    return x

def entropy(x, eps=1e-8):
    ex = - x * torch.log(x+eps)
    return ex

def rand_seed(SEED=1234):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

def show_img(patch, preds, masks):
    bs, a_num, width, height = masks.size()
    concat_pred = np.zeros([3*width, height * (a_num+2)])
    soft_preds = preds.mean(1).squeeze(0)
    soft_masks = masks.mean(1).squeeze(0)
    concat_pred[:width, :height] = norm_img(patch[0,0]).cpu().numpy()
    concat_pred[width:2*width, :height] = norm_img(entropy(soft_preds)).numpy()
    concat_pred[2*width:, :height] = norm_img(entropy(soft_masks)).numpy()

    concat_pred[width:2*width, height:2*height] = soft_preds.numpy()
    concat_pred[2*width:, height:2*height] = soft_masks.numpy()
    concat_pred[:width, height:2*height] = (soft_preds > 0.5).float().numpy()

    for idx in range(2, a_num+2):
        concat_pred[:width, height * idx:height * (idx+1)] = (preds[0,idx-2] > 0.5).float().numpy()
        concat_pred[width:2*width, height * idx:height * (idx+1)] = preds[0,idx-2].numpy()
        concat_pred[2*width:, height * idx:height * (idx+1)] = masks[0,idx-2].numpy()
    return concat_pred

def get_label_pred_list(opt, bs, masks, output_sample_list):
    label_list = []
    for idx in range(bs):
        temp_label_list = []
        for anno_no in range(opt.MASK_NUM):
            temp_label = masks[idx, anno_no, :, :].to(dtype=torch.float32)
            temp_label_list.append(temp_label)
        label_list.append(temp_label_list)

    pred_list = []
    for idx in range(bs):
        temp_pred_list = []
        for pred_no in range(len(output_sample_list)):
            temp_pred = torch.sigmoid(output_sample_list[pred_no][idx, :, :])
            temp_pred_list.append(temp_pred)
        pred_list.append(temp_pred_list)
    return label_list, pred_list
