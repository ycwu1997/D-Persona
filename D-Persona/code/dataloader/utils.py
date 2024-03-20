import torch
import numpy as np
import random
import copy

from dataloader.transform import rotate, flip, elastic_transform


def data_augmentation(patch, mask_labels, apply_elastic_transform=False):
	inputs = mask_labels + [patch]

	# flip
	inputs = flip(inputs)
	# rotate
	outputs = rotate(inputs)

	# elastic transform
	if apply_elastic_transform:
		# elastic transform
		image_num = len(outputs)
		im_stack = []
		for i in range(image_num):
			if len(inputs[i].shape) == 2:
				im_stack.append(inputs[i].astype('uint8')[..., None])
			else:
				im_stack.append(inputs[i].astype('uint8'))

		outputs = elastic_transform(im_stack)
		masks, patch = outputs[..., :len(mask_labels)], np.squeeze(outputs[..., len(mask_labels):])

		assert len(mask_labels) == masks.shape[2]
		for i in range(len(mask_labels)):
			mask_labels[i] = masks[..., i]
	else:
		for i in range(len(mask_labels)):
			mask_labels[i] = outputs[i]
		patch = outputs[len(mask_labels)]

	return patch, mask_labels


def preprocess_func(patch, mask_labels, augmentation=True, apply_elastic_transform=False):
	# augmentation
	if augmentation:
		patch, mask_labels = data_augmentation(patch, mask_labels, apply_elastic_transform=apply_elastic_transform)

	# expand dimension & convert to torch tensors
	if len(patch.shape) == 2:
		patch = np.expand_dims(patch, axis=0)
	else:
		# color patch
		patch = np.transpose(patch, (2, 0, 1))
	patch = torch.from_numpy(patch).type(torch.FloatTensor) / 255.0
	patch = patch.unsqueeze(0)

	for i in range(len(mask_labels)):
		mask_labels[i] = torch.from_numpy(np.expand_dims(mask_labels[i], axis=0)).type(torch.FloatTensor)
		mask_labels[i] = (mask_labels[i] > 0.5).float().unsqueeze(0)
	return patch, mask_labels


def data_preprocess(patch, mask_labels, training=True):
	patch_list, mask_labels_list = [], []
	assert (patch.shape[0] == mask_labels[0].shape[0])
	batch_num = patch.shape[0]
	for i in range(batch_num):
		patch_list.append(patch[i, ...].numpy().astype('uint8'))
		m = []
		for l in range(len(mask_labels)):
			m.append(mask_labels[l][i, ...].numpy().astype('uint8'))
		mask_labels_list.append(m)

	if not training:
		assert batch_num == 1
		p, m_list = preprocess_func(patch_list[0], mask_labels_list[0], augmentation=False)
		m_list = ranking(m_list, is_rank = True)
		return p, m_list

	patch_stack, mask_stack = [], []
	for i in range(batch_num):
		p, m_list = preprocess_func(copy.deepcopy(patch_list[i]),copy.deepcopy(mask_labels_list[i]))
		
		patch_stack.append(p)
		# mask generate
		m = ranking(m_list, is_rank = True)
		mask_stack.append(m)
	patch = torch.cat(patch_stack)
	mask = torch.cat(mask_stack)

	return patch, mask


def ranking(masks, is_rank = True):
	masks = torch.cat(masks, dim=1)
	if is_rank:
		area_m = masks.sum(2).sum(2)
		sort_index = torch.sort(area_m).indices
		new_masks = masks[0, sort_index]
		return new_masks
	else:
		return masks