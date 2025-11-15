from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import KFold
import os
import copy
import numpy as np
import cv2
import pickle

class Dataset(Dataset):
    def __init__(self, dataset_location, input_size=128):
        self.images = []
        self.mask_labels = []
        self.series_uid = []

        # read dataset
        max_bytes = 2 ** 31 - 1
        data = {}
        print("Loading file", dataset_location)
        bytes_in = bytearray(0)
        file_size = os.path.getsize(dataset_location)
        with open(dataset_location, 'rb') as f_in:
            for _ in range(0, file_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        new_data = pickle.loads(bytes_in)
        data.update(new_data)

        # load dataset
        for key, value in data.items():
            # image 0-255, alpha 0-255, mask [0,1]
            self.images.append(pad_im(value['image'], input_size))
            masks = []
            for mask in value['masks']:
                masks.append(pad_im(mask, input_size))
            self.mask_labels.append(masks)
            self.series_uid.append(value['series_uid'])

        # check
        assert (len(self.images) == len(self.mask_labels) == len(self.series_uid))
        for image in self.images:
            assert np.max(image) <= 255 and np.min(image) >= 0
        for mask in self.mask_labels:
            assert np.max(mask) <= 1 and np.min(mask) >= 0

        # free
        del new_data
        del data

    def __getitem__(self, index):
        image = copy.deepcopy(self.images[index])
        mask_labels = copy.deepcopy(self.mask_labels[index])
        series_uid = self.series_uid[index]

        return image, mask_labels, series_uid

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)


def pad_im(image, size, value=0):
    shape = image.shape
    if len(shape) == 2:
        h, w = shape
    else:
        h, w, c = shape

    if h == w:
        if h == size:
            padded_im = image
        else:
            padded_im = cv2.resize(image, (size, size), cv2.INTER_CUBIC)
    else:
        if h > w:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            padded_im = cv2.copyMakeBorder(image, 0, 0, pad_1, pad_2, cv2.BORDER_CONSTANT, value=value)
        else:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            padded_im = cv2.copyMakeBorder(image, pad_1, pad_2, 0, 0, cv2.BORDER_CONSTANT, value=value)
    if padded_im.shape[0] != size:
        padded_im = cv2.resize(padded_im, (size, size), cv2.INTER_CUBIC)

    return padded_im

class DatasetSpliter():
	def __init__(self, opt, input_size):
		self.opt = opt
		self.train_dataset = Dataset(dataset_location=opt.DATA_PATH, input_size=input_size)
		self.test_dataset = Dataset(dataset_location=opt.DATA_PATH, input_size=input_size)
		self.kf = KFold(n_splits=opt.KFOLD, shuffle=False)

		self.splits = []

		if opt.DATASET == 'LIDC':
			uid_dict = {}
			for idx, uid in enumerate(self.train_dataset.series_uid):
				pid = uid.split('_')[0]
				if pid in uid_dict.keys():
					uid_dict[pid].append(idx)
				else:
					uid_dict[pid] = [idx]

			pids = list(uid_dict.keys())
			np.random.seed(opt.RANDOM_SEED)
			np.random.shuffle(pids)
			for (train_pid_index, test_pid_index) in self.kf.split(np.arange(len(pids))):
				train_index = []
				test_index = []
				for pid_idx in train_pid_index:
					train_index += uid_dict[pids[pid_idx]]
				for pid_idx in test_pid_index:
					test_index += uid_dict[pids[pid_idx]]
				self.splits.append({'train_index': train_index, 'test_index': test_index})
		else:
			indices = list(range(len(self.train_dataset)))
			np.random.seed(opt.RANDOM_SEED)
			np.random.shuffle(indices)
			for (train_index, test_index) in self.kf.split(np.arange(len(self.train_dataset))):
				self.splits.append({
					'train_index': [indices[i] for i in train_index.tolist()],
					'test_index': [indices[i] for i in test_index.tolist()]})

	def get_datasets(self, fold_idx):
		train_indices = self.splits[fold_idx]['train_index']
		test_indices = self.splits[fold_idx]['test_index']
		train_sampler = SubsetRandomSampler(train_indices)
		test_sampler = SubsetRandomSampler(test_indices)
		train_loader = DataLoader(self.train_dataset, batch_size=self.opt.TRAIN_BATCHSIZE, sampler=train_sampler)
		test_loader = DataLoader(self.test_dataset, batch_size=self.opt.VAL_BATCHSIZE, sampler=test_sampler)
		print("Number of training/test patches:", (len(train_indices), len(test_indices)))

		return train_loader, test_loader
