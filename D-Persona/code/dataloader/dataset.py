import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
from skimage import exposure

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        modality="t1c",
        transform=None
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.modality = modality

        if self.split == "train":
            self.sample_list = os.listdir(self._base_dir + "/training_2d")
        elif self.split == "val":
            self.sample_list = os.listdir(self._base_dir + "/validation")
        elif self.split == "test":
            self.sample_list = os.listdir(self._base_dir + "/testing")
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +"/training_2d/{}".format(case), "r")
        elif self.split == "val":
            h5f = h5py.File(self._base_dir + "/validation/{}".format(case), "r")
        elif self.split == "test":
            h5f = h5py.File(self._base_dir + "/testing/{}".format(case), "r")

        # image = h5f[self.modality][:]
        image_modality_list = ["t1", "t1c", "t2"]
        image = np.array([h5f[modality][:] for modality in image_modality_list])
        
        if self.split == "train":
            label = np.zeros((4, image.shape[1], image.shape[2]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]
        else:
            label = np.zeros((4, image.shape[1], image.shape[2], image.shape[3]))
            label[0] = h5f["label_a1"][:]
            label[1] = h5f["label_a2"][:]
            label[2] = h5f["label_a3"][:]
            label[3] = h5f["label_a4"][:]

        sample = {"image": image, "label": label}
        sample = self.transform(sample)

        sample["idx"] = case
        return sample

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)

    if len(image.shape) == 2:
        image = np.rot90(image, k)
        image = np.flip(image, axis=axis).copy()
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = np.rot90(image[i], k)
            image[i] = np.flip(image[i], axis=axis).copy()
    if len(label.shape) == 2:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = np.rot90(label[i], k)
            label[i] = np.flip(label[i], axis=axis).copy()

    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if len(image.shape) == 2:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
    elif len(image.shape) == 3:
        for i in range(image.shape[0]):
            image[i] = ndimage.rotate(image[i], angle, order=0, reshape=False)
    if len(label.shape) == 2:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    elif len(label.shape) == 3:
        for i in range(label.shape[0]):
            label[i] = ndimage.rotate(label[i], angle, order=0, reshape=False)
    return image, label

def random_noise(image, label, mu=0, sigma=0.1):
    if len(image.shape) == 2:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1]), -2 * sigma, 2 * sigma)
    elif len(image.shape) == 3:
        noise = np.clip(sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * sigma, 2 * sigma)
    else:
        pass
    noise = noise + mu
    image = image + noise
    return image, label


def random_rescale_intensity(image, label):
    image = exposure.rescale_intensity(image)
    return image, label

def random_equalize_hist(image, label):
    image = exposure.equalize_hist(image)
    return image, label

class RandomGenerator_Multi_Rater(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        _, x, y = image.shape

        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        if len(label.shape) == 2:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        elif len(label.shape) == 3:
            label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = random_noise(image, label)
        # if random.random() > 0.5:
        #     image, label = random_rescale_intensity(image, label)
        # if random.random() > 0.5:
        #     image, label = random_equalize_hist(image, label)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample

class ZoomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        c, d, x, y = image.shape

        image = zoom(image, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (1, 1, self.output_size[0] / x, self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
