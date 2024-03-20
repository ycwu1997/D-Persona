import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import random
import torchvision.transforms
import torchvision.transforms.functional as F
from PIL import Image


# Function to distort image
def elastic_transform_func(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


def elastic_transform(im_stack):
    # Merge images into separete channels (shape will be (cols, rols, dims))
    im_merge = np.concatenate(im_stack, axis=2)

    # Apply transformation on image
    im_merge_t = elastic_transform_func(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.15, im_merge.shape[1] * 0.05)

    # Split image and mask
    outputs = im_merge_t

    return outputs


# -180~180 rotation
def rotate(inputs, angle=None):
    temp = []
    for i in range(len(inputs)):
        temp.append(Image.fromarray(inputs[i]))
    if angle is None:
        angle = torchvision.transforms.RandomRotation.get_params([-180, 180])
    if isinstance(angle, list):
        angle = random.choice(angle)
    for i in range(len(temp)):
        temp[i] = temp[i].rotate(angle)
    for i in range(len(inputs)):
        inputs[i] = np.array(temp[i])
    return inputs


# flip
def flip(inputs):
    temp = []
    for i in range(len(inputs)):
        temp.append(Image.fromarray(inputs[i]))
    if random.random() > 0.5:
        for i in range(len(temp)):
            temp[i] = F.hflip(temp[i])
    if random.random() < 0.5:
        for i in range(len(temp)):
            temp[i] = F.vflip(temp[i])
    for i in range(len(inputs)):
        inputs[i] = np.array(temp[i])
    return inputs
