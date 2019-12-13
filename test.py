# from skimage.filters import median
from scipy import ndimage
from skimage import transform as tf
import matplotlib.pyplot as plt

import random
import os

import numpy as np


def transform_image(img):

    # tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -20))
    tform = tf.AffineTransform(scale=(1.3, 1.1), shear=0.5)
    # tform = tf.AffineTransform(rotation=0.5)
    # tform = tf.AffineTransform(translation=(0, -20))

    return tf.warp(img, tform)


def delete_background(img):

    tranf = ndimage.median_filter(img, size=10)

    tranf[(tranf > 0.6)] = 1
    tranf[(tranf <= 0.6)] = 0

    return tranf



dirname = "out/"

files = os.listdir(dirname)

dir_img = files[random.randint(0, len(files) - 1)]

print(dir_img)

img1 = plt.imread(dirname + dir_img)

# img1 = transform_image(img1)
# img1_t = median(img1, )
# img1_t = median(img1_t)


img1_t = delete_background(img1)

# trans = np.copy(img1_t)

img1_t[(img1_t == 0)] = 2
img1_t[(img1_t == 1)] = 0
img1_t[(img1_t == 2)] = 1

img1_t = transform_image(img1_t)

print(img1.shape)
print(img1_t.shape)

fig = plt.figure()

ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side

ax1.imshow(img1, cmap='gray')
ax2.imshow(img1_t, cmap='gray')

plt.show()