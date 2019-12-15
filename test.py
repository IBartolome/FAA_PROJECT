from skimage import morphology
from scipy import ndimage
from skimage import transform as tf
import matplotlib.pyplot as plt

import random
import os

import numpy as np


def transform_image(img):

    images = []

    # tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -20))
    # tform = tf.AffineTransform(shear=0.3, translation=(30, 0))
    # tform = tf.AffineTransform(scale=(1.3, 1.1), shear=-0.3, translation=(-50, 0))
    # tform = tf.AffineTransform(rotation=0.5)
    # tform = tf.AffineTransform(translation=(0, -20))


    # t1_form = tf.AffineTransform(scale=(1.5, 1.3), shear=0.2)
    # t2_form = tf.AffineTransform(scale=(2.4, 2.2), translation=(-60, -80))
    # t3_form = tf.AffineTransform(scale=(3, 2.8), translation=(-60, -80))
    # t4_form = tf.AffineTransform(shear=0.2, translation=(30, 0))
    # t5_form = tf.AffineTransform(scale=(1.5, 1.3), shear=-0.2, translation=(-50, 0))
    # t6_form = tf.AffineTransform(scale=(1.4, 1.2), rotation=0.2)
    t7_form = img * np.random.randint(2, size=img.shape)
    t7_form = img * np.random.random_sample(img.shape)


    # images.append( img )

    # images.append( tf.warp(img, t1_form) )
    # images.append( tf.warp(img, t2_form) )
    # images.append( tf.warp(img, t3_form) )
    # images.append( tf.warp(img, t4_form) )
    # images.append( tf.warp(img, t5_form) )
    # images.append( tf.warp(img, t6_form) )
    images.append( t7_form )
    

    # return [tf.warp(img, tform)]
    return images


def delete_background(img):

    # tranf = ndimage.median_filter(img, size=3)

    tranf = img.copy()

    tranf[(img > 0.6)] = 0
    tranf[(img <= 0.6)] = 1

    # tranf = morphology.binary_closing(tranf)
    tranf = morphology.binary_opening(tranf)

    return tranf



dirname = "out/"

files = os.listdir(dirname)

dir_img = files[random.randint(0, len(files) - 1)]

print(dir_img)

# dir_img = "00421_e.png"


img1 = plt.imread(dirname + dir_img)

# img1 = transform_image(img1)
# img1_t = median(img1, )
# img1_t = median(img1_t)


img1_t = delete_background(img1)

# img1_t[(img1_t == 0)] = 2
# img1_t[(img1_t == 1)] = 0
# img1_t[(img1_t == 2)] = 1

# img1_t = transform_image(img1_t)
images = transform_image(img1_t)

print(img1.shape)
print(img1_t.shape)

for transf in images:

    fig = plt.figure()

    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side

    ax1.imshow(img1, cmap='gray')
    ax2.imshow(transf, cmap='gray')

    plt.show()