
from scipy import ndimage
from skimage import morphology
from skimage import transform as tf
import numpy as np


def delete_background(img):

    tranf = img.copy()

    tranf[(img > 0.6)] = 0
    tranf[(img <= 0.6)] = 1

    tranf = morphology.binary_opening(tranf)

    return tranf


def transform_image(img):

    images = []

    # t1_form = tf.AffineTransform(scale=(1.5, 1.3), shear=0.2)
    t2_form = tf.AffineTransform(scale=(2.4, 2.2), translation=(-60, -80))
    # t3_form = tf.AffineTransform(scale=(3, 2.8), translation=(-60, -80))
    # t4_form = tf.AffineTransform(shear=0.2, translation=(30, 0))
    # t5_form = tf.AffineTransform(scale=(1.5, 1.3), shear=-0.2, translation=(-50, 0))
    # t6_form = tf.AffineTransform(scale=(1.4, 1.2), rotation=0.2)
    t7_form = img * np.random.random_sample(img.shape)


    images.append( img )

    # images.append( tf.warp(img, t1_form) )
    images.append( tf.warp(img, t2_form) )
    # images.append( tf.warp(img, t3_form) )
    # images.append( tf.warp(img, t4_form) )
    # images.append( tf.warp(img, t5_form) )
    # images.append( tf.warp(img, t6_form) )
    images.append( t7_form )

    return images