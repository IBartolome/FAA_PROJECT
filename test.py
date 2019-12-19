from skimage import morphology
from scipy import ndimage
from skimage import transform as tf
import matplotlib.pyplot as plt

import random
import os

import numpy as np



def center_image(img):

    height, width = img.shape

    col_sum = np.where( np.sum(img, axis=0) > 0 )
    row_sum = np.where( np.sum(img, axis=1) > 0 )

    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1:y2, x1:x2]

    move_y = int(((y1 + height - y2) / 2) - y1)
    move_x = int(((x1 + width - x2) / 2) - x1)

    tf_form = tf.AffineTransform(translation=(-move_x, -move_y))

    return tf.warp(img, tf_form)




def transform_image(img):

    images = []
    t_form = []

    # t_form.append( tf.AffineTransform(scale=(1.2, 1), shear=0.2) )
    # t_form.append( tf.AffineTransform(scale=(1.2, 1), shear=-0.2) )

    # t_form.append( tf.AffineTransform(rotation=0.2) )
    # t_form.append( tf.AffineTransform(rotation=0.2) )

    # t2_form = tf.AffineTransform(scale=(2.4, 2.2), translation=(-60, -80))
    # t3_form = tf.AffineTransform(scale=(3, 2.8), translation=(-60, -80))
    # t4_form = tf.AffineTransform(shear=0.2, translation=(30, 0))
    # t5_form = tf.AffineTransform(scale=(1.5, 1.3), shear=-0.2, translation=(-50, 0))
    # t6_form = tf.AffineTransform(scale=(1.4, 1.2), rotation=0.2)
    # t7_form = img * np.random.choice([0, 1], img.shape, p=[0.1, 0.9])
    # t8_form = img * np.random.choice([0, 1], img.shape, p=[0.25, 0.75])


    images.append( img )


    for n_form in t_form:

        img_tf = tf.warp(img, n_form)

        img_tf[(img_tf > 0.6)] = 1
        img_tf[(img_tf <= 0.6)] = 0

        img_tf = center_image( img_tf )

        images.append( img_tf )    


    # images.append( t7_form )
    # images.append( t8_form )
    

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
# dir_img = "00032_i.png"
# dir_img = "00602_i.png"


img1 = plt.imread(dirname + dir_img)

img1_t = delete_background(img1)

img1_t = center_image(img1_t)

images = transform_image(img1_t)


for transf in images:

    fig = plt.figure()

    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side

    # ax1.imshow(img1_t, cmap='gray')
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(transf, cmap='gray')

    plt.show()