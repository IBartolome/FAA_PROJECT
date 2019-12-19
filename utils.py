
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
    t_form = []

    t_form.append( tf.AffineTransform(scale=(1.2, 1), shear=0.3) )
    t_form.append( tf.AffineTransform(scale=(1.2, 1), shear=-0.3) )

    t7_form = img * np.random.choice([0, 1], img.shape, p=[0.1, 0.9])
    # t8_form = img * np.random.choice([0, 1], img.shape, p=[0.25, 0.75])


    images.append( img )


    for n_form in t_form:

        img_tf = tf.warp(img, n_form)

        img_tf[(img_tf > 0.6)] = 1
        img_tf[(img_tf <= 0.6)] = 0

        img_tf = center_image( img_tf )

        images.append( img_tf )    


    images.append( t7_form )
    # images.append( t8_form )

    return images


# def center_image(img):

#     cm = ndimage.measurements.center_of_mass(img)

#     tform = np.zeros(img.shape)

#     real_y = int(img.shape[0] / 2)
#     real_x = int(img.shape[1] / 2)

#     actu_y = int(cm[0])
#     actu_x = int(cm[1])

#     move_y = real_y - actu_y
#     move_x = real_x - actu_x

#     tf_form = tf.AffineTransform(translation=(-move_x, -move_y))

#     return tf.warp(img, tf_form)


def center_image(img):

    height = img.shape[-2]
    width = img.shape[-1]

    col_sum = np.where( np.sum(img, axis=0) > 0 )
    row_sum = np.where( np.sum(img, axis=1) > 0 )

    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1:y2, x1:x2]

    move_y = int(((y1 + height - y2) / 2) - y1)
    move_x = int(((x1 + width - x2) / 2) - x1)

    tf_form = tf.AffineTransform(translation=(-move_x, -move_y))

    return tf.warp(img, tf_form)