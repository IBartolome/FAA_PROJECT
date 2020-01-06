from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy

from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np


def lenet5(obj, verbose=False):
    """Arquitectura Lenet 5

    Returns
    -------
    type
        modelo Lenet 5

    """


    INIT_LR = 1e-3

    classes = np.unique(obj.y)
    nClasses = len(classes)

    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(1,165,120)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    if verbose:
        model.summary()

    #optimizer
    sgd = SGD(lr=0.1)
    model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

    return model
