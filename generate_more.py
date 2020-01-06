import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# generar mas imagenes
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Lee todas las imagenes que se encuentren en el directorio.
for filename in os.listdir('out/'):
    if filename == 'more': continue
    image = load_img('out/' + filename)
    x = img_to_array(image)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, save_to_dir='out/more',save_prefix=filename[-5]):
        i += 1
        if i >5: break
