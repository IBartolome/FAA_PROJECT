## Modelo con LENET5

Uno de los modelos que hemos realizado es LeNet5, una arquitectura más que conocida que nació en los 90.

### Arquitectura

La arquitectura es basa en varias convoluciones, max-poolings, dos full connections y un softmax.

![](/home/hnko/Desktop/lenet5.png)

Con la ayuda de Keras pudimos realizarlo un pocas líneas de código.

```python
    model = Sequential()
    model.add(Conv2D(filters=6, 
                     kernel_size=(5,5), 
                     activation='relu', 
                     input_shape=(1,165,120)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    #optimizer
    sgd = SGD(lr=0.1)
    model.compile(loss=categorical_crossentropy, 
                  optimizer=sgd, 
                  metrics=['accuracy'])

```
### Parámetros

Los parámetros de entrenamiento fueron **60 épocas** y un **tamaño de batch de 128**.



### Resultados

Los resultados que se obtuvieron con esta arquitectura son los siguientes:

Captura de pantalla donde el programa reconoce correctamente la letra 'e'.

![97%](/home/hnko/Desktop/97%.png)



Si miramos la terminal, vemos que el accuracy del modelo fue del 97%.

![](/home/hnko/Desktop/zoom-accuracy97%.png)



Loss del modelo:

![](/home/hnko/Desktop/97%_loss.png)

Accuracy del modelo:

![97%_accuracy](/home/hnko/Desktop/97%_accuracy.png)



### Observación:

Con el dataset inicial, se consiguió alrededor del 94% de accuracy, pero quisimos probar esta arquitectura con un dataset mayor, ya que en el proporcionado sólo había mil imágenes.

Para ello, utilizamos la clase **ImageDataGenerator** de

```python
keras.preprocessing.image.ImageDataGenerator
```

Con ella pudimos generar más de 6000 imágenes adicionales para entrenar a nuestro modelo. Las opciones seleccionadas fueron las siguientes:

```python
ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)
```

Mostramos a continuación un ejemplo:

![](/home/hnko/Desktop/u-lenet5.png)

Podemos observar que la letra está desplaza para arriba-izquierda y el espacio que deja al desplazarse se rellena con los valores más cercanos, como se especificó en fill_mode='nearest'.



