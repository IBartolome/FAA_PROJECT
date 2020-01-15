
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import numpy as np

import numpy as np
from skimage.feature import corner_peaks,peak_local_max

from skimage import io,color
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve2d


def detectar_puntos_interes_harris(imagen, sigma = 2, k = 0.000005):
    """
    Esta funcion detecta puntos de interes en una imagen dada.
    
    Para realizar la deteccion de puntos de interes se utilizara el detector
    de esquinas de Harris.
    
    Argumentos de entrada:
      - imagen: numpy array de tamaño [imagen_height, imagen_width].  
      - sigma: valor de tipo double o float que determina el factor de suavizado aplicado
      - k: valor de tipo double o float que determina la deteccion de Harris
    Argumentos de salida
      - coords_esquinas: numpy array de tamaño [num_puntos_interes, 2] con las coordenadas 
                         de los puntos de interes detectados en la imagen. 
    
    NOTA: no modificar los valores por defecto de las variables de entrada sigma y k, 
          pues se utilizan para verificar el correcto funciomaniento de esta funcion
    """

    #Filtro de sobel
    sobel = np.array([[1,2,1],
                      [0,0,0],
                      [-1,-2,-1]])
                    

    #Normalizado el formato de la imagen
    imagen = imagen.astype(float)
    normImg = np.array(imagen / 255)
    
    #Calculadas las derivadas con el kernel de sobel
    grady = convolve2d(normImg, sobel, mode='same')
    gradx = convolve2d(normImg, sobel.transpose(), mode='same')
    
    #Calculadas las derivadas parciales con el kernel de sobel
    dx2 = gradx**2
    dy2 = grady**2
    dxy = gradx * grady

    #Aplicamos el filtro Gaussiano al resultado de las derivadas parcioales
    gdx  = gaussian_filter(dx2,sigma,mode="constant")
    gdy  = gaussian_filter(dy2,sigma,mode="constant")
    gdxy = gaussian_filter(dxy,sigma,mode="constant")
    
    rxy = (gdx*gdy - gdxy*gdxy) - k*((gdx+gdy)**2)

    # corner_peaks suprime maximos
    det_corner = corner_peaks(rxy,min_distance=5,threshold_rel=0.2)

    # Peak_local_max coge todos
    coords_esquinas = peak_local_max(rxy,min_distance=5,threshold_rel=0.2)

    return coords_esquinas

def load_images(dirname="out/",verbose=1):
    images = []
    labels = []
    clases = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'a':5, 'e':6, 'i':7, 'o':8, 'u':9}
    if verbose:
        print("leyendo imagenes de ",dirname)

    for filename in os.listdir(dirname):
        
        image = plt.imread(dirname+ filename)

        kp =   detectar_puntos_interes_harris(image)
        
        images.append([sum(image.flatten()),sum(image.flatten() <= 0.4) / len(image.flatten()),sum(image.flatten() >= 0.6) / len(image.flatten()),len(kp)])
        labels.append(clases[filename[-5]])
        

    y = np.array(labels)
    X = np.array(images) #convierto de lista a numpy

    # Find the unique numbers from the train labels
    
    
    if verbose:            
        print('Imagenes leidas:',len(images))
        print('Total number of outputs : ', 10)
        print('Output classes : ', clases)

    return X,y

# Instantiate model with 1000 decision trees
errores = []
X,y = load_images()

for i in range(1,50):
    rf = RandomForestClassifier(n_estimators=20,max_depth=i, random_state=0)


    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 42)
    # Train the model on training data
    hist = rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Print out the mean absolute error (mae)
    print(i,'Error: ', np.sum(test_labels != predictions) / len(predictions.flatten()))
    errores.append(np.sum(test_labels != predictions) / len(predictions.flatten()))

import matplotlib.pyplot as plt

plt.plot(errores)
plt.ylabel("Error")
plt.xlabel("max_depth")
plt.show()
