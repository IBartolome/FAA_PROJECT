# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU


class LetterClasifier:

    # Codificación de las clases
    clases = {'A':0, 'E':1, 'I':2, 'O':3, 'U':4, 'a':5, 'e':6, 'i':7, 'o':8, 'u':9}
    id_class = ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']
    
    
    def __init__(self):
        '''
        Constructor.

        Argumentos:
            - self : Objeto
            
        Return:
            - Objeto
        '''
        self.images = []
        self.labels = []
        
        
    def generateModel(self, verbose=0):
        '''
        Genera el modelo CNN, con el cual resolver el problema.

        Argumentos:
            - self : Objeto
            - verbose : Imprime informacion sobre lo que sucede.
            
        Return:
            - Modelo sin entrenar
        '''

        INIT_LR = 1e-3
        
        classes = np.unique(self.y)
        nClasses = len(classes)
         
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(1,165,120)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2),padding='same'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(32, activation='linear'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5)) 
        model.add(Dense(nClasses, activation='softmax'))

        if verbose:
            model.summary()

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(INIT_LR),metrics=['accuracy'])

        self.model = model

        return model

        
    def load_images(self, dirname="out/", verbose=1):
        '''
        Carga las imagenes.

        Argumentos:
            - self : Objeto.
            - dirname : Raiz del dataset.
            - verbose : Imprime informacion sobre lo que sucede.
            
        Return:
            - Modelo sin entrenar
        '''

        if verbose:
            print("leyendo imagenes de ", dirname)

        # Lee todas las imagenes que se encuentren en el directorio.
        for filename in os.listdir(dirname):
            
            image = plt.imread(dirname + filename)
            
            # Guarda la imagen y la clase.
            self.images.append( np.expand_dims(image, axis=0) )
            self.labels.append(self.clases[filename[-5]])
            

        # Convierte de lista a numpy
        self.y = np.array(self.labels)
        self.X = np.array(self.images)
        
        self.generateModel()

        if verbose:

            # Find the unique numbers from the train labels
            classes = np.unique(self.y)
            nClasses = len(classes)

            print('Imagenes leidas:',len(self.images))
            print('Total number of outputs : ', nClasses)
            print('Output classes : ', classes)

            
            
    def setModel(self, model):
        self.model = model




    def train(self,epochs = 50,batch_size = 128,test_size=0.2,verbose=1):

        if os.path.exists("model.h5py"):
            self.load()
            return -1
        
        train_X,test_X,train_Y,test_Y = train_test_split(self.X,self.y,test_size=test_size)
        if verbose:
            print('Training data shape : ', train_X.shape, train_Y.shape)
            print('Testing data shape : ', test_X.shape, test_Y.shape)
         
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        
         
        # Change the labels from categorical to one-hot encoding
        train_Y_one_hot = to_categorical(train_Y)
        if verbose: 
            # Display the change for category label using one-hot encoding
            print('Original label:', train_Y[0])
            print('After conversion to one-hot:', train_Y_one_hot[0])
         
        train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=test_size, random_state=13)
        if verbose:
            print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)


    

        self.history = self.model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=verbose,validation_data=(valid_X, valid_label))
 
        # guardamos la red, para reutilizarla en el futuro, sin tener que volver a entrenar
        self.model.save("model.h5py")
        
    def evaluate(self,verbose=1):

        train_X,test_X,train_Y,test_Y = train_test_split(self.X,self.y,test_size=0.2)
        test_Y_one_hot = to_categorical(test_Y)
        test_eval = self.model.evaluate(test_X, test_Y_one_hot, verbose=verbose)
         
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])
        
    def show(self):
        
        train_X,test_X,train_Y,test_Y = train_test_split(self.X,self.y,test_size=0.2)
        y_prob = self.model.predict(train_X[0:1]) 
        y_classes = y_prob.argmax(axis=-1)
        plt.title("Clase: "+self.id_class[y_classes[0]])
        plt.imshow(train_X[0][0], cmap='gray')
        plt.show()
        
    def load(self,filename="model.h5py"):
        print("Cargado el modelo")
        self.model.load_weights(filename)


if __name__ == "__main__":
    
    m = LetterClasifier()
    m.load_images(verbose=0)
    m.train(epochs = 400,batch_size = 128,verbose=1)
    m.evaluate(verbose=0)
    m.show()

    try:
        
        plt.plot(m.history.history['acc'])
        plt.plot(m.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(m.history.history['loss'])
        plt.plot(m.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    except:
        pass