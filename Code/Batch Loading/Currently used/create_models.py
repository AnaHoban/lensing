import numpy as np
import tensorflow as tf
from tensorflow import keras

#### LOSS FUNCTIONS ####
def custom_loss_cfis(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_cfis), y_pred*np.sqrt(weights_cfis))

def custom_loss_ps1(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_ps1), y_pred*np.sqrt(weights_ps1))

def custom_loss_all(y_true, y_pred):
    return keras.losses.MSE(y_true*np.sqrt(weights_all), y_pred*np.sqrt(weights_all))


#### MODELS ####

def create_autoencoder1(shape):
    '''Autoencoder with pooling layers'''
    input_img = keras.Input(shape=shape)
    
    x = keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128)(x)
    encoded = keras.layers.Dense(1024*4)(x)
    
    x = keras.layers.Reshape((32,32,4))(encoded)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2DTranspose(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2,2))(x)
    x = keras.layers.Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same')(x)
    decoded = keras.layers.Conv2D(shape[2], (3,3), activation='linear', padding='same')(x)
    
    return keras.Model(input_img, decoded)


def create_autoencoder2(shape):
    '''Autoencoder with convolutional layers'''
    input_img = keras.Input(shape=shape)
    x = keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same')(input_img)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)

    y = keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(input_img)
    y = keras.layers.BatchNormalization()(y)
    encoded = keras.layers.Add()([x,y])
    
    x = keras.layers.Conv2DTranspose(32, kernel_size=4, activation='relu', padding='same')(encoded)
    x = keras.layers.Conv2DTranspose(16, kernel_size=4, activation='relu', padding='same')(x)
    decoded = keras.layers.Conv2D(shape[2], kernel_size=3, activation='linear', padding='same')(x)
    
    return keras.Model(input_img, decoded)

def create_classifier(encoder):
    '''Binary classifier'''
    model = keras.Sequential(encoder)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(1, activation="sigmoid"))   
    return model