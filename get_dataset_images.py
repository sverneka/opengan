import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
from copy import deepcopy


from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np
import cv2



# Prediction on normal mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255.
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# notMNIST dataset
data_notMnist = np.load('notMNIST.npy').astype('float32')/255.


# ICDAR-2003 dataset
data = np.load('icdar.npy')
labels = data.item().get('labels')
alphabets = data.item().get('alphabets').astype('float32')/255.


img_1 = X_train[0][:,:,0]
img_2 = data_notMnist[1][:,:,0]
img_3 = alphabets[0][:,:,0]
print img_1.shape
print img_2.shape
print img_3.shape


fig, axs = plt.subplots(1, 3)
#fig.suptitle("DCGAN: Generated digits", fontsize=12)
axs[0].imshow(img_1, cmap='gray')
axs[0].axis('off')
axs[1].imshow(img_2, cmap='gray')   
axs[1].axis('off')    
axs[2].imshow(img_3, cmap='gray')       
axs[2].axis('off')

fig.savefig("dataset_images.png")

