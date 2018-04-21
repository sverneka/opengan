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

a = np.load('mnist_sense.npy')*100
print a.shape
b = np.load('gan_sense.npy')*100
print b.shape
c = np.load('gan_with_loss_sense.npy')*100
print c.shape


thresh = np.linspace(0,1,20)

plt.figure()
plt.title('Accuracy vs threshold for outlier detection methods for ICDAR-2003')
plt.xlabel('threshold [0-1]')
plt.ylabel('Accuracy (%)')
plt.plot(thresh, a[1][0:20])
plt.plot(thresh, b[1][0:20])
plt.plot(thresh, c[1][0:20])

plt.legend(('Thresholding','GAN Method','GAN with attention loss'), fontsize='x-small')
plt.show()
#plt.savefig('sense_ICDAR.png');

