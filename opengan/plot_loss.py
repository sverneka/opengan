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

a = pd.read_csv('out_a_loss.csv')
b = pd.read_csv('out_b_loss.csv')

plt.figure()
plt.title('Plot of cross-entropy loss and attention loss for "GAN with attention loss" training')
plt.xlabel('epochs [0-29]')
plt.ylabel('loss')
plt.plot(a['Step'], a['Value'])
plt.plot(b['Step'], b['Value'])

plt.legend(('cross-entorpy loss','attention loss'), fontsize='x-small')
plt.show()
#plt.savefig('sense_ICDAR.png');

