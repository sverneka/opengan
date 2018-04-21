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


def do_sensitivity_analysis(preds, y_true):
    thresh = np.linspace(0,1,21)
    y_pred = np.argmax(preds, axis=-1)
    y_max = np.max(preds, axis=-1)
    acc = []
    for th in thresh:
        y_pred_copy = deepcopy(y_pred)
        ind = np.where(y_max <= th)[0]
        y_pred_copy[ind] = 10
        acc.append(np.where(y_pred_copy == y_true)[0].shape[0]*1.0/y_true.shape[0])
    print(acc)
    return acc

model = load_model('mnist_gan.h5')

# Prediction on normal mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)


#plt.imshow(X_train[0], cmap='gray')
#plt.title('Class '+ str(y_train[0]))

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255.
X_test/=255.

X_train.shape

number_of_classes = 11

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
y_train[0], Y_train[0]

print("evaluate on train")
print model.evaluate(X_train, Y_train)

print("evaluate on test")
print model.evaluate(X_test, Y_test)

# notMNIST dataset
data = np.load('notMNIST.npy').astype('float32')/255.
test = np.vstack((X_test, data))

y_true = np.concatenate((np.argmax(Y_test, axis=-1), 10 * np.ones(data.shape[0])))

preds = model.predict(test)

Y_true = np_utils.to_categorical(y_true, number_of_classes)
model.evaluate(test, Y_true)
a=[]
a.append(do_sensitivity_analysis(preds, y_true))


# ICDAR-2003 dataset
data = np.load('icdar.npy')
#digits = data.item().get('digits').astype('float32')/255.
labels = data.item().get('labels')
alphabets = data.item().get('alphabets').astype('float32')/255.

Y_digits = np_utils.to_categorical(labels, number_of_classes)

#print("evaluate on Y_digits")
#model.evaluate(digits, Y_digits)

#X_test = np.vstack((X_test, digits))
#Y_test = np.vstack((Y_test, Y_digits))

#print("evaluate on Y_test+Y_digits")
#model.evaluate(X_test, Y_test)

test = np.vstack((X_test, alphabets))

preds = model.predict(test)

y_true = np.concatenate((np.argmax(Y_test, axis=-1), 10 * np.ones(alphabets.shape[0])))

a.append(do_sensitivity_analysis(preds, y_true))
np.save('gan_sense.npy', a)




