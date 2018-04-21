import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Input
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
from copy import deepcopy
from keras import backend as K

img_height, img_width = 28, 28
number_of_classes = 10



def attention_loss(y_true, y_pred):
    #y_pred[:,10:] /= K.sum(y_pred[:,num_classes:],
    #                            len(y_pred[:,num_classes:].get_shape()) - 1,
    #                            True)
    margin = 0.1
    #return K.categorical_crossentropy(y_true, y_pred[:,0:num_classes]) + 1./ K.maximum(0.1, K.sum(y_true * K.abs(y_pred[:,0:num_classes]-y_pred[:,num_classes:])))
    #return K.categorical_crossentropy(y_true, y_pred[:,0:num_classes]) - K.sum(y_true * K.log(K.maximum(0.1, K.abs(y_pred[:,0:num_classes]-y_pred[:,num_classes:]))))
    #return K.categorical_crossentropy(y_true, y_pred[:,0:number_of_classes]) + K.mean(K.maximum(y_pred[:,number_of_classes:] - margin, 0))
    return K.sum(K.maximum(y_pred - margin, 0), axis=-1) #+return K.categorical_crossentropy(y_true, y_pred[:,0:number_of_classes])


def base_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_height,img_width,1)))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(trainable=False))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization(trainable=False))
    model.add(Conv2D(64,(3, 3)))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(trainable=False))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # Fully connected layer
    #model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(Activation('relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(number_of_classes))
    # model.add(Convolution2D(10,3,3, border_mode='same'))
    # model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    return model

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

base_network = base_model()
input_a = Input(shape=(img_height,img_width,1))
input_b = Input(shape=(img_height,img_width,1))

processed_a = base_network(input_a)
processed_b = base_network(input_b)

#merged_output = concatenate([processed_a, processed_b], axis=-1)
out_a = Lambda(lambda x: x, name = 'out_a')(processed_a)
out_b = Lambda(lambda x: x, name = 'out_b')(processed_b)

#model = Model(input=[input_a, input_b], output=merged_output)
model = Model(inputs=[input_a, input_b], outputs=[out_a, out_b])
model.compile(loss={'out_a':'categorical_crossentropy', 'out_b':attention_loss}, loss_weights={'out_a': 1., 'out_b': 0.5}, optimizer=Adam(0.0001), metrics={'out_a':'acc'})

model.load_weights('mnist_gan_newloss.h5')

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

number_of_classes = 10

Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
y_train[0], Y_train[0]

print("evaluate on train")
print model.evaluate([X_train, X_train], [Y_train, Y_train])

print("evaluate on test")
print model.evaluate([X_test, X_test], [Y_test, Y_test])

# notMNIST dataset
data = np.load('notMNIST.npy').astype('float32')/255.
test = np.vstack((X_test, data))

y_true = np.concatenate((np.argmax(Y_test, axis=-1), 10 * np.ones(data.shape[0])))

preds = model.predict([test, test])
preds = preds[0]

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

preds = model.predict([test, test])
preds = preds[0]
y_true = np.concatenate((np.argmax(Y_test, axis=-1), 10 * np.ones(alphabets.shape[0])))

a.append(do_sensitivity_analysis(preds, y_true))


np.save('gan_with_loss_sense.npy', a)

