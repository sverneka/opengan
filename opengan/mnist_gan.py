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

np.random.seed(25)

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

X_train/=255
X_test/=255

X_train.shape

number_of_classes = 11




#Load generator produced data to be labeled as class-11
gen_data = np.load('gen_data.npy')
# rescale data
gen_data = ((gen_data * 127.5) + 127.5)/255.

X_train = np.vstack((X_train, gen_data))
y_gen_data = 10 * np.ones(gen_data.shape[0])
y_train = np.concatenate((y_train, y_gen_data))


Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

y_train[0], Y_train[0]


# Three steps to Convolution
# 1. Convolution
# 2. Activation
# 3. Polling
# Repeat Steps 1,2,3 for adding more hidden layers

# 4. After that make a fully connected network
# This fully connected network gives ability to the CNN
# to classify the samples

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
#BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
#BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
# Fully connected layer

#BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
#BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(number_of_classes))

# model.add(Convolution2D(10,3,3, border_mode='same'))
# model.add(GlobalAveragePooling2D())
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=64, shuffle=True)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)


class_weight = {}
for i in range(number_of_classes-1):
    class_weight[i] = 1

class_weight[number_of_classes-1] = 1

tensorboard = TensorBoard(log_dir="mnist_gan_logs/{}".format(time()))

filepath="mnist_gan.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//64, epochs=30, validation_data=test_generator, validation_steps=X_test.shape[0]//64, class_weight=class_weight, callbacks=callbacks_list)

