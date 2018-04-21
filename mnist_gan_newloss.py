import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#%matplotlib inline
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.models import load_model
from time import time


np.random.seed(25)
img_height, img_width = 28, 28

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)


#plt.imshow(X_train[0], cmap='gray')
#plt.title('Class '+ str(y_train[0]))

X_train = X_train.reshape(X_train.shape[0], img_height, img_width, 1)#[0:59968,]
X_test = X_test.reshape(X_test.shape[0], img_height, img_width, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255.
X_test/=255.

X_train.shape

number_of_classes = 10
batch_size = 64



#Load generator produced data to be labelled as class-11
gen_data = np.load('gen_data.npy')
# rescale data
gen_data = ((gen_data * 127.5) + 127.5)/255.0
#gen_data = gen_data[0:33024,]
gen_data = np.vstack((gen_data, gen_data))[0:X_train.shape[0],]


Y_train = np_utils.to_categorical(y_train, number_of_classes)#[0:59968,]
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

def compute_accuracy(y_true, y_pred):
    #return K.categorical_accuracy(y_true, y_pred[:,0:num_classes])
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred[:,0:number_of_classes], axis=-1)),
                  K.floatx())



def attention_loss(y_true, y_pred):
    #y_pred[:,10:] /= K.sum(y_pred[:,num_classes:],
    #                            len(y_pred[:,num_classes:].get_shape()) - 1,
    #                            True)
    margin = 0.1
    #return K.categorical_crossentropy(y_true, y_pred[:,0:num_classes]) + 1./ K.maximum(0.1, K.sum(y_true * K.abs(y_pred[:,0:num_classes]-y_pred[:,num_classes:])))
    #return K.categorical_crossentropy(y_true, y_pred[:,0:num_classes]) - K.sum(y_true * K.log(K.maximum(0.1, K.abs(y_pred[:,0:num_classes]-y_pred[:,num_classes:]))))
    #return K.categorical_crossentropy(y_true, y_pred[:,0:number_of_classes]) + K.mean(K.maximum(y_pred[:,number_of_classes:] - margin, 0))
    return K.sum(K.maximum(y_pred - margin, 0), axis=-1) #+return K.categorical_crossentropy(y_true, y_pred[:,0:number_of_classes])


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

model.summary()

model.compile(loss={'out_a':'categorical_crossentropy', 'out_b':attention_loss}, loss_weights={'out_a': 1., 'out_b': 0.5}, optimizer=Adam(0.0001), metrics={'out_a':'acc'})
model_fe = Model(input_a, out_a)

gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)


#gen = ImageDataGenerator()

def multiple_generator(x1, x2, y1, y2):
    gen_0 = gen.flow(x1, y1, shuffle=False, batch_size=batch_size)
    gen_1 = gen.flow(x2, y2, shuffle=False, batch_size=batch_size)
    while True:
        i0 = gen_0.next()
        i1 = gen_1.next()
        yield [i0[0], i1[0]], [i0[1], i0[1]]


test_generator = multiple_generator(X_test, X_test, Y_test, Y_test)

train_generator = multiple_generator(X_train, gen_data, Y_train, np.zeros((gen_data.shape[0], number_of_classes)))


tensorboard = TensorBoard(log_dir="mnist_gan_newloss_logs/{}".format(time()))

filepath="mnist_gan_newloss.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_out_a_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0]//batch_size, epochs=30, validation_data=test_generator, validation_steps=X_test.shape[0]//batch_size, callbacks=callbacks_list)


#model.fit([X_train, gen_data], [Y_train, np.zeros((gen_data.shape[0], number_of_classes))], validation_data=([X_test, X_test], [Y_test, Y_test]), batch_size = batch_size, shuffle = True, epochs=1, callbacks=callbacks_list)

#model.fit([X_train, gen_data], [Y_train, np.zeros((gen_data.shape[0], number_of_classes))], validation_data=([X_train, X_train], [Y_train, Y_train]), batch_size = batch_size, shuffle = True, epochs=1, callbacks=callbacks_list)

#save model_fe
model = load_model(filepath, custom_objects={'attention_loss': attention_loss})
model_fe.layers[1] = model.layers[2]
model_fe.save(filepath)


#model_fe.compile(loss={'out_a':'categorical_crossentropy'}, loss_weights={'out_a': 1.0}, optimizer=Adam(0.00001), metrics={'out_a':'acc'})
#model_fe.fit(X_train, Y_train, validation_data=(X_train, Y_train), batch_size = batch_size, shuffle = True, epochs=1, callbacks=callbacks_list)





#model_fe.evaluate(X_test,Y_test)
#y_pred = model.predict([])

#model.evaluate([X_test, X_test],[Y_test, Y_test])
