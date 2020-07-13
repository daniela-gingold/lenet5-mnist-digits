from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import layers
import numpy as np
import keras
import os

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize value to [0, 1]
x_train = x_train/255
x_test = x_test/255

# transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# reshape the data set into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)


# instantiate an empty model
model = Sequential()
# build model
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding='same'))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

# compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])

# print model architecture
model.summary()

# prepare directory for saving model
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'mnist_digits_%s_model.{epoch:03d}.h5' % 'lenet5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True)
csvlogger = CSVLogger('training_log', separator=',', append = False)
callbacks = [checkpoint, csvlogger]

# train the model
hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),epochs=1, batch_size=128, verbose=1, callbacks=callbacks)

# freeze the model
# model.save('lenet5_digits.h5')

# evaluate predictions
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


