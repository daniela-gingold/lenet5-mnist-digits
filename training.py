from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
from keras import layers
import keras
import os

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize values to range [0, 1]
x_train = x_train/255
x_test = x_test/255

# convert labels to one-hot encodings
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# reshape data set into 4D array
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

# instantiate an empty model
model = Sequential()
# build the model
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

# print the summary of the model
model.summary()

# prepare directory to save model
filepah = os.path.join(os.getcwd(), 'model.h5')

# prepare callbacks
checkpoint = ModelCheckpoint(filepath=filepah, monitor='val_accuracy', verbose=1, save_best_only=True)
csvlogger = CSVLogger('training_log.csv', separator=',', append = False)
callbacks = [checkpoint, csvlogger]

# train the model
hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128, verbose=1, callbacks=callbacks)



# plot loss and accuracy
plt.subplot(211)
plt.title('Cross Entropy Loss')
plt.plot(hist.history['loss'], color='purple', label='train')
plt.plot(hist.history['val_loss'], color='green', label='test')
plt.legend(loc="upper right")

plt.subplot(212)
plt.title('Classification Accuracy')
plt.plot(hist.history['accuracy'], color='purple', label='train')
plt.plot(hist.history['val_accuracy'], color='green', label='test')
plt.legend(loc="lower right")
plt.show()

# save plot
plt.savefig('plot.png')
plt.close()

# evaluate predictions
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


