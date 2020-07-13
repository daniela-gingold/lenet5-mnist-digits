# lenet5-mnist-digits

This is keras implementation for MNIST Digits classification using LeNet-5. 

training.py file contains the full implementation of the model training pipline.
model.h5 is the best model (with the smallest validation loss).
training_log.csv contains loss and accuracy documentation for each epoch during the trainig.
plot.png is the training progress plot of loss and accuracy as follows:

![plot](/plot.png)

MNIST Digits is a database of 28x28 grayscale images of 10 handwritten digits, which includes 60,000 training images and 10,000 test images. 

LeNet-5 is a Convolutional Neural Network proposed by Yann LeCun et al. in 1998. The network consists of 5 parametarized layers ~ 60,000 parameters. Attached is the architecture of the network and its summary table:

![1](/lenet5-architecture/pic1.png)

![2](/lenet5-architecture/pic2.png)




 