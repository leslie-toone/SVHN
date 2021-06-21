# create a neural network that classifies real-world images digits into one of ten classes.
'''The SVHN dataset (http://ufldl.stanford.edu/housenumbers/) is an image dataset of over 600,000 digit images in all,
and is a harder dataset than MNIST as the numbers appear in the context of natural scene images.
SVHN is obtained from house numbers in Google Street View images.

Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng. "Reading Digits in Natural Images with Unsupervised
Feature Learning". NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.
Your goal is to develop an end-to-end workflow for building, training, validating, evaluating and saving a neural
 network that classifies a real-world image into one of ten classes.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Run this cell first to import all required packages. Do not make any imports elsewhere in the notebook
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


def load_data():
    x_train = np.load('data/x_train_SVHN.npy')
    y_train = np.load('data/y_train_SVHN.npy')
    x_test = np.load('data/x_test_SVHN.npy')
    y_test = np.load('data/y_test_SVHN.npy')
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

# look at the frequency counts of y_train
(unique, counts) = np.unique(y_train, return_counts=True)
percent = pd.DataFrame(
    (unique, counts, np.round((counts / len(y_train)) * 100, 0)))  # percent=(round(counts/len(y_train)),2)*100
frequencies = percent.T
print(frequencies)

plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i].reshape(32, 32), cmap='gray')
    plt.axis('off')
plt.show()
print('label for each of the above image: %s' % (y_train[0:10]))

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

print('shape of single image:', x_train[0].shape)
print(y_train[0])
# keras expects an integer vector from 0 to num_classes, so to get the 10 classes,
# will need to shift each class by 1

y_test = tf.keras.utils.to_categorical(np.array(y_test - 1), num_classes=10)
y_train = tf.keras.utils.to_categorical(np.array(y_train - 1), num_classes=10)
print(y_train[0])
plt.imshow(x_train[0])

# reshape data from 2D to 1D -> 32X32 to 1024
x_train_1D = np.asarray(x_train).reshape(73257, 1024)
x_test_1D = np.asarray(x_test).reshape(26032, 1024)
print("x_train:", x_train_1D.shape, "x_test:", x_test_1D.shape, "y_train:", y_train.shape, "y_test", y_test.shape)
# https://medium.com/data-science-bootcamp/multilayer-perceptron-mlp-vs-convolutional-neural-network-in-deep-learning-c890f487a8f1
# MLP is great for simpler more straight forward datasets (like MNIST) but lags behind CNN when it comes to real world
# application, specifically image classification. It is the vanilla neural network in use before all the fancy NN
# such as CNN, LSTM came along.
"""A multilayer perceptron (MLP) is a class of feedforward artificial neural network. A MLP consists of at least three 
layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron 
that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for 
training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish 
data that is not linearly separable.

Multilayer Perceptron (MLP): used to apply in computer vision, now succeeded by Convolutional Neural Network (CNN).
MLP is now deemed insufficient for modern advanced computer vision tasks. Has the characteristic of fully connected
layers, where each perceptron is connected with every other perceptron. Disadvantage is that the number of total 
parameters can grow to very high (number of perceptron in layer 1 multiplied by # of p in layer 2 multiplied by # 
of p in layer 3…). This is inefficient because there is redundancy in such high dimensions. Another disadvantage is 
that it disregards spatial information. It takes flattened vectors as inputs. A light weight MLP (2–3 layers) can 
easily achieve high accuracy with MNIST dataset."""

"""Convolutional Neural Network (CNN): the incumbent, current favorite of computer vision algorithms, winner of multiple 
ImageNet competitions. Can account for local connectivity (each filter is panned around the entire image according to 
certain size and stride, allows the filter to find and match patterns no matter where the pattern is located in a given 
image). The weights are smaller, and shared — less wasteful, easier to train than MLP. More effective too. Can 
also go deeper. Layers are sparsely connected rather than fully connected. It takes matrices as well as vectors as 
inputs. The layers are sparsely connected or partially connected rather than fully connected. 
Every node does not connect to every other node.
"""

'''Build an MLP classifier model using the Sequential API. Use only Flatten and Dense layers,
with the final layer having a 10-way softmax output.

Compile and train the model, making use of both training and validation sets during the training run.
Track at least one appropriate metric, and use at least two callbacks during training,
one of which should be a ModelCheckpoint callback.

Aim to achieve a final categorical cross entropy training loss of less than 1.0 (the validation loss might be higher).

Plot the learning curves for loss vs epoch and accuracy vs epoch for both training and validation sets.
Compute and display the loss and accuracy of the trained model on the test set.'''


def get_model(shape):
    model = Sequential([
        # dense is a fully connected layer with 64 hidden units,
        # in the first layer you must specify the expected shape
        Dense(64, kernel_initializer='he_normal', activation='relu', input_shape=shape),
        Dense(256, kernel_initializer='he_normal', activation='relu'),
        # we could pass any activation such as sigmoid/linear/tanh but it is proved that relu performs the best in
        # these kinds of models
        Flatten(),
        Dense(10, activation='softmax')
    ])
    return model


model_seq = get_model(x_train_1D[0].shape)
model_seq.summary()

"""
 I was getting warnings such as "``WARNING:tensorflow:Unresolved object in checkpoint: (root).optimizer.beta_1"
 solution found : https://gist.github.com/yoshihikoueno/4ff0694339f88d579bb3d9b07e609122
 
The root problem is that Adam.__init__ will initialize variables with python float objects which will not be tracked 
by tensorflow. We need to let them tracked and make them appear in Adam._checkpoint_dependencies in order to load 
weights without actually calling the optimizer itself. By converting Python float to tf.Variable, they will be 
tracked because tf.Variable is a subclass of ``trackable.Trackable``. """
adam = tf.keras.optimizers.Adam(
    learning_rate=tf.Variable(0.001),
    beta_1=tf.Variable(0.9),
    beta_2=tf.Variable(0.999),
    epsilon=tf.Variable(1e-7),
)
adam.iterations
adam.decay = tf.Variable(0.0)


def compile_model(model):
    model.compile(optimizer=adam,
                  # usually use 'adam' but because of warning above I had to initialize variables so that's why
                  # created my own adam
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


compile_model(model_seq)


def get_callbacks():
    checkpoint = ModelCheckpoint(filepath='checkpoints_best_only',
                                 save_weights_only=True,
                                 save_best_only=True,
                                 save_freq='epoch',
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1)
    early_stop = EarlyStopping(mode='min', patience=10, verbose=1, monitor='loss')
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10)
    return checkpoint, early_stop, learning_rate_reduction


callbacks = get_callbacks()


def train_model(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=30,
                        validation_split=0.15,
                        batch_size=128,
                        callbacks=callbacks,
                        verbose=1)
    return history


history = train_model(model_seq, x_train_1D, y_train)

# Plot the learning curves

frame = pd.DataFrame(history.history)

print(frame.columns)

# Plot the epoch vs accuracy graph

# use sharey or sharex=True if you want to use the same axes for plots
fig, (axes1, axes2) = plt.subplots(1, 2)
axes1.set_ylim(0, 2.2)
axes1.plot(history.history['loss'])
axes1.plot(history.history['val_loss'])
axes1.set_title('MLP Loss')
axes1.set_xlabel('Epochs')
axes2.plot(history.history['accuracy'])
axes2.plot(history.history['val_accuracy'])
axes2.set_title('MLP Accuracy')
axes2.legend(['Training', 'Validation'], loc='lower right')
axes2.set_xlabel('Epochs')
plt.tight_layout()
plt.show()


# plt.waitforbuttonpress()


# Function to evaluate the model
def get_test_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=1)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))
    return test_loss, test_acc


# Create new model with the saved weights
def get_model_best_epoch(model, x_test, y_test):
    model.load_weights('checkpoints_best_only')
    compile_model(model)
    # re-evaluate the model
    best_epoch_test_loss, best_epoch_test_acc = get_test_accuracy(model, x_test, y_test)
    print("Model with best epoch weights, Accuracy: {:5.2f}% , Loss: {:5.2f}".format(100 * best_epoch_test_acc,
                                                                                     best_epoch_test_loss))
    model_best_epoch = model
    return model_best_epoch, best_epoch_test_loss, best_epoch_test_acc


# Model using test data with the best epoch weights based on training, Accuracy:
model_best_epoch, best_epoch_test_loss, best_epoch_test_acc = get_model_best_epoch(get_model(x_test_1D[0].shape),
                                                                                   x_test_1D, y_test)

test_loss, test_accuracy = get_test_accuracy(model_best_epoch, x_test_1D, y_test)

print('> Test accuracy: %.3f %%' % (test_accuracy * 100.0))
print('> Test loss: %.3f' % test_loss)

'''
CNN neural network classifier

CNN classifier model using the Sequential API. 
Use the Conv2D, MaxPool2D, BatchNormalization, 
Flatten, Dense and Dropout layers. The final layer should again have a 10-way softmax output.

Aim to beat the MLP model performance with fewer parameters! '''
# go back to 2D data (conv2d expects 4 dimensions)

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

print('shape of single image:', x_train[0].shape)
print(y_train[0])

# recall you use padding to add columns & rows of zeroes to keep spatial sizes constant after convolution, this might
# improve performance as it retains the information at the borders
'''Great tips on how to build a model: 
https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper

Batch Normalization:- Generally the normalized input after passing through various adjustments in intermediate 
layers becomes too big or too small while 
it reaches far away layers which causes a problem of internal co-variate shift which impacts learning to solve this 
we add a batch normalization layer to standardize (mean centering and variance scaling) the input given to the later 
layers. This layer must generally be placed in the architecture after passing it through the layer containing 
activation function and before the Dropout layer(if any) . An exception is for the sigmoid activation function 
wherein you need to place the batch normalization layer before the activation to ensure that the values lie in linear 
region of sigmoid before the function is applied.
 
classic networks:  
Conv-Pool-Conv-Pool or Conv-Conv-Pool-Conv-Conv-Pool 
Number of channels 32–64–128 or 32–32-64–64 
##ADDING the 2nd conv-pool layers increased accuracy about 8%!!!############
Keep adding layers until you over-fit. As once we achieved a considerable accuracy in our validation set we can use 
regularization components like l1/l2 regularization, dropout, batch norm, data augmentation etc. to reduce over-fitting'''


def get_CNN_model(shape, rate, wd):
    model = Sequential([Conv2D(32, (3, 3), padding='same', activation='relu',
                               kernel_initializer='he_uniform', input_shape=shape, name='Conv2D_layer_1'),
                        # default data_format='channels_last'
                        MaxPooling2D((3, 3), name='MaxPool_layer_2'),
                        Conv2D(64, (3, 3), padding='same', activation='relu'),
                        MaxPooling2D((3, 3,)),
                        BatchNormalization(),
                        Flatten(name='Flatten_layer_3'),
                        Dense(100, kernel_regularizer=regularizers.l2(wd), activation='relu',
                              kernel_initializer='he_uniform', name='Dense_layer_4'),
                        BatchNormalization(),
                        Dropout(rate),
                        #Dense(120, activation='relu'), These didn't improve accuracy, comment out
                        #Dense(84, activation='relu'),
                        Dense(10, activation='softmax', name='Dense_layer_5')
                        ])
    return model


CNN_model = get_CNN_model(x_train[0].shape, 0.3, 0.001)
print(CNN_model.summary())
print(x_train[0].shape)

compile_model(CNN_model)

CNN_history = train_model(CNN_model, x_train, y_train)

# Plot the learning curves

frame2 = pd.DataFrame(CNN_history.history)
print(frame2.columns)

# Plot the epoch vs accuracy graph

plt, (axes11, axes12) = plt.subplots(1, 2,
                                     sharey=True)
# use sharey or sharex=True if you want to use the same axes for plots
axes11.set_ylim(0, 2.2)
axes11.plot(CNN_history.history['loss'])
axes11.plot(CNN_history.history['val_loss'])
axes11.set_title('CNN Loss')
axes11.set_xlabel('Epochs')
axes12.plot(CNN_history.history['accuracy'])
axes12.plot(CNN_history.history['val_accuracy'])
axes12.set_title('CNN Accuracy')
axes12.legend(['Training', 'Validation'], loc='lower right')
axes12.set_xlabel('Epochs')
plt.tight_layout()
plt.show()
plt.waitforbuttonpress()

CNN_model_best = get_model_best_epoch(get_CNN_model(x_test[0].shape, 0.3, 0.001), x_test, y_test)

# Model with best epoch weights
model_best_epoch, best_epoch_test_loss, best_epoch_test_acc = get_model_best_epoch(
    get_CNN_model(x_train[0].shape, 0.3, 0.001), x_test, y_test)

test_loss, test_accuracy = get_test_accuracy(model_best_epoch, x_test, y_test)

print('> CNN Test accuracy: %.3f %%' % (test_accuracy * 100.0))
print('> CNN Test loss: %.3f' % test_loss)
