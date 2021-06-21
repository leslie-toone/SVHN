# create a neural network that classifies real-world images digits into one of ten classes.
'''The SVHN dataset (http://ufldl.stanford.edu/housenumbers/) is an image dataset of over 600,000 digit images in all,
and is a harder dataset than MNIST as the numbers appear in the context of natural scene images.
SVHN is obtained from house numbers in Google Street View images.

Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu and A. Y. Ng. "Reading Digits in Natural Images with Unsupervised
Feature Learning". NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.
Your goal is to develop an end-to-end workflow for building, training, validating, evaluating and saving a neural
 network that classifies a real-world image into one of ten classes.
'''

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from numpy import save

# Run this cell to load the dataset (Matlab format)

train = loadmat('data/train_32x32.mat')
test = loadmat('data/test_32x32.mat')

# INSPECT AND PREPROCESS THE DATA
# Extract the training and testing images and labels separately from the train and test dictionaries loaded for you.
print(train.keys())
x_train = train['X']/255#standardize pixel quality
y_train = train['y']
x_test = test['X']/255
y_test = test['y']
# notice the number of samples is last, so we'll need to transpose data

print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# converting it from (width, height, channels, size) -> (size, width, height, channels)
# Transpose the image arrays
x_train, y_train = x_train.transpose((3, 0, 1, 2)), y_train[:, 0]
x_test, y_test = x_test.transpose((3, 0, 1, 2)), y_test[:, 0]
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))


# Select a random sample of images and corresponding labels and display them in a figure.
# create grid of 3x3 images to visualize the data set
num_train_images = x_train.shape[0]
#select 9 random images
random_num = np.random.choice(num_train_images,9)
print('Random Sample ', random_num)

random_train_x = x_train[random_num, ...]
random_train_y = y_train[random_num, ...]
print('Shape of Random Sample:', random_train_x.shape,'Shape of Random Labels: ', random_train_y.shape)

def plot_images(img, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        #axes.flat=flattens the array so ([[2,3],[4,5]]) reads in as 2,3,4,5
        # For each iteration it would yield the next axes from that array, such that you may easily plot to all axes in a single loop.
        ax.imshow(img[i], cmap=plt.get_cmap('gray'))#if you don't set the color map, you'll get greenish images in grayscale
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_title(labels[i])
# Plot some training set images
plot_images(random_train_x, random_train_y, 3, 3)
plt.show()
#not random images
plot_images(x_train, y_train, 3, 3)
plt.show()
# Convert the training and test images to grayscale (which greatly reduces the amount of data we will have to process)

def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

# Converting to Float for numpy computation

x_train_gray = rgb2gray(x_train).astype(np.float32)
x_test_gray = rgb2gray(x_test).astype(np.float32)

print("Gray Training Set Shape", x_train_gray.shape)
print("Gray Test Set Shape", x_test_gray.shape)

random_train_x_gray = np.array(x_train_gray)[random_num]
random_train_y_gray = y_train[random_num]

# Plot some training set images, set the colormap parameter to gray
plot_images(random_train_x_gray, random_train_y_gray, 3, 3)

plt.show()
#delete rgb sets to free up RAM
del x_train, x_test
#rename
x_train=x_train_gray
x_test=x_test_gray
#save needed data sets
save('data/y_train_SVHN.npy', y_train)
save('data/y_test_SVHN.npy', y_test)
save('data/x_train_SVHN.npy', x_train)
save('data/x_test_SVHN.npy', x_test)
