import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# 模型准确度很差
# 模型准确度很差
# 模型准确度很差

from resnets_utils import *

import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# %matplotlib inline


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well.

    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # layer group1 32*32*32
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # layer group2 16*16*64
    X = ZeroPadding2D((2, 2))(X)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (5, 5), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # layer group3 8*8*128
    X = ZeroPadding2D((1, 1))(X)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    # layer group4 8*8*64
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(64, (1, 1), strides=(1, 1), name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    # layer group5 4*4*32
    X = ZeroPadding2D((1, 1))(X)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv5')(X)
    X = BatchNormalization(axis=3, name='bn5')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool5')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(128, activation='sigmoid', name='fc1')(X)
    X = Dense(32, activation='sigmoid', name='fc2')(X)
    X = Dense(1, activation='sigmoid', name='fc3')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    ### END CODE HERE ###

    return model


def predicatImage(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(happyModel.predict(x))


### START CODE HERE ### (1 line)
happyModel = HappyModel((64, 64, 3))
### END CODE HERE ###
### START CODE HERE ### (1 line)
happyModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
### END CODE HERE ###
### START CODE HERE ### (1 line)
happyModel.fit(x=X_train, y=Y_train, epochs=1, batch_size=1)
### END CODE HERE ###
### START CODE HERE ### (1 line)
preds = happyModel.evaluate(x=X_test, y=Y_test)
### END CODE HERE ###
print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

# Loss = 0.0385172860821
# Test Accuracy = 0.993333333333
# [[ 0.99550456]]

### START CODE HERE ###
predicatImage('happy_house/images/my_image.jpg')
predicatImage('happy_house/images/my_image_3.jpg')
predicatImage('happy_house/images/my_image_4.jpg')
predicatImage('happy_house/images/old_man.jpg')
### END CODE HERE ###

happyModel.summary()

# plot_model(happyModel, to_file='HappyModel.png')
# SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
