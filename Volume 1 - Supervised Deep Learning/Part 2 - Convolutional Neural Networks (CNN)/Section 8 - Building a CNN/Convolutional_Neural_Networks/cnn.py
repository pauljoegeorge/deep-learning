# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential # for neural network
from keras.layers import Conv2D     # for convolution
from keras.layers import MaxPooling2D # to peform pooling
from keras.layers import Flatten # to flatten the pooled_features
from keras.layers import Dense  #connecting hidden layer and output layer

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
"""
32 - number of features detectors we are going to apply on the image (result in number 32 feature maps)
(3,3) - height and weight of feature detector ie; 3x3 matrix
input_shape - (64x64) matrix and 3 means colored image RGB -this reduced image size to 64x64
(64,64,3) - 64x64 RGB pictures
"""
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
"""
- reducing the size of feature map
- to reduce the no of nodes in fully connected layer
- we are using stride of size 2
"""
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
"""
128 - no of input nodes  ( value choosed in general, it can be any value. recommended to be value > 100)
"""
classifier.add(Dense(units = 128, activation = 'relu')) #fully connected layer
classifier.add(Dense(units = 1, activation = 'sigmoid')) #output layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)