"""
=================================================================================================================
CONVOLUTIONAL NEURAL NETWORKS MODELS.
Modelos de Redes Neuronales Convolucionales.
author : Gerardo Cano Perea.
date : February 26, 2021
=================================================================================================================
Step 1A. Convolution Transform for the Image.
Step 1B. Apply a ReLU Rectified Linear Unit.
Step 2. Max Pooling / Avoid in General Terms the Overfitting.
Step 3. Flattening Transform a Matrix in a Lineal Vector.
Step 4. Full Connection.
Additional.
A1. Activation Softmax. Used for Transform k-Values into Probability Values from 0 to 1
12. Optimization using Crossentropy.
"""

# Items for Convolutional Neural Network Model.
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Building the Neural Network.
classifier = Sequential()
# Step 1. Convolution Layer. Usually the first number of filters is 32. input_shape = [row, columns, color channels]
classifier.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2. Max Polling Layer.
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3. Flatten Layer.
classifier.add(Flatten())
# Step 4 Full Connection.
# Activation in Last Layer can be replace with a 'softmax' operation.
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
# Summary for the final convolutional neural network model.
classifier.summary()

# Compiling the Convolutional Model.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the Images for Train the Convolutional Network
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1. / 255)
training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size = (64, 64),
                                                     batch_size = 32,
                                                     class_mode = 'binary')
testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size = (64, 64),
                                                   batch_size = 32,
                                                   class_mode = 'binary')
# Training the Convolutional Model.
# Every epoch the model receive 8000 images for training and 2000 for validation.
# The model is correct but computational resources are not enough even in a Google Colab session.
classifier.fit_generator(training_dataset,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = testing_dataset,
                         validation_steps = 2000,
                         verbose = 1)
