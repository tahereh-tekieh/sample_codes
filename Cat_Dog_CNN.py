#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:30:42 2021
CNN (Conv. Neural Network)to identify dog and cats
@author: Tahereh Tekieh
"""
import numpy as np
import tensorflow as tf

# Part 1 - Data Preprocessing

# Image augmentation to train set
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# trscaling the test images
test_datagen =  tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, activation = 'relu',input_shape = [64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 120, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))


# Part 3 - Training the CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(training_set,validation_data=test_set, epochs = 25 )


# Part 4 - Predict an image

test_image = tf.keras.preprocessing.image.load_img('/Users/ttek2642/Downloads/Section 40 - Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_2.jpg',
                                                 target_size = (64, 64))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
