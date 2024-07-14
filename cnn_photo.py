# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:00:52 2024

@author: 91961
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define directories
train_dir = r'C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\image_dataset\train'
test_dir = r'C:\Users\91961\OneDrive\Desktop\gen ai DR\datasets\image_dataset\test'

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.55,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of tensor image data with real-time data augmentation
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=20,
                                                    color_mode = 'grayscale',
                                                    class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(150, 150),
                                                  batch_size=10,
                                                  color_mode = 'grayscale',
                                                  class_mode='binary')

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=(train_generator.samples // train_generator.batch_size),
          epochs=10,
          validation_data=test_generator,
          validation_steps=(test_generator.samples // test_generator.batch_size)
          )

# Save the model
model.save('person_Class.keras')
