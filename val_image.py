# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:22:47 2024

@author: 91961
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('person_Class.keras')

def preprocess_image(image_path):
    # Load the image with the target size
    img = load_img(image_path, color_mode = 'grayscale',target_size=(150, 150))
    # Convert the image to an array
    img_array = img_to_array(img)
    # Rescale the image (normalize)
    img_array /= 255.0
    # Expand dimensions to match the model's input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    # Make a prediction
    prediction = model.predict(img_array)
    # The output is a probability. Convert it to class label.
    if prediction[0][0] > 0.5:
        print("This is Jaswanth")
    else:
        print("This is Vamsi")

# Example usage
image_path = r"C:\Users\91961\Downloads\jash.jpg"
predict_image(image_path)
