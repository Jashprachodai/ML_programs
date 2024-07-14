# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:30:22 2024

@author: 91961
"""

import os
import pandas as pd
import cv2  # OpenCV for image processing

# Directory containing images
image_dir = r'C:\Users\91961\OneDrive\Desktop\gen ai DR\data\data\testing_data\dogs'  # Replace with your image directory

# List of image file names
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Initialize a list to store the image data and labels
data = []

# Loop through each image file
for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale (1 channel)
    
    # Resize image to a fixed size (e.g., 28x28)
    image = cv2.resize(image, (28, 28))
    
    # Flatten the image to a 1D array
    image_flat = image.flatten()
    
    # Assuming the label is part of the file name (e.g., 'cat_01.jpg' has label 'cat')
    label = image_file.split('_')[0]
    
    # Append the label and image data to the list
    data.append([label] + image_flat.tolist())

# Create a DataFrame
columns = ['label'] + [f'pixel{i}' for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
df.to_excel('images.xlsx', index=False)