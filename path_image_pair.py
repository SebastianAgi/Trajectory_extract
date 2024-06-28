import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import time
from datetime import datetime
import re

def extract_timestamps(file_path):
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comment lines
            parts = line.split()
            if parts:
                timestamps.append(float(parts[0]))  # Extract the timestamp as an integer
    
    # Create the desired list format
    result = [(i, timestamp) for i, timestamp in enumerate(timestamps)]
    
    return result

def find_closest_timestamp(timestamp, timestamp_list):
    # Find the closest timestamp in the list
    closest_timestamp = min(timestamp_list, key=lambda x: abs(x[1] - timestamp))
    return closest_timestamp


# Example usage
coordinates_path = '/home/sebastian/Documents/ANYmal_data/output_fastlio2_extracted/path_coordinates.txt'
rgb_timestamps_path = '/home/sebastian/Documents/ANYmal_data/anymal_real_message_grass02_extracted/rgb_timestamps.txt'
timestamp_list = extract_timestamps(coordinates_path)
for i in range(0, len(timestamp_list)):
    timestamp_list[i] = (i, timestamp_list[i][1]/1000000000)
# print(timestamp_list[0:15])
rgb_timestamp_list = extract_timestamps(rgb_timestamps_path)
# print(rgb_timestamp_list[0:15])

closest_timestamp = []
# Find the closest timestamp in the list
for i in range(0, len(timestamp_list)):
    closest_timestamp.append(find_closest_timestamp(timestamp_list[i][1], rgb_timestamp_list))

# make a variable (chosen_images) be a list of the first value in the tuple of closest_timestamp
chosen_images = [x[0] for x in closest_timestamp]

# put the chosen images in a new directory
# Directory containing your images
image_dir = '/home/sebastian/Documents/ANYmal_data/chosen_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

# drop the images based on index that are not in chosen_images
for i in range(0, len(image_files)):
    if i not in chosen_images:
        os.remove(os.path.join(image_dir, image_files[i]))