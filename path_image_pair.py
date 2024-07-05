import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import time
from datetime import datetime
import re

def extract_data(file_path):
    data = []
    line_number = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip comment lines
            parts = line.split()
            if parts:
                row = [line_number]
                for part in parts:
                    try:
                        row.append(float(part))
                    except ValueError:
                        continue  # Skip the element if conversion fails
                data.append(row)
            line_number += 1
    return data

def find_closest_timestamp(timestamp, odom_data):
    # Find the closest timestamp in the list and return the index and the timestamp
    closest = min(odom_data, key=lambda x: abs(x[1] - timestamp))
    return closest

def extract_every_halfsecond(data):
    # Extract every half second from the list
    result = []
    past_timestamp = 0
    for i in range(len(data)):
        
        if data[i][1] - past_timestamp > 0.5:
            result.append(data[i])
            past_timestamp = data[i][1]
    
    return result

# Example usage
coordinates_path = '/home/sebastian/Documents/ANYmal_data/output_fastlio2_extracted/odom_data.txt'
rgb_timestamps_path = '/home/sebastian/Documents/ANYmal_data/anymal_real_message_grass02_extracted/rgb_timestamps.txt'
odom_data = extract_data(coordinates_path)

print(odom_data[0:5])
print('\n')

odom_data = extract_every_halfsecond(odom_data)
# # timestamp_list = [(i, t) for i, t in timestamp_list]  # Normalize the timestamps

print(odom_data[0:5])
print(len(odom_data))
print('\n')

image_data = extract_data(rgb_timestamps_path)
print('camera_data', image_data[0:15])

closest_image_timestamp = []
# Find the closest timestamp in the list
for t in odom_data:
    closest_index, closest_value = find_closest_timestamp(t[1], image_data)
    closest_image_timestamp.append((closest_index, closest_value))

print(len(closest_image_timestamp))
print(closest_image_timestamp[0:15])
print('\n')

# make a variable (chosen_images) be a list of the first value in the tuple of closest_timestamp
chosen_images = [x[0] for x in closest_image_timestamp]

print(len(chosen_images))
print(chosen_images[0:15])
print(chosen_images[-15:])

# put the chosen images in a new directory
# Directory containing your images
image_dir = '/home/sebastian/Documents/ANYmal_data/odom_chosen_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

# # drop the images based on index that are not in chosen_images
for i in range(0, len(image_files)):
    if i not in chosen_images:
        os.remove(os.path.join(image_dir, image_files[i]))