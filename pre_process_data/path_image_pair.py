#This script is for creating synchornized image pairs from the camera and the odometry data

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import time
from datetime import datetime
import shutil

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
coordinates_path = '/home/sebastian/Documents/code/Trajectory_extract/odom_data.txt'
rgb_timestamps_path = '/home/sebastian/Documents/ANYmal_data/anymal_real_message_grass02_extracted/rgb_timestamps.txt'
odom_data = extract_data(coordinates_path)

# print(odom_data[0:5])
# print('\n')

# odom_data = extract_every_halfsecond(odom_data)
# # # timestamp_list = [(i, t) for i, t in timestamp_list]  # Normalize the timestamps

# #save the new odom_data to a new file
# with open('/home/sebastian/code/Trajectory_extraction/odom_data_halfsecond.txt', 'w') as file:
#     for line in odom_data:
#         file.write(' '.join([str(x) for x in line]) + '\n')

# print(odom_data[0:5])
# print(len(odom_data))
# print('\n')

image_data = extract_data(rgb_timestamps_path)
# print('camera_data', image_data[0:15])

closest_image_timestamp = []
prev_img_index = None
overlap_count = 0
overlap_images = {}
# Find the closest timestamp in the list
for t in odom_data:
    closest_index, closest_value = find_closest_timestamp(t[1], image_data)
    if prev_img_index == closest_index:
        overlap_count += 1
        #amek a dictionary of the overlapping images with image index as key and number of occurances as value
        if closest_index in overlap_images:
            overlap_images[closest_index] += 1
        else:
            overlap_images[closest_index] = 1

    prev_img_index = closest_index
    closest_image_timestamp.append((closest_index, closest_value))


print('overlap_count: ', overlap_count)
print('overlap_images: \n', overlap_images)
# print(len(closest_image_timestamp))
# print(closest_image_timestamp[0:15])

# make a variable (chosen_images) be a list of the first value in the tuple of closest_timestamp
chosen_images = [x[0] for x in closest_image_timestamp]

# print(len(chosen_images))
# print(chosen_images[0:15])
# print(chosen_images[-15:])

# put the chosen images in a new directory
# Directory containing your images
image_dir = '/home/sebastian/Documents/ANYmal_data/odom_chosen_images_2'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

# drop the images based on index that are not in chosen_images
for i in range(0, len(image_files)):
    if i not in chosen_images:
        os.remove(os.path.join(image_dir, image_files[i]))
    #add copy of image if it is in the overlap_images dictionary
    if i in overlap_images.keys():
        # Construct full file path
        original_file = os.path.join(image_dir, image_files[i])
        
        # Construct new file path with '_copy' suffix
        new_file = os.path.join(image_dir, f"{image_files[i]}_copy")
        
        # Make a copy of the file
        shutil.copy2(original_file, new_file)