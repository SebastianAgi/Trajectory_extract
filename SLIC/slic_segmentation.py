#file to run SLIC segmentation on an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fast_slic import Slic
import time
import json
from collections import defaultdict

class SLIC():
    def __init__(self, path):
        self.image_path = path
        self.num_superpixels = 500  # Adjust the number of superpixels
        self.compactness = 10.0     # Adjust the compactness factor

    def SLIC_segmentation(self, pixels):
        # Load your image
        image = cv2.imread(self.image_path)
        only_img = image[20:-20, 30:-30]

        # Convert BGR image to RGB for matplotlib
        image_rgb = cv2.cvtColor(only_img, cv2.COLOR_BGR2RGB)

        # Create Slic object
        slic = Slic(num_components=self.num_superpixels, compactness=self.compactness)

        # Perform segmentation
        segmented_image = slic.iterate(image_rgb)

        # Assuming pixels is a list of (x, y) tuples or a 2D array where each row is an (x, y) pair
        pixels_array = np.array(pixels)

        # Extract the x and y coordinates
        x_coords = pixels_array[:, 0]
        y_coords = pixels_array[:, 1]

        # Use advanced indexing to get the segment values at the given (x, y) coordinates
        segment_values = segmented_image[x_coords, y_coords]

        # Create a dictionary to hold lists of pixel coordinates for each segment
        segment_dict = defaultdict(list)

        # Populate the dictionary with pixel coordinates grouped by their segment
        for i in range(len(segment_values)):
            segment = segment_values[i]
            pixel = (x_coords[i], y_coords[i])
            segment_dict[segment].append(pixel)

        return segment_dict, segmented_image
    
    def make_masks_smaller(segment_values, wanted_size):
        # choose segment values to make a binary mask out of the image with the wanted size for choosing the DINO feature output pixels
        pass
    
def run_SLIC_segmentation():
    """Run SLIC on an image and visualize the segmented image"""

    ##############################################
    # This should all be coming from a config file
    ##############################################
    #read json file
    with open('/home/sebastian/Documents/code/Trajectory_extract/all_points.json', 'r') as f:
        path = json.load(f)
    pixels = path[1318]
    sam_mask = cv2.imread('/home/sebastian/Documents/ANYmal_data/OPS_grass/Masked_data/odom_data_masked/masks/mask_1318.png', cv2.IMREAD_GRAYSCALE)
    img_path = '/home/sebastian/Documents/ANYmal_data/OPS_grass/odom_chosen_images_2/003325.png'
    ##############################################

    slic = SLIC(img_path)

    segments, segmented_image = slic.SLIC_segmentation(pixels)

    # Optionally, visualize the segmented image
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image, cmap='inferno')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_SLIC_segmentation()