import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw
import os
import cv2

# Initialize lists to store coordinates and orientations
coordinates = []
orientations = []
directions = []
images = []  # List of images

# Load the coordinates and orientations
coordinates_path = '/home/sebastian/Documents/ANYmal_data/output_fastlio2_extracted/path_coordinates.txt'
with open(coordinates_path, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comment lines
        parts = line.split()
        if parts:
            coordinates.append((float(parts[1]), -float(parts[2]), -float(parts[3])))
            orientations.append((float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])))

def calculate_directions(data):
    for i in range(len(data) - 1):
        x1, y1, z1 = data[i]
        x2, y2, z2 = data[i+1]
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        if length > 0:  # Avoid division by zero
            directions.append((dx/length, dy/length, dz/length))
        else:
            directions.append((0, 0, 0))  # Same point, no direction
    return directions

# Calculate the directions between consecutive coordinates
directions = calculate_directions(coordinates)

def convert_to_2d(coords):
    """Assuming a simple orthographic projection ignoring z-component."""
    return [(x, y) for x, y, z in coords]

def scale_to_image(coords, image_size):
    """Scale coordinates to fit a given image size."""
    min_x = min(coords, key=lambda t: t[0])[0]
    min_y = min(coords, key=lambda t: t[1])[1]
    max_x = max(coords, key=lambda t: t[0])[0]
    max_y = max(coords, key=lambda t: t[1])[1]

    scale_x = image_size[0] / (max_x - min_x)
    scale_y = image_size[1] / (max_y - min_y)

    return [(int((x - min_x) * scale_x), int((y - min_y) * scale_y)) for x, y in coords]

def overlay_path_on_cv2_image(cv2_image, points, start_index):
    """Overlay the path from the start_index on the cv2 image."""
    for i in range(start_index, len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        # Draw line with color (red) in BGR format, thickness 3
        cv2.line(cv2_image, start_point, end_point, (0, 0, 255), thickness=3)
    
    # Display the image - this can be replaced with cv2.imwrite if you wish to save the image
    cv2.imshow('Path Overlay', cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#load the images
image_dir = '/home/sebastian/Documents/ANYmal_data/chosen_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
for i in range(0, len(image_files)):
    image = cv2.imread(os.path.join(image_dir, image_files[i]))
    # print(f"Image {i}: {image_files[i]}")
    images.append(image)
image_size = (1280, 720)  # Example size, replace with your actual image dimensions

# Convert coordinates to 2D and scale them
coordinates_2d = convert_to_2d(coordinates)
scaled_points = scale_to_image(coordinates_2d, image_size)

# Process each image (for example, the first 10 images for demonstration)
for i in range(10):  # Adjust range as needed
    if i < len(images):
        overlay_path_on_cv2_image(images[i], scaled_points, i)
