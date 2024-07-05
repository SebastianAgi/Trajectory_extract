import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

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

# Load the images
image_dir = '/home/sebastian/Documents/ANYmal_data/chosen_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
for i in range(0, len(image_files)):
    image = cv2.imread(os.path.join(image_dir, image_files[i]))
    images.append(image)

# Camera intrinsic parameters
fx = 607.9638061523438
fy = 607.9390869140625
cx = 638.83984375
cy = 367.0916748046875

# Camera intrinsic matrix
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

def project_points(points_3d, camera_matrix, center_point):
    """ Project 3D points to 2D using the camera intrinsic matrix and center the first point. """
    points_2d = []
    for point in points_3d:
        x, y, z = point
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy
        points_2d.append([u, v])
    points_2d = np.array(points_2d)
    
    # Shift to center the first point at the bottom center of the image
    shift_u = center_point[0] - points_2d[0, 0]
    shift_v = center_point[1] - points_2d[0, 1]
    points_2d[:, 0] += shift_u
    points_2d[:, 1] += shift_v
    
    return points_2d

def overlay_path(image, points_2d):
    """ Overlay the path on the image. """
    for i in range(len(points_2d) - 1):
        pt1 = tuple(map(int, points_2d[i]))
        pt2 = tuple(map(int, points_2d[i + 1]))
        image = cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    return image

# Define the bottom center point of the image
image_height, image_width, _ = images[0].shape
center_point = (image_width // 2, image_height - 1)

for i in range(len(images) - 5):
    image = images[i]
    points_3d = coordinates[i:i+5]
    
    # Calculate direction to the next point
    direction = np.array(points_3d[1]) - np.array(points_3d[0])
    direction /= np.linalg.norm(direction)
    
    # Adjust subsequent points
    adjusted_points = [points_3d[0]]
    for j in range(1, len(points_3d)):
        adjusted_points.append(adjusted_points[-1] + direction * np.linalg.norm(np.array(points_3d[j]) - np.array(points_3d[j-1])))
    
    points_2d = project_points(adjusted_points, camera_matrix, center_point)
    image_with_path = overlay_path(image, points_2d)
    
    cv2.imshow('Image with path', image_with_path)
    cv2.waitKey(0)

cv2.destroyAllWindows()
