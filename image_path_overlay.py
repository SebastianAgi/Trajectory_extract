import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Example data structure
images = []  # List of images
coordinates = []  # List of (x, y, z) tuples
orientations = []  # List of (qw, qx, qy, qz) tuples

#load the images
image_dir = '/home/sebastian/Documents/ANYmal_data/chosen_images'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()
for i in range(0, len(image_files)):
    image = cv2.imread(os.path.join(image_dir, image_files[i]))
    # print(f"Image {i}: {image_files[i]}")
    images.append(image)

#load the coordinates and orientations
coordinates_path = '/home/sebastian/Documents/ANYmal_data/output_fastlio2_extracted/path_coordinates.txt'
with open(coordinates_path, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comment lines
        parts = line.split()
        if parts:
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
            orientations.append((float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])))

#plot coordniate data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Path coordinates')
x, y, z = zip(*coordinates)
ax.plot(x, y, -z, marker='o')
plt.show()


# Camera intrinsic parameters from Azure Kinect
fx = 607.9638061523438
fy = 607.9390869140625
cx = 638.83984375
cy = 367.0916748046875

# Camera intrinsic parameters (example values)
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion (qw, qx, qy, qz) to a rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def project_points(points, rotation_matrix, translation_vector, camera_matrix):
    """Project 3D points to 2D image plane."""
    points_3d = np.array(points).reshape(-1, 3)
    points_2d = cv2.projectPoints(points_3d, rotation_matrix, translation_vector, camera_matrix, distCoeffs=None)[0]
    return points_2d.reshape(-1, 2)

def draw_path_on_image(image, points_2d):
    """Draw points and lines on the image."""
    for i in range(len(points_2d) - 1):
        pt1 = tuple(map(int, points_2d[i]))
        pt2 = tuple(map(int, points_2d[i + 1]))
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(image, pt1, 5, (0, 0, 255), -1)
    # Draw the last point
    cv2.circle(image, tuple(map(int, points_2d[-1])), 5, (0, 0, 255), -1)
    return image

for i in range(len(images)):
    if i + 4 >= len(images):
        break
    current_image = images[i].copy()
    points_3d = coordinates[i:i + 5]
    orientation = orientations[i]
    rotation_matrix = quaternion_to_rotation_matrix(orientation)
    translation_vector = np.array(coordinates[i]).reshape(3, 1)
    points_2d = project_points(points_3d, rotation_matrix, translation_vector, camera_matrix)
    image_with_path = draw_path_on_image(current_image, points_2d)
    cv2.imshow(f'Image with path {i}', image_with_path)
    cv2.waitKey(0)