#use matplotlib to plot a series of x, y, z coordinates from a text file

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as R

# Initialize lists to store coordinates and orientations
coordinates = []
orientations = []
directions = []
images = []  # List of images

# Load the coordinates and orientations
coordinates_path = '/home/sebastian/code/Trajectory_extraction/odom_data.txt'
with open(coordinates_path, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comment lines
        parts = line.split()
        if parts:
            coordinates.append((float(parts[1]), float(parts[2]), float(parts[3])))
            orientations.append((float(parts[7]), float(parts[4]), float(parts[5]), float(parts[6])))

# Plot coordinate data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Path coordinates')
ax.set_zlim(-5, 1)

#limit the number of points to plot
# coordinates = coordinates[:100]
# orientations = orientations[:100]

# Plot the coordinates
x_coords, y_coords, z_coords = zip(*coordinates)
ax.scatter(x_coords, y_coords, z_coords, color='b', label='Coordinates')

# Plot the rotation vectors
for coord, quat in zip(coordinates, orientations):
    rot_vec = R.Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
    ax.quiver(coord[0], coord[1], coord[2], rot_vec.as_rotvec()[0], rot_vec.as_rotvec()[1], rot_vec.as_rotvec()[2], color='r', length=1.0, normalize=True)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()