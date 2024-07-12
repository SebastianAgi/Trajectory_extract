#use matplotlib to plot a series of x, y, z coordinates from a text file

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Initialize lists to store coordinates and orientations
coordinates = []
orientations = []
directions = []
images = []  # List of images

##########
traslate_odom_to_camera = True
##########

# Function to multiply two quaternions
def quaternion_multiply(q1, q2):
    q = R.from_quat(q1)
    p = R.from_quat(q2)
    
    # Calculate the product
    r = q * p

    return r.as_quat()

# Load the coordinates and orientations
coordinates_path = '/home/sebastian/Documents/code/Trajectory_extract/odom_data_halfsecond.txt'
with open(coordinates_path, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comment lines
        parts = line.split()
        if parts:
            coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
            orientations.append(np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])])) #qx, qy, qz, qw

#difference between odometry frame and camera frame
translation = [-0.739, -0.056, -0.205] #x, y, z
# rotation = [0.466, -0.469, -0.533, 0.528] #quaternion
rotation = [-0.469, -0.533, 0.528, 0.466] #quaternion

if traslate_odom_to_camera:
    #transform coordinate-orientation pairs to camera frame
    for i in range(len(coordinates)):
        #transform coordinates
        coord = coordinates[i]
        x = coord[0] + translation[0]
        y = coord[1] + translation[1]
        z = coord[2] + translation[2]
        coordinates[i] = (x, y, z)

        #transform orientations
        orientations[i] = quaternion_multiply(rotation, orientations[i])

# Plot coordinate data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Path coordinates')
ax.set_zlim(-5, 1)

#limit the number of points to plot
coordinates = coordinates[:200]
orientations = orientations[:200]

# Plot the coordinates
x_coords, y_coords, z_coords = zip(*coordinates)
ax.scatter(x_coords, y_coords, z_coords, color='b', label='Coordinates')

# Plot the rotation vectors
for coord, quat in zip(coordinates, orientations):
    rot_vec = R.from_quat([quat[0], quat[1], quat[2], quat[3]])
    # quiver takes x, y, z, dx, dy, dz
    ax.quiver(coord[0], coord[1], coord[2], rot_vec.as_rotvec()[0], rot_vec.as_rotvec()[1], rot_vec.as_rotvec()[2], color='r', length=0.5, normalize=False)
    print(coord[0], coord[1], coord[2], rot_vec.as_rotvec()[0], rot_vec.as_rotvec()[1], rot_vec.as_rotvec()[2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()