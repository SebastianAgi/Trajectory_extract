#use matplotlib to plot a series of x, y, z coordinates from a text file

import matplotlib.pyplot as plt
import numpy as np

# Load the data from the text file, the data comes in the format: timestamp x y z qx qy qz qw
data = np.loadtxt('/home/sebastian/Documents/ANYmal_data/output_fastlio2_extracted/path_coordinates.txt', comments='#', delimiter=' ', unpack=True)

# Extract the x, y, and z coordinates
x = data[1]
y = data[2]
z = data[3]

# Plot the x, y, and z coordinates on same sized axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(-x, y, -z, label='Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#set z axis to be from -60 to 20
ax.set_xlim(-20, 60)
ax.set_ylim(-60, 20)
ax.set_zlim(-60, 20)

plt.show()
