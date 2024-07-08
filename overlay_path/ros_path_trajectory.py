#use matplotlib to plot a series of x, y, z coordinates from a text file

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from nav_msgs.msg import Odometry as odom

# Initialize lists to store coordinates and orientations
coordinates = []
orientations = []
directions = []

##########
traslate_odom_to_camera = True
##########

# Initialize the ROS node
rospy.init_node('trajectory_publisher', anonymous=True)
# Initialize the publisher
pub = rospy.Publisher('/trajectory', odom, queue_size=10)

# Function to multiply two quaternions
def quaternion_multiply(q1, q2):
    # Calculate the product
    r = R.from_quat(q1) * R.from_quat(q2)
    # from_quat([qx qy qz qu])
    return r.as_quat()

# Load the coordinates and orientations
coordinates_path = '/home/sebastian/Documents/code/Trajectory_extract/odom_data_halfsecond.txt'
T_odom_list = []
with open(coordinates_path, 'r') as file:
    for line in file:
        if line.startswith('#'):
            continue  # Skip comment lines
        parts = line.split()
        if parts:
            coordinates.append(np.array([float(parts[2]), float(parts[3]), float(parts[4])]))
            # print(coordinates[-1])
            orientations.append(np.array([float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])])) #qx, qy, qz, qw
            # print(orientations[-1])

            T_odom = np.eye(4, 4)
            T_odom[:3, :3] = R.from_quat(orientations[-1]).as_matrix()[:3, :3]
            T_odom[:3, 3] = coordinates[-1]
            T_odom_list.append(T_odom)

#difference between odometry frame and camera frame
translation = [-0.739, -0.056, -0.205] #x, y, z
rotation = [0.466, -0.469, -0.533, 0.528] #quaternion
T_imu_camera = np.eye(4, 4)
T_imu_camera[:3, :3] = R.from_quat(rotation).as_matrix()[:3, :3]
T_imu_camera[:3, 3] = translation

# rotation = [-0.469, -0.533, 0.528, 0.466] #quaternion

if traslate_odom_to_camera:
    #transform coordinate-orientation pairs to camera frame
    for i in range(len(coordinates)):
        T_world_camera = np.linalg.inv(T_imu_camera) @ T_odom_list[i] @ T_imu_camera

        coordinates[i] = T_world_camera[:3, 3]
        orientations[i] = R.from_matrix(T_world_camera[:3, :3]).as_quat()

# T1 = [R1 t1-> odometry from world to imu
#       0   1]

# T2 = [R2 t2 -> extrinsics from imu to camera
#       0  1]

# T1 * T2 = [R1*R2 R1*t2 + t1] -> odometry from world to camera
#             0             1]

# T = T1 @ T2

#create a list of odometry messages from the coord and orientation lists
for i in range(len(coordinates)):
    # Create a new odometry message
    odom_msg = odom()
    # Set the header
    odom_msg.header.stamp = rospy.Time.now()
    odom_msg.header.frame_id = "odom"

    # Set the position
    odom_msg.pose.pose.position.x = coordinates[i][0]
    odom_msg.pose.pose.position.y = coordinates[i][1]
    odom_msg.pose.pose.position.z = coordinates[i][2]
    # Set the orientation
    odom_msg.pose.pose.orientation.x = orientations[i][0]
    odom_msg.pose.pose.orientation.y = orientations[i][1]
    odom_msg.pose.pose.orientation.z = orientations[i][2]
    odom_msg.pose.pose.orientation.w = orientations[i][3]
    # Append the message to the list
    directions.append(odom_msg)

# publish data to ros topic
for i in range(len(coordinates)):
    # Publish the message
    pub.publish(directions[i])
    # Sleep for 0.1 seconds
    rospy.sleep(0.01)
    print(f"Published message {i+1}/{len(coordinates)}", end='\r')