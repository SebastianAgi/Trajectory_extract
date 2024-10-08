import os
import cv2
from cv_bridge import CvBridge
import rosbag
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

class ExportData:
    def __init__(self, output_folder, bag_path) -> None:
        # Initialize the ROS bag path
        self.bag_path = bag_path
        # name a variable the last part of the bag path
        self.bag_name = os.path.basename(self.bag_path)[:-4] + '_extracted'
        # Initialize the folder to save the extracted data
        self.extraction_folder = output_folder
        # Create the output folder structure
        self.create_output_folders()
        # Initialize the list to store the extracted data
        self._rgb_data = []
        self._path_data = []
        self._odom_data = []
        self.image_count = 0

    def create_output_folders(self):
        os.makedirs(self.extraction_folder, exist_ok=True)
        os.makedirs(os.path.join(self.extraction_folder, self.bag_name), exist_ok=True)
        #now make a folder inside the bag_extracted folder to store the images
        # os.makedirs(os.path.join(self.extraction_folder, self.bag_name, 'images'), exist_ok=True)

    def save_rgb_text_data(self, image_folder, rgb_data):
        rgb_file_path = os.path.join(self.extraction_folder, self.bag_name, 'rgb_timestamps.txt')
        with open(rgb_file_path, 'w') as f:
            f.write("# color compressed images\n")
            f.write("# timestamp filename\n")
            for timestamp, filename in rgb_data:
                f.write(f"{timestamp} {os.path.join(filename)}\n")
    
    # Function to save the coordinate data from a PoseStamped message
    def save_coordinate_data(self, coordinate_data):
        path_file_path = os.path.join(self.extraction_folder, self.bag_name, 'path_coordinates.txt')
        with open(path_file_path, 'w') as f:
            f.write("# coordinate data\n")
            f.write("# timestamp x y z qx qy qz qw\n")
            for _, path in coordinate_data:
                x = path.poses[-1].pose.position.x
                y = path.poses[-1].pose.position.y
                z = path.poses[-1].pose.position.z
                qx = path.poses[-1].pose.orientation.x
                qy = path.poses[-1].pose.orientation.y
                qz = path.poses[-1].pose.orientation.z
                qw = path.poses[-1].pose.orientation.w
                f.write(f"{path.poses[-1].header.stamp} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
    
    def save_Odometry_data(self, odom_data):
        odom_file_path = os.path.join(self.extraction_folder, self.bag_name, 'odom_data.txt')
        with open(odom_file_path, 'w') as f:
            f.write("# Odometry data\n")
            f.write("# timestamp x y z qx qy qz qw\n")
            for timestamp, odom, in odom_data:
                x = odom.pose.pose.position.x
                y = odom.pose.pose.position.y
                z = odom.pose.pose.position.z
                qx = odom.pose.pose.orientation.x
                qy = odom.pose.pose.orientation.y
                qz = odom.pose.pose.orientation.z
                qw = odom.pose.pose.orientation.w
                f.write(f"{timestamp} {x} {y} {z} {qx} {qy} {qz} {qw}\n")
                # print(f"{timestamp} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

    def extract_bag(self):
        # Initialize data arrays or variables to store data
        # Initialize CvBridge for converting ROS Image messages to OpenCV images
        bridge = CvBridge()
        get_rgb_info = False
        # Open the ROS bag for reading
        with rosbag.Bag(self.bag_path, 'r') as bag:
            # Get a list of all topics in the bag file
            available_topics = bag.get_type_and_topic_info()[1].keys()
            print('available_topics:\n', available_topics)
            # Loop through the bag messages
            for topic, msg, t in bag.read_messages(topics=available_topics):
                if topic == '/zed2/zed_node/depth/depth_registered' and '/zed2/zed_node/depth/depth_registered' in available_topics:
                    if isinstance(msg, CompressedImage):
                    # Convert the ROS CompressedImage message to an OpenCV image
                        np_arr = np.frombuffer(msg.data, np.uint8)
                        rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # OpenCV >= 3.0
                    else:
                        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

                    # Save the rgb image as a PNG file
                    image_filename = f'{self.image_count:06d}.tiff'
                    image_filepath = os.path.join(self.extraction_folder, self.bag_name, 'images', image_filename)
                    
                    #if folder does not exist, create it
                    os.makedirs(os.path.join(self.extraction_folder, self.bag_name, 'images'), exist_ok=True)
                    
                    # Save the image to a specified folder
                    cv2.imwrite(image_filepath, rgb_image)

                    # Append the timestamp and filename to the image_data list
                    self._rgb_data.append((t.to_sec(), image_filename))
                    # Increment the image count
                    self.image_count += 1

                if topic == '/path' and '/path' in available_topics:
                    # Get the path message
                    path = msg
                    # Append the timestamp and path to the path_data list
                    self._path_data.append((t.to_sec(), path))

                if topic == '/Odometry' and '/Odometry' in available_topics:
                    # Get the Odometry message
                    odom = msg
                    # Append the timestamp and Odometry to the odom_data list
                    self._odom_data.append((t.to_sec(), odom))

                if get_rgb_info and topic == '/rgb/camera_info' and '/rgb/camera_info' in available_topics:
                    # Get the camera info message
                    camera_info = msg
                    # Save the camera info message to a file
                    camera_info_file_path = os.path.join(self.extraction_folder, self.bag_name, 'rgb_camera_info.txt')
                    with open(camera_info_file_path, 'w') as f:
                        f.write("# Camera info\n")
                        f.write(f"# {camera_info.header.stamp}\n")
                        f.write(f"# {camera_info.height} {camera_info.width}\n")
                        f.write(f"# {camera_info.distortion_model}\n")
                        f.write(f"# {camera_info.D}\n")
                        f.write(f"# {camera_info.K}\n")
                        f.write(f"# {camera_info.R}\n")
                        f.write(f"# {camera_info.P}\n")
                    get_rgb_info = False
                if self.image_count % 100 == 0:
                    print(f'Extracted {self.image_count} images.', end='\r')
                if self.image_count > 100:
                    break

    def save_data(self):
        self.extract_bag()
        if self._rgb_data:
            self.save_rgb_text_data('images', self._rgb_data)
        if self._path_data:
            self.save_coordinate_data(self._path_data)
        if self._odom_data:
            print('odom was not empty')
            self.save_Odometry_data(self._odom_data)


if __name__ == "__main__":
    # Choose the output folder location
    output_folder = '/home/sebastian/Documents/code/seb_trav/results/experiment_extracts/outdoor_ridge_depth'
    bag_path = '/home/sebastian/Documents/anymal_experiment_rosbag/sebastian_experiments/anymal_real_message_20240829_122352.bag'
    export_data = ExportData(output_folder, bag_path)
    export_data.save_data()
    print('Data extraction complete.')
