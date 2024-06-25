import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def track_features_in_video_frames(folder_path):
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])
    # images = images[-100:]
    # Select every 5th image from the list
    every_fifth_image = images[::2]

    # Initialize variables
    previous_frame = None
    previous_keypoints = None
    previous_descriptors = None

    # Parameters for SIFT
    sift = cv2.SIFT_create()

    # Parameters for FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(image_path):
        nonlocal previous_frame, previous_keypoints, previous_descriptors
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}.")
            return

        height = frame.shape[0]
        bottom_half = frame[height//2:height, 250:frame.shape[1]-250]
        temp =  cv2.cvtColor(bottom_half, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if previous_frame is not None and previous_descriptors is not None:
            matches = flann.knnMatch(previous_descriptors, descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

            # trajectory_image = np.zeros_like(frame)
            # trajectory_image.fill(255)  # white background

            #set background as current frame
            trajectory_image = temp

            vectors = []
            directions = []
            avg_dir = (0, 0)
            mag = 50

            for match in good_matches:
                pt1 = tuple(np.round(previous_keypoints[match.queryIdx].pt).astype("int"))
                pt2 = tuple(np.round(keypoints[match.trainIdx].pt).astype("int"))
                pt2 = (pt2[0], pt2[1])  # adjust y-coordinate
                # cv2.circle(trajectory_image, (710, 270), 5, (255, 0, 0), -1)  # blue start
                cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)    # green line
                hyp = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                if hyp != 0.0:
                    vectors.append(((pt1[0] - pt2[0])/hyp, (pt1[1] - pt2[1])/hyp))
            
            if len(vectors) != 0:
                directions = np.array(vectors)
                mid_point = (int(temp.shape[1]/2), int(temp.shape[0])-100)
                avg_dir = np.mean(directions, axis=0)
                print("avg_dir:", avg_dir)
                dif = (int(mid_point[0] + avg_dir[0]*mag), int(mid_point[1] + avg_dir[1]*mag))
                cv2.line(trajectory_image, mid_point, dif, (255, 0, 0), 5)

            ax.clear()
            ax.imshow(trajectory_image)
            ax.set_title('Feature Movements')
            ax.axis('off')

        previous_frame = frame
        previous_keypoints = keypoints
        previous_descriptors = descriptors

    ani = FuncAnimation(fig, update, frames=every_fifth_image, repeat=False)
    plt.show()

# Path to your folder containing images
folder_path = '/home/sebastian/Documents/code/Trajectory_extract/hike_frame_by_frame'
track_features_in_video_frames(folder_path)
