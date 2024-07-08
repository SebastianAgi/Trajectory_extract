import cv2
import numpy as np
import matplotlib.pyplot as plt
print(cv2.__version__)
import os

def track_features_in_video_frames(folder_path):
    # List all image files in the folder
    images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])

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

    for image_path in images:
        # Read the current frame
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}.")
            continue

        # Consider only the bottom half of the image
        height = frame.shape[0]
        bottom_half = frame[height//2:height, 250:frame.shape[1]-250]

        # Convert to grayscale
        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)

        # Detect SIFT features and compute descriptors
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if previous_frame is not None:
            # Match descriptors between the previous and current frames
            matches = flann.knnMatch(previous_descriptors, descriptors, k=2)

            # Filter matches using the Lowe's ratio test
            good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

            # Create an image to draw the trajectories
            trajectory_image = np.zeros_like(previous_frame)
            trajectory_image.fill(255)  # Fill with white

            # Draw the movement of keypoints
            for match in good_matches:
                pt1 = tuple(np.round(previous_keypoints[match.queryIdx].pt).astype("int"))
                pt2 = tuple(np.round(keypoints[match.trainIdx].pt).astype("int"))
                pt2 = (pt2[0], pt2[1])  # Adjust the y-coordinate for the bottom half
                cv2.circle(trajectory_image, pt1, 5, (255, 0, 0), -1)  # Blue starting point
                cv2.circle(trajectory_image, pt2, 5, (0, 0, 255), -1)  # Red ending point
                cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)   # Green line for movement

            # Display the result
            plt.figure(figsize=(12, 6))
            plt.imshow(trajectory_image)
            plt.title('Feature Movements')
            plt.axis('off')
            plt.show()
            
            # Update variables for the next iteration
            previous_frame = trajectory_image if previous_frame is not None else np.zeros_like(bottom_half)
            previous_frame.fill(255)
            previous_keypoints = keypoints
            previous_descriptors = descriptors
            print(previous_keypoints)

# Path to your folder containing images
folder_path = '/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame'
track_features_in_video_frames(folder_path)