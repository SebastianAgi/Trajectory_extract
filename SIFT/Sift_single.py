import cv2
import numpy as np
import matplotlib.pyplot as plt
print(cv2.__version__)
import os

# Function to process a video
def process_video(frame1, frame2):

    # Parameters for SIFT
    magnitude = 5

    height1 = frame1.shape[0]
    bottom_half1 = frame1[height1//2:height1, 250:frame1.shape[1]-250]

    height2 = frame2.shape[0]
    bottom_half2 = frame2[height2//2:height2, 250:frame2.shape[1]-250]

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(bottom_half1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(bottom_half2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    print(type(keypoints1[0].pt[0]))
    # Create FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors between the two frames
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    # Create an image to draw the trajectories
    height, width = gray1.shape
    trajectory_image = np.zeros((height, width, 3), dtype=np.uint8)
    trajectory_image.fill(255)  # Fill with white

    # Draw the movement of keypoints
    for match in good_matches:
        pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype("int"))
        pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype("int"))
        cv2.circle(trajectory_image, pt1, 5, (255, 0, 0), -1)  # Blue starting point
        
        cv2.circle(trajectory_image, pt1, 5, (255, 0, 0), -1)  # Blue starting point
        cv2.circle(trajectory_image, pt2, 5, (0, 0, 255), -1)  # Red ending point
        # cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)   # Green line for movement

        # Calculate vector for extension
        vector_x = pt2[0] - pt1[0]
        vector_y = pt2[1] - pt1[1]

        # Extend the endpoint by 50% of the vector length beyond the second point
        extended_pt2 = (int(pt2[0] + magnitude*vector_x), int(pt2[1] + magnitude*vector_y))

        cv2.line(trajectory_image, pt1, extended_pt2, (0, 255, 0), 2)  # Extended green line

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.imshow(trajectory_image)
    plt.title('Feature Movements')
    plt.axis('off')
    plt.show()

    print("Number of good matches:", len(good_matches))

# Path to your video file
frame1 = cv2.imread('/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame/frame_0056.jpg')
frame2 = cv2.imread('/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame/frame_0057.jpg')
process_video(frame1, frame2)