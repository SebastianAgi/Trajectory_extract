import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_video_with_optical_flow(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find initial points (features) to track
    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)

    # Calculate optical flow (i.e., track feature points)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points (where st=1, indicating that the flow for these points has been found)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Create an image to draw the tracks
    mask = np.zeros_like(frame1)

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Convert coordinates to integer
        a, b = int(a), int(b)
        c, d = int(c), int(d)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame1 = cv2.circle(frame1, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame1, mask)

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Optical Flow')
    plt.axis('off')
    plt.show()

# Path to your video file
frame1 = cv2.imread('/home/sebastian/Documents/code/Trajectory_extraction/hike_frame_by_frame/frame_0056.jpg')
frame2 = cv2.imread('/home/sebastian/Documents/code/Trajectory_extraction/hike_frame_by_frame/frame_0057.jpg')
process_video_with_optical_flow(frame1, frame2)
