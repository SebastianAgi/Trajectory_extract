import cv2
import numpy as np

def extract_sift_features(frame):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(frame, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def estimate_camera_motion(matches, kp1, kp2):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2)
    return R, t

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_kp, prev_desc = extract_sift_features(prev_frame)
    trajectory = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        kp, desc = extract_sift_features(frame)
        matches = match_features(prev_desc, desc)
        R, t = estimate_camera_motion(matches, prev_kp, kp)
        trajectory.append((R, t))
        prev_kp, prev_desc = kp, desc

    cap.release()
    return trajectory

trajectory = main('/home/sebastian/Documents/code/Trajectory_extract/data/virtual_hikes/forrest_hike_small.mp4')

# To visualize trajectory
import matplotlib.pyplot as plt
trajectory_positions = np.cumsum([t for _, t in trajectory], axis=0)

x = trajectory_positions[:, 0]
y = trajectory_positions[:, 1]
z = trajectory_positions[:, 2]

plt.figure()
plt.plot(x, y, label='Camera trajectory')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.show()
