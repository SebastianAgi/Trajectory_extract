import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import sys
import time
from PIL import Image
from torchvision import transforms as T
import os
from segment_anything import sam_model_registry, SamPredictor
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score


class SAM:
    def __init__(self) -> None:
        self.sam_checkpoint = "/home/sebastian/Documents/code/SAM/sam_vit_h_4b8939.pth"
        # self.sam_checkpoint = "/home/sebastian/Documents/code/SAM/sam_vit_b_01ec64.pth"
        self.model_type = "vit_h"
        self.device = "cuda"

    def show_mask(self, mask):
        color = np.array([30/255, 144/255, 255/255], dtype=np.float32)  # ensure float32 if using with float images
        mask_region = mask * color.reshape(1, 1, 3)  # reshape color to broadcast properly
        mask_region = (mask_region * 255).astype(np.uint8)
        return mask_region

    def sam_segmentation(self, image, input_points, previous_mask) -> np.ndarray:
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        print('SHAPE:',image.shape)
        predictor.set_image(image)
        # print('input_points: ', input_points)
        input_label = np.ones(input_points.shape[0])

        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=True,
        )

        #return segmentation of the image
        max_score_index = np.argmax(scores)
        
        # if np.sum(previous_mask)/3 > 5*masks[max_score_index]:

        # ones_count = np.sum(masks, axis=(1, 2))
        # min_ones_index = np.argmin(ones_count)

        mask = np.stack([masks[max_score_index]]*3, axis=-1)
        return mask * image, mask


class SIFT(SAM):

    def __init__(self) -> None:
        # Call the constructor of the parent class SAM
        super().__init__()
        # Initialize variables
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.points_to_use = None
        self.matches = []
        self.previous_mask = None

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.axis('off')

    def sift_features(self, segmented_image):
        # Parameters for SIFT
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        return keypoints, descriptors
    
    def flann_matcher(self, descriptors1, descriptors2):
        # Parameters for FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        # Filter matches using the Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]

        return good_matches
    
    def optimal_kmeans(self, data, max_clusters=10):
        """
        Apply K-means clustering to determine the optimal number of clusters
        using the Davies-Bouldin index.

        Parameters:
        - data: np.array of shape (n_samples, 2), where each row represents a 2D point.
        - max_clusters: int, maximum number of clusters to try.

        Returns:
        - centroids: np.array, centroids of the clusters that best represent the points.
        """
        # Ensure max_clusters is an integer
        max_clusters = int(max_clusters)
        
        # Dictionary to store the Davies-Bouldin scores for different numbers of clusters
        db_scores = {}
        
        # Apply K-means clustering for each possible number of clusters from 2 to max_clusters
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Set n_init explicitly
            labels = kmeans.fit_predict(data)
            # Calculate Davies-Bouldin index
            db_index = davies_bouldin_score(data, labels)
            db_scores[k] = db_index
        
        # Find the number of clusters with the minimum Davies-Bouldin index
        best_k = min(db_scores, key=db_scores.get)
        
        # Recompute K-means with the optimal number of clusters
        kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)  # Set n_init explicitly
        kmeans.fit(data)
        
        # Convert centroids to integer tuples for OpenCV drawing
        centroids = np.round(kmeans.cluster_centers_).astype(int)
        
        # Return the centroids of the optimal clusters
        return centroids
    
    def draw_trajectory(self, image, keypoints1, keypoints2, good_matches, mask):
        # Create an image to draw the trajectories
        trajectory_image = image
        self.matches = []
        points_to_cluster = []
        
        #overlay mask as a transparent layer on top of the image
        mask_region = self.show_mask(mask)
        trajectory_image = cv2.addWeighted(trajectory_image, 1, mask_region, 0.5, 0)

        # Draw the movement of keypoints
        for match in good_matches:
            pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype("int"))
            pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype("int"))
            pt2 = (pt2[0], pt2[1])
            points_to_cluster.append(pt2)
            # cv2.circle(trajectory_image, pt1, 5, (255, 0, 0), -1)  # Red starting point
            # cv2.circle(trajectory_image, pt2, 5, (0, 0, 255), -1)  # Blue ending point
            cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)   # Green line for movement

        # Calculate the centroid of the keypoints
        try:
            centroids = self.optimal_kmeans(points_to_cluster)           

            for (x, y) in centroids:
                cv2.circle(trajectory_image, (int(x), int(y)), 5, (255, 0, 0), -1)  # Red centroid
        except:
            centroids = []
            print('Error: Cannot calculate the optimal number of clusters.')

        return trajectory_image, centroids
    def update(self, i):
        # Read the current frame
        image_path = self.images[i]
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}.")
            return self.im,  # Return current image artist for FuncAnimation

        temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg, mask = self.sam_segmentation(temp, self.points_to_use, self.previous_mask)
        keypoints, descriptors = self.sift_features(seg)

        if self.previous_frame is not None:
            good_matches = self.flann_matcher(self.previous_descriptors, descriptors)
            trajectory_image, self.matches = self.draw_trajectory(temp, self.previous_keypoints, keypoints, good_matches, mask)
            self.previous_mask = mask
            self.im.set_data(trajectory_image)  # Update the image displayed

        self.previous_frame = np.zeros_like(frame)
        self.previous_frame.fill(255)
        self.points_to_use = np.array(self.matches) if len(self.matches) != 0 else self.points_to_use
        print('points_to_use: ', self.points_to_use)
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

        return self.im,

    def track_features_in_video_frames(self, folder_path, init_points):
        self.images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])
        # self.images = self.images[::3]
        self.points_to_use = init_points

        # Initialize the image artist for FuncAnimation
        first_frame = cv2.imread(self.images[0])
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        self.im = self.ax.imshow(first_frame)
        
        # Create animation
        ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.images), blit=True)
        plt.show()

if __name__ == "__main__":
    sift = SIFT()
    folder_path = '/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame'
    # init_points = np.array([[1920/2, 900], [1920/2, 920], [1920/2, 940], [1920/2, 960], [1920/2, 980]])
    init_points = np.array([[1920/2 - 10, 1000],[1920/2 - 20, 1020],[1920/2 - 30, 1050],[1920/2 - 40, 1060],[1920/2, 1030]])
    sift.track_features_in_video_frames(folder_path, init_points)