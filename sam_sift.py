import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import time
from PIL import Image
from torchvision import transforms as T
import os
from segment_anything import sam_model_registry, SamPredictor


class SAM:
    def __init__(self) -> None:
        self.sam_checkpoint = "/home/sebastian/code/segment_anything/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda"

    def show_mask(self, mask):
        color = np.array([30/255, 144/255, 255/255], dtype=np.float32)  # ensure float32 if using with float images
        mask_region = mask * color.reshape(1, 1, 3)  # reshape color to broadcast properly
        mask_region = (mask_region * 255).astype(np.uint8)
        return mask_region

    def sam_segmentation(self, image, input_points) -> np.ndarray:
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        print('input_points: ', input_points)
        input_label = np.ones(input_points.shape[0])

        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_label,
            multimask_output=True,
        )

        #return segmentation of the image
        mask = np.stack([masks[0]]*3, axis=-1)
        segment = mask * image

        return segment, mask

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
    
    def draw_trajectory(self, image, keypoints1, keypoints2, good_matches, mask):
        # Create an image to draw the trajectories
        trajectory_image = image

        #overlay mask as a transparent layer on top of the image
        mask_region = self.show_mask(mask)
        trajectory_image = cv2.addWeighted(trajectory_image, 1, mask_region, 0.5, 0)

        # Draw the movement of keypoints
        for match in good_matches:
            pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype("int"))
            pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype("int"))
            pt2 = (pt2[0], pt2[1])
            cv2.circle(trajectory_image, pt1, 5, (255, 0, 0), -1)  # Red starting point
            cv2.circle(trajectory_image, pt2, 5, (0, 0, 255), -1)  # Blue ending point
            cv2.line(trajectory_image, pt1, pt2, (0, 255, 0), 2)   # Green line for movement
            self.matches.append(pt2)

        return trajectory_image, self.matches
    
    def track_features_in_video_frames(self, folder_path, init_points):
        # List all image files in the folder
        images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])

        for image_path in images:
            # Read the current frame
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Cannot read image {image_path}.")
                continue

            if self.points_to_use is None:
                points = init_points
            else:
                points = self.points_to_use

            temp =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg, mask = self.sam_segmentation(temp, points)
            keypoints, descriptors = self.sift_features(seg)

            if self.previous_frame is not None:
                # Match descriptors between the previous and current frames
                good_matches = self.flann_matcher(self.previous_descriptors, descriptors)
                
                # Draw the movement of keypoints
                trajectory_image, self.matches = self.draw_trajectory(temp, self.previous_keypoints, keypoints, good_matches, mask)

                # Display the result
                plt.figure(figsize=(12, 6))
                plt.imshow(trajectory_image)
                plt.title('Feature Movements')
                plt.axis('off')
                plt.show()

            # Update variables for the next iteration
            self.previous_frame = np.zeros_like(frame)
            self.previous_frame.fill(255)
            self.points_to_use = np.array(self.matches) if len(self.matches) != 0 else points
            self.previous_keypoints = keypoints
            self.previous_descriptors = descriptors


if __name__ == "__main__":
    sift = SIFT()

    # Path to your folder containing images
    folder_path = '/home/sebastian/code/Trajectory_extraction/hike_frame_by_frame'
    init_points = np.array([[1920/2, 900],[1920/2, 920],[1920/2, 940],[1920/2, 960],[1920/2, 980]])

    sift.track_features_in_video_frames(folder_path, init_points)

    test_sam = False

    if test_sam == True:
        #test sam segmentation
        test_image = cv2.imread('/home/sebastian/code/Trajectory_extraction/hike_frame_by_frame/frame_0000.jpg')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # track_features_in_video_frames(folder_path, init_points)
        segment = sift.sam_segmentation(test_image, init_points)

        plt.figure(figsize=(12, 6))
        plt.imshow(segment)
        plt.title('Segmentation')
        plt.axis('off')
        plt.show()