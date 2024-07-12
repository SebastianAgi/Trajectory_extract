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
from sklearn.cluster import KMeans
import json
import rospy


class Sam:
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

    def sam_segmentation(self, image, input_points) -> np.ndarray:
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        # print('input_points: ', input_points)
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

class Traversability(Sam):

    def __init__(self) -> None:
        # Call the constructor of the parent class SAM
        super().__init__()
        # Initialize variables
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.points_to_use = None
    
    def draw_trajectory(self, image, points, mask):
        # Create an image to draw the trajectories
        trajectory_image = image

        #overlay mask as a transparent layer on top of the image
        mask_region = self.show_mask(mask)
        trajectory_image = cv2.addWeighted(trajectory_image, 1, mask_region, 0.5, 0)

        # print('points: ', points)

        past_point = None

        # Draw the movement of keypoints
        for point in points:
            # print(point)
            # cv2.circle(trajectory_image, point, 5, (255, 0, 0), -1)  # Red centroid
            # cv2.line(trajectory_image, past_point, point, (0, 255, 0), 2)   # Green line for movement
            
            cv2.circle(trajectory_image, (point[0], point[1]), 5, (0, 0, 255), -1) 

            if past_point != None:
                cv2.line(trajectory_image, (point[0], point[1]), (past_point[0], past_point[1]), (0, 255, 0), 2)    # green line

            past_point = point

        return trajectory_image
    
    def track_features_in_video_frames(self, folder_path, path_points):
        # List all image files in the folder
        images = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith((".png", ".jpg", ".jpeg"))])
        point_count =-1

        start = 700
        mask_count = 0
        save_flag = True

        #create cv2 window
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1280, 720)

        print('path_points:', path_points[-7])
        images = images[start:]
        path_points = path_points[start:]

        for image_path in images:
            # Read the current frame
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Cannot read image {image_path}.")
                continue
            
            if path_points[point_count] == []:
                print('No points for this frame, skipping frame', end='\r')
                point_count += 1
                continue
            
            # print('\n')

            print('point_count:', point_count, '/', len(path_points), end='\r')
            # print('path_points:', path_points[point_count])

            temp =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            seg, mask = self.sam_segmentation(frame, np.array(path_points[point_count]))

            # Draw the movement of keypoints
            trajectory_image = self.draw_trajectory(frame, path_points[point_count], mask)

            # Display the result
            cv2.imshow('image', trajectory_image)
            # wait 0.01 seconds
            rospy.sleep(0.01)
            # cv2.waitKey(0)
            point_count += 1

            if save_flag:
                #save mask as a binary image
                path_mask = mask[:, :, 0]
                path_mask = path_mask.astype(np.uint8)
                mask_count += 1
                cv2.imwrite(f'/home/sebastian/Documents/ANYmal_data/Masked_data/odom_data_masked/masks/mask_{mask_count}.png', path_mask*255)
                cv2.imwrite(f'/home/sebastian/Documents/ANYmal_data/Masked_data/odom_data_masked/mask_overlay/trajectory_{mask_count}.png', trajectory_image)



if __name__ == "__main__":
    trav = Traversability()

    # Path to your folder containing images
    folder_path = '/home/sebastian/Documents/ANYmal_data/odom_chosen_images_2'

    #load points from json file
    with open('all_points.json', 'r') as f:
        path_points = json.load(f)
    
    # print(path_points[-7])

    trav.track_features_in_video_frames(folder_path, path_points)

    test_sam = False

    if test_sam == True:
        #test sam segmentation
        test_image = cv2.imread('/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame/frame_0000.jpg')
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # track_features_in_video_frames(folder_path, path_points)
        segment = trav.sam_segmentation(test_image, path_points)

        plt.figure(figsize=(12, 6))
        plt.imshow(segment)
        plt.title('Segmentation')
        plt.axis('off')
        plt.show()