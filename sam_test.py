import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import time
from PIL import Image
from torchvision import transforms as T

image = cv2.imread('/home/sebastian/code/Trajectory_extraction/hike_frame_by_frame/frame_0056.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)

sam_checkpoint = "/home/sebastian/code/segment_anything/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

predictor.set_image(image)

#create 5 points
input_points = np.array([[1920/2, 900],[1920/2, 920],[1920/2, 940],[1920/2, 960],[1920/2, 980]])

# input_points = np.array([[1920/2, 900],[1920/2, 1000]]) 
# input_label = np.array([1, 1, 1, 1, 1])
input_label = np.ones(input_points.shape[0])

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_label,
    multimask_output=True,
)

masks.shape  # (number_of_masks) x H x W
#print number of non-zero elements in the mask
for i, mask in enumerate(masks):
    print(f"Mask {i+1} has {mask.sum()} non-zero elements")

for i, (mask, score) in enumerate(zip(masks, scores)):
    # if score < 0.8:
    #     continue
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_points, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()