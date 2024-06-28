import cv2
import numpy as np
import matplotlib.pyplot as plt

def SLIC_segmentation(image, num_superpixels=35, compactness=100):
    # Convert the image to a 3-channel image if it's grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Perform SLIC segmentation
    slic = cv2.ximgproc.createSuperpixelSLIC(image, cv2.ximgproc.SLICO, num_superpixels, compactness)
    slic.iterate(10)
    slic.enforceLabelConnectivity()
    
    # Get the labels and mask of the superpixels
    labels = slic.getLabels()
    mask = slic.getLabelContourMask()
    
    # Create an output image with superpixel boundaries
    segmented_image = image.copy()
    segmented_image[mask == 255] = [0, 0, 255]
    
    return labels, mask, segmented_image

def main(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Perform SLIC segmentation
    labels, mask, segmented_image = SLIC_segmentation(image)
    
    # Display the original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Display the superpixel labels
    # plt.subplot(1, 2, 1)
    # plt.imshow(labels, cmap='tab20b')
    # plt.title('Superpixel Labels')
    # plt.axis('off')
    
    # Display the segmented image with superpixel boundaries
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image with Boundaries')
    plt.axis('off')
    
    plt.show()

if __name__ == "__main__":
    main('/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame/frame_0000.jpg')
