import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        print(f"Saved {frame_filename}")

    # Release the video capture object
    cap.release()
    print("Done extracting frames.")

# Usage
video_path = '/home/sebastian/Documents/code/Trajectory_extraction/forrest_hike_small.mp4'  # Replace with your video file path
output_folder = 'hike_frame_by_frame'        # Folder to save the extracted frames
extract_frames(video_path, output_folder)
