import cv2
import os
import subprocess

# Directory containing frames
frame_directory = '/home/sebastian/Documents/ANYmal_data/mine_hanheld_forest/anymal_real_message_mine_handheld_forest_extracted/images'
# Output video file path
output_video_path = '/home/sebastian/Documents/ANYmal_data/mine_hanheld_forest/anymal_real_message_mine_handheld_forest_extracted/video.mp4'
# Frame rate
frame_rate = 15.0

# Get a list of all files in the directory
frames = [f for f in os.listdir(frame_directory) if f.endswith('.png')]
# Sort the frames if they are not in the correct order
frames.sort()

frames = frames[255:-330]

# Read the first frame to get the dimensions
first_frame = cv2.imread(os.path.join(frame_directory, frames[0]))
height, width, layers = first_frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 files, use 'X264' for better compression
video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

# Write each frame to the video
for frame in frames:
    img = cv2.imread(os.path.join(frame_directory, frame))
    video_writer.write(img)

# Release the VideoWriter
video_writer.release()

print(f'Video saved to {output_video_path}')

# Additional compression using ffmpeg
# Path to the compressed video file
compressed_video_path = '/home/sebastian/Documents/ANYmal_data/mine_hanheld_forest/anymal_real_message_mine_handheld_forest_extracted/compressed_video.mp4'

# ffmpeg command to compress the video
ffmpeg_command = [
    'ffmpeg', 
    '-i', output_video_path, 
    '-vcodec', 'libx264', 
    '-crf', '28',  # Adjust the CRF value as needed (23-28 is a good range)
    compressed_video_path
]

# Run the ffmpeg command
subprocess.run(ffmpeg_command)

print(f'Compressed video saved to {compressed_video_path}')