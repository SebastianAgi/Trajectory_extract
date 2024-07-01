import os
import cv2


#function to convert mp4 file to frames using opencv
def mpy_to_frames(video_path, frames_path):
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_path, 'frame{:d}.jpg'.format(count)), frame)
        count += 1
    cap.release()
    print('Frames extracted successfully')

#Usage
mpy_to_frames('/home/sebastian/code/virtual_forrest_hike_720.mp4', '/home/sebastian/code/virtual_hike/virtual_forrset_hike_720')

