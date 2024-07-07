import matplotlib.pyplot as plt
import os

# Directory containing your images
image_dir = "/home/sebastian/Documents/code/Trajectory_extract/data/hike_frame_by_frame"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

# This will store the coordinates
coordinates = {}

# Define the index for the current image
current_image_index = 0

def onclick(event):
    global current_image_index
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        print(f"Recorded coordinates: ({x}, {y})")
        if current_image_index not in coordinates:
            coordinates[current_image_index] = []
        coordinates[current_image_index].append((x, y))
        if len(coordinates[current_image_index]) >= 3:
            plt.close()

def display_image():
    global current_image_index
    img = plt.imread(os.path.join(image_dir, image_files[current_image_index]))
    plt.figure(figsize=(16, 10))
    plt.imshow(img)
    plt.title(f"Image {current_image_index + 1}/{len(image_files)}: {image_files[current_image_index]}")
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

fig = plt.figure()
# Display the first image
for i in range(0, len(image_files), 100):
    print(f"{image_files[i]}")
    display_image()
    current_image_index = i

with open("coordinates.txt", "w") as file:
    for idx, coords in coordinates.items():
        file.write(f"{image_files[idx]}: {coords}\n")
