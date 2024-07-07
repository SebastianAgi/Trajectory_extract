pixel_width = 5776
pixel_height = 4336
sensor_width = 17.3
sensor_height = 13.0
focal_length = 6.0
fx = (pixel_width/sensor_width)*focal_length
fy = (pixel_height/sensor_height) * focal_length
cx = pixel_width / 2
cy = pixel_height / 2


print("fx: ", fx)
print("fy: ", fy)
print("cx: ", cx)
print("cy: ", cy)