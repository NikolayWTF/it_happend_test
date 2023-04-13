import cv2

colors = [['black', 0,0,0], ['white', 255, 255, 255], ['red', 255, 0, 0],
 ['lime color', 0, 255, 0], ['blue', 0, 0, 255], ['yellow', 255, 255, 0],
 ['blue aqua', 0, 255, 255], ['pink', 255, 0, 255], ['Silver', 192, 192, 192],
 ['grey', 128, 128, 128], ['burgundy', 128, 0, 0], ['green', 0, 128, 0], 
 ['violet', 128, 0, 128], ['turquoise', 0, 128, 128], ['Navy blue', 0, 0, 128]]

image = cv2.imread("video_to_img/tmp.jpg")
b, g, r = image[145, 296]
minimum = 1000
color_name = 'black'
for i in range(len(colors)):
    d = abs(colors[i][1] - r) + abs(colors[i][2]- g)+ abs(colors[i][3]- b)
    if d <= minimum:
      minimum = d
      color_name = colors[i][0]
print (r, g, b)
print(color_name)
