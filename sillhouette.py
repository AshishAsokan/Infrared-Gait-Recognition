import cv2
import numpy as np
import glob

images = []
walking = []
contents = glob.glob(r"E:\PES\CDSAML\GaitDatasetC-silh\001\*")

for path in contents:
    walking = []
    sub = glob.glob(path + '\*.png')
    for temp in sub:
        img = cv2.imread(temp)
        if img is None:
            break
        walking.append(img)
    images.append(walking)

width, height, ch = images[0][0].shape
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
for i in range(len(images)):
    video = cv2.VideoWriter('Walk' + str(i + 1) + '.mp4', fourcc, 20.0, (height, width))
    for j in images[i]:
        video.write(j)

cv2.destroyAllWindows()
video.release()
