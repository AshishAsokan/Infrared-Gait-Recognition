
'''
This program converts infrared footage into
binary silhouette footage for better feature extraction
'''

# * Modules imported

import numpy as np
import cv2 as cv


# * Functions

def background_subtraction(cap, fgbg):

    while(1):
        ret, frame = cap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            blur = cv.GaussianBlur(fgmask, (5, 5), 0)
            ret3, th3 = cv.threshold(
                blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            if ret:
                cv.imshow('frame', th3)
                k = cv.waitKey(30) & 0xff
                if k == 27:
                    break
            else:
                break
        else:
            break


cap = cv.VideoCapture('video.avi')
fgbg = cv.createBackgroundSubtractorMOG2()
background_subtraction(cap, fgbg)
video.release()
cv.destroyAllWindows()
