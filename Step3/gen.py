from glob import glob
import cv2

p = glob(r'D:\CDSAML_2019\Gait_IR\CT\Valid_videos\*')

for vid in p:
    occ = 0
    a = cv2.VideoCapture(vid)
    pth = vid.replace('Valid_videos','NEW')
    b = cv2.VideoWriter(pth,cv2.VideoWriter_fourcc(*'MP4V'),25.0,(320,240))
    while True:
        ret, frame = a.read()
        if ret == False:
            break
        fr = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        n = cv2.countNonZero(fr)
        if n < 3930 and occ == 0:
            continue
        elif n > 3930 and n < 6000:
            occ  = 1
        if occ == 1:
            b.write(frame)
        if occ >= 1 and n < 2530:
            occ += 1
        if occ > 5:
            break
    b.release()