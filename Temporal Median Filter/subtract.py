
import cv2
import numpy as np

def calc_magnitude(image):

    image = cv2.GaussianBlur(image, (5, 5), 3)
    # Calculating image gradients using Sobel derivative
    derivative_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    derivative_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    # Calculating magnitude of image gradients
    dxabs = cv2.convertScaleAbs(derivative_x)
    dyabs = cv2.convertScaleAbs(derivative_y)
    magnitude = cv2.addWeighted(dxabs, 9.0, dyabs, 9.0, 9)

    return magnitude

video_path = r"D:\CDSAML_2019\Gait_IR\DatasetC\DatasetC\videos\01010fn00.avi"
median_path = r"D:\CDSAML_2019\Gait_IR\Temporal Median Filter\Median.mp4"

video_obj = cv2.VideoCapture(video_path)
median_obj = cv2.VideoCapture(median_path)

output = list()

while True:
    ret_video, frame_video = video_obj.read()
    ret_median, frame_median = median_obj.read()

    if ret_video is False or ret_median is False:
        break
    
    diff_frame = cv2.absdiff(frame_video, frame_median)
    cv2.imshow("mag",calc_magnitude(diff_frame))
    cv2.waitKey(30)
    output.append(diff_frame)

out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'XVID'), 25.0, (320,240))

for i in output:
    out.write(i)
out.release()
video_obj.release()
median_obj.release()
cv2.destroyAllWindows()