import cv2
import numpy as np
from os import getcwd
from glob import glob


def median_image(path):

    """
    Calculates the median image based on the number of images
    :param      path: Path of the infrared video
    :return:    frame: The median image when n value is encountered
    :return:    n_frames: Number of frames read from video
    """

    video_obj = cv2.VideoCapture(path)
    width = int(video_obj.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_obj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(video_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    images = np.zeros((height, width) + (frames,))

    for i in range(frames):
        ret, frame = video_obj.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        images[:, :, i] = frame

    result = np.median(images, axis=2)
    result = np.uint8(result)
    return result


def contour_largest(image):

    image = cv2.GaussianBlur(image, (5, 5), 3)
    ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], np.uint8)

    if len(contours) > 0:
        large = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [large], 0, (255, 255, 255), 2)

    return mask


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


def contour_closing(image):

    # thresh = cv2.adaptiveThreshold(dilated_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape[:2], np.uint8)
    height, width = image.shape[:2]

    for c in contours:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 1)

    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    cv2.floodFill(mask, mask1, (0, 0), 255)  # line 27
    mask_inv = cv2.bitwise_not(mask)
    # eroded = cv2.erode(mask_inv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return mask_inv


def detect_roi(path, background):

    video = cv2.VideoCapture(path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    name = "D:\\CDSAML_2019\\Gait_IR\\CT\\Videos\\" + path[-13:-4] + ".mp4"
    video_write = cv2.VideoWriter(name, fourcc, 25.0, (width, height))
    while True:

        ret, frame = video.read()

        if ret is False:
            break

        diamond_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        table = np.array([((i / 255.0) ** 0.6) * 255 for i in np.arange(0, 256)]).astype("uint8")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.absdiff(frame, background)
        magnitude = calc_magnitude(roi)
        magnitude_frame = calc_magnitude(frame)

        gamma = cv2.LUT(magnitude, table)
        blur = cv2.bilateralFilter(gamma, 9, 100, 100)
        blur = cv2.bilateralFilter(blur, 9, 100, 100)

        max_contour = contour_largest(blur)
        result = cv2.bitwise_or(blur, blur, mask=max_contour)

        result = cv2.dilate(result, diamond_kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        result = cv2.erode(result, circle_kernel)

        filled = contour_closing(result)
        result = cv2.bitwise_and(blur, blur, mask=filled)
        result[result < 130] = 0

        blurred_result = cv2.GaussianBlur(contour_closing(result), (3, 3), 0)

        
        blurred_result = cv2.cvtColor(blurred_result, cv2.COLOR_GRAY2BGR)
        video_write.write(blurred_result)
    video_write.release()

path_to_videos_folder = getcwd() + "\\Gait_IR\\DatasetC\\DatasetC\\videos\\*.avi"
path_to_all_videos_list = glob(path_to_videos_folder)
for path in path_to_all_videos_list:
    back_image = median_image(path)
    detect_roi(path, back_image)
    