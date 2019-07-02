import cv2
import numpy as np


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

    diamond_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    table = np.array([((i / 255.0) ** 0.4) * 255 for i in np.arange(0, 256)]).astype("uint8")

    video = cv2.VideoCapture(path)
    prev_frame = None

    while True:

        ret, frame = video.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is None:
            prev_frame = frame

        roi = cv2.absdiff(frame, background)
        diff = cv2.absdiff(frame, prev_frame)
        diff = cv2.LUT(diff, table)

        magnitude = calc_magnitude(roi)
        prev_frame = frame

        gamma = cv2.LUT(magnitude, table)
        blur = cv2.bilateralFilter(gamma, 5, 100, 100)
        blur = cv2.bilateralFilter(blur, 5, 100, 100)

        max_contour = contour_largest(blur)
        result = cv2.bitwise_or(blur, blur, mask=max_contour)

        result = cv2.dilate(result, diamond_kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        result = cv2.erode(result, circle_kernel)

        filled = contour_closing(result)
        result = cv2.bitwise_and(blur, blur, mask=filled)
        # result[result < 150] = 0

        blurred_result = cv2.GaussianBlur(contour_closing(result), (3, 3), 0)
        blurred_res = cv2.GaussianBlur(result, (3, 3), 0)

        cv2.imshow("Magnitude", gamma)
        cv2.imshow("Result", blurred_result)
        cv2.imshow("Thresh", blurred_res)
        cv2.imshow("Difference", diff)
        cv2.waitKey(30)

    video.release()


video_path = r'E:\PES\CDSAML\DatasetC\videos\01010fn00.avi'
median_value = median_image(video_path)
detect_roi(video_path, median_value)
cv2.destroyAllWindows()
