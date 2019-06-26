import cv2
import numpy as np


def calc_mean_background(path):

    video = cv2.VideoCapture(path)
    initial_frame = None
    avg = None

    while True:

        ret, frame = video.read()

        if ret is False:
            break

        if initial_frame is None:
            initial_frame = frame
            avg = np.float32(frame)

        cv2.accumulateWeighted(frame, avg, 0.08)
        result = cv2.convertScaleAbs(avg)

    video.release()
    return result


def contour_validation(image, size):

    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask_contours = np.ones(image.shape[:2], np.uint8)
    for c in contours:
        if cv2.contourArea(c) < size:
            cv2.drawContours(mask_contours, [c], -1, 0, -1)

    image = cv2.bitwise_and(image, image, mask=mask_contours)
    return image


def skeleton(image):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(image.shape[:2], np.uint8)
    dim = np.size(image)
    done = False

    while not done:
        eroded = cv2.erode(image, kernel)
        dilated = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(image, dilated)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        size = dim - cv2.countNonZero(image)
        if size == dim:
            done = True

    return skel


def contour_closing(dilated_image, gradient_image):

    thresh = cv2.adaptiveThreshold(dilated_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(gradient_image.shape[:2], np.uint8)
    height, width = gradient_image.shape[:2]

    for c in contours:
        cv2.drawContours(mask, [c], 0, (255, 255, 255), 1)

    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    cv2.floodFill(mask, mask1, (0, 0), 255)  # line 27
    mask_inv = cv2.bitwise_not(mask)
    eroded = cv2.erode(mask_inv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    cv2.imshow("Final", eroded)
    cv2.imshow("Magnitude", gradient_image)
    cv2.waitKey(30)


def detect_roi(path, background):

    video = cv2.VideoCapture(path)
    while True:

        ret, frame = video.read()

        if ret is False:
            break

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        roi = cv2.absdiff(frame, background)
        # ret, thresh = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)
        # canny = cv2.Canny(thresh, 100, 250)

        # Calculating image gradients using Sobel derivative
        derivative_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
        derivative_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1)

        # Calculating magnitude of image gradients
        dxabs = cv2.convertScaleAbs(derivative_x)
        dyabs = cv2.convertScaleAbs(derivative_y)
        magnitude = cv2.addWeighted(dxabs, 9.0, dyabs, 9.0, 3)
        gray = cv2.cvtColor(magnitude, cv2.COLOR_BGR2GRAY)

        magnitude[magnitude < 32] = 0

        cv2.imshow("Magnitude", magnitude)
        cv2.waitKey(30)

    video.release()


path_video = r'E:\PES\CDSAML\DatasetC\videos\01001fn00.avi'
back_image = calc_mean_background(path_video)
detect_roi(path_video, back_image)
cv2.destroyAllWindows()
