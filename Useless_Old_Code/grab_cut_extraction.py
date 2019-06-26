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


def contour_closing(dilated_image, gradient_image):

    thresh = cv2.adaptiveThreshold(dilated_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 11)

    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(gradient_image.shape[:2], np.uint8)
    height, width = gradient_image.shape[:2]

    for c in contours:
        cv2.drawContours(mask, [c], 0, (255, 255, 255), 1)

    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    cv2.floodFill(mask, mask1, (0, 0), 255)  # line 27
    mask_inv = cv2.bitwise_not(mask)

    cv2.imshow("Final", mask_inv)
    cv2.imshow("Magnitude", gradient_image)
    cv2.waitKey(0)


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
        magnitude = cv2.addWeighted(dxabs, 1.5, dyabs, 1.5, 0)

        magnitude[magnitude < 32] = 0
        gray = cv2.cvtColor(magnitude, cv2.COLOR_BGR2GRAY)
        con = contour_validation(gray, 800)

        contour_closing(con, magnitude)
        # cv2.imshow("Magnitude", magnitude)
        # cv2.imshow("Contours", gray)
        # cv2.waitKey(0)

    video.release()


path_video = r'E:\PES\CDSAML\DatasetC\videos\01010fn00.avi'
back_image = calc_mean_background(path_video)
detect_roi(path_video, back_image)
cv2.destroyAllWindows()
