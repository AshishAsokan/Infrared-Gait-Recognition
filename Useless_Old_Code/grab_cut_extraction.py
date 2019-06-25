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


def contour_closing(image):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) > 0:
    #     final = calc_contours(image, contours, thresh)
    # else:
    #     final = image
    #
    # cv2.imshow("Result", final)
    # cv2.waitKey(30)

    mask = np.zeros(image.shape[:2], np.uint8)
    height, width = image.shape[:2]

    for c in contours:
        if cv2.contourArea(c) > 800:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 1)

    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    cv2.floodFill(mask, mask1, (0, 0), 255)  # line 27
    mask_inv = cv2.bitwise_not(mask)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))
    # final_result = cv2.erode(mask_inv, kernel_erode)
    cv2.imshow("Final", mask_inv)
    cv2.waitKey(0)


def detect_roi(path, background):

    video = cv2.VideoCapture(path)
    while True:

        ret, frame = video.read()

        if ret is False:
            break

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        roi = cv2.absdiff(frame, background)
        # ret, thresh = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)
        # canny = cv2.Canny(thresh, 100, 250)

        # Calculating image gradients using Sobel derivative
        derivative_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0)
        derivative_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1)

        # Calculating magnitude of image gradients
        dxabs = cv2.convertScaleAbs(derivative_x)
        dyabs = cv2.convertScaleAbs(derivative_y)
        mag = cv2.addWeighted(dxabs, 1.5, dyabs, 1.5, 0)

        mag = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
        # mag = cv2.morphologyEx(mag, cv2.MORPH_OPEN, kernel, iterations=2)

        mag = cv2.dilate(mag, kernel)
        mag = cv2.morphologyEx(mag, cv2.MORPH_OPEN, kernel, iterations=2)
         
        canny = cv2.Canny(mag, 100, 250)
        # contour_closing(mag)

        cv2.imshow("Mag", mag)
        cv2.waitKey(30)

    video.release()


path = r'E:\PES\CDSAML\DatasetC\videos\01010fn00.avi'
back_image = calc_mean_background(path)
detect_roi(path, back_image)
cv2.destroyAllWindows()
