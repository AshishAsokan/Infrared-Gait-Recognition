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


def contour_largest(image):

    image = cv2.GaussianBlur(image, (5, 5), 3)
    ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], np.uint8)

    if len(contours) > 0:
        large = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [large], 0, (255, 255, 255), 4)

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


def contour_closing(dilated_image):

    # thresh = cv2.adaptiveThreshold(dilated_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 1)

    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(dilated_image.shape[:2], np.uint8)
    height, width = dilated_image.shape[:2]

    for c in contours:
        if cv2.contourArea(c) > 100:
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 1)

    mask1 = np.zeros((height + 2, width + 2), np.uint8)  # line 26
    cv2.floodFill(mask, mask1, (0, 0), 255)  # line 27
    mask_inv = cv2.bitwise_not(mask)
    # eroded = cv2.erode(mask_inv, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return mask_inv


def detect_roi(path, background):

    video = cv2.VideoCapture(path)
    while True:

        ret, frame = video.read()

        if ret is False:
            break

        roi = cv2.absdiff(frame, background)
        magnitude = calc_magnitude(roi)

        table = np.array([((i / 255.0) ** 0.6) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # roi = cv2.LUT(roi, table)
        gamma = cv2.LUT(magnitude, table)

        blur = cv2.bilateralFilter(gamma, 10, 100, 100)
        blur = cv2.bilateralFilter(blur, 10, 100, 100)
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        max_contour = contour_largest(blur)
        result = cv2.bitwise_or(blur, blur, mask=max_contour)

        filled = contour_closing(result)
        result = cv2.bitwise_and(blur, blur, mask=filled)

        cv2.imshow("Magnitude", blur)
        cv2.imshow("Result", result)
        cv2.waitKey(30)

    video.release()


path_video = r'E:\PES\CDSAML\DatasetC\videos\01001fn00.avi'
back_image = calc_mean_background(path_video)
detect_roi(path_video, back_image)
cv2.destroyAllWindows()
