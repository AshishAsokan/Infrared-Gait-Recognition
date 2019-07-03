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
    # ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], np.uint8)

    if len(contours) > 0:
        large = max(contours, key=cv2.contourArea)
        if cv2.contourArea(large) > 200:
            cv2.drawContours(mask, [large], 0, (255, 255, 255), 2)

    return mask


def convex_hull(image):

    image = cv2.GaussianBlur(image, (5, 5), 3)
    # ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], np.uint8)

    hull = []
    for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 3000:
            cv2.drawContours(mask, contours, i, (255, 0, 0), 1, 8, hierarchy)
            cv2.drawContours(mask, hull, i, (255, 0, 0), 1, 8)

    return mask


def calc_magnitude(image, order):

    image = cv2.GaussianBlur(image, (3, 3), 0)
    # Calculating image gradients using Sobel derivative
    derivative_x = cv2.Sobel(image, cv2.CV_64F, order, 0)
    derivative_y = cv2.Sobel(image, cv2.CV_64F, 0, order)

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

    while True:

        ret, frame = video.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.absdiff(frame, background)

        magnitude_roi = calc_magnitude(roi, 1)
        magnitude = magnitude_roi.copy()
        magnitude[magnitude < 160] = 0

        masked_image = convex_hull(cv2.dilate(magnitude, circle_kernel))
        filled_convex = contour_closing(masked_image)
        result = cv2.bitwise_and(magnitude_roi, magnitude_roi, mask=filled_convex)

        gamma = cv2.LUT(result, table)
        blur = cv2.bilateralFilter(gamma, 5, 100, 100)
        blur = cv2.bilateralFilter(blur, 5, 100, 100)

        cv2.imshow("Blur", masked_image)
        cv2.imshow("Filled", blur)
        cv2.waitKey(0)

    video.release()


video_path = r'E:\PES\CDSAML\DatasetC\videos\01010fn00.avi'
median_value = median_image(video_path)
detect_roi(video_path, median_value)
cv2.destroyAllWindows()
