import cv2
import numpy as np
from skimage import feature, img_as_ubyte
from skimage.filters import sobel
from skimage.filters.thresholding import threshold_adaptive


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
        if cv2.contourArea(large) > 2000:
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
        if cv2.contourArea(contours[i]) > 500:
            cv2.drawContours(mask, hull, i, (255, 0, 0), 1, 8)

    mask = contour_closing(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
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


def get_weighted_sum(list_images):

    weighted = list_images[len(list_images) - 1]
    weight = 0.5
    for i in range(1, len(list_images)):
        weighted = cv2.addWeighted(weighted, 1.5, list_images[len(list_images) - (i + 1)], weight, 0)
        weight -= 0.02

    return weighted


def threshold_image(image):

    random_values = np.random.randint(low=85, high=200, size=(1, 10), dtype=np.uint8).tolist()
    random_values[0].sort()
    print(random_values)
    thr_images = []
    for i in random_values[0]:
        temp = image.copy()
        temp[temp < i] = 0
        thr_images.append(temp)
        del temp

    weight_image = get_weighted_sum(thr_images)
    final = cv2.morphologyEx(weight_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                             iterations=2)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    hull = convex_hull(final)
    person = cv2.bitwise_or(weight_image, weight_image, mask=hull)
    cv2.imshow("person", person)
    cv2.imshow("Weighted", weight_image)
    cv2.waitKey(0)


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
    fgbg = cv2.createBackgroundSubtractorMOG2()

    video = cv2.VideoCapture(path)

    while True:

        ret, frame = video.read()

        if ret is False:
            break

        diamond_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        circle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        table = np.array([((i / 255.0) ** 0.6) * 255 for i in np.arange(0, 256)]).astype("uint8")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.absdiff(frame, background)
        magnitude = calc_magnitude(roi, 1)

        # gamma = cv2.LUT(magnitude, table)
        # blur = cv2.bilateralFilter(gamma, 5, 100, 100)
        # blur = cv2.bilateralFilter(blur, 5, 100, 100)

        # max_contour = contour_largest(blur)
        max_contour = contour_largest(magnitude)
        # result = cv2.bitwise_or(blur, blur, mask=max_contour)
        result = cv2.bitwise_or(magnitude, magnitude, mask=max_contour)

        result = cv2.dilate(result, diamond_kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        result = cv2.erode(result, circle_kernel)

        filled = contour_closing(result)
        result = cv2.bitwise_and(magnitude, magnitude, mask=filled)
        threshold_image(result)
        # result_1 = result.copy()
        # result_2 = result.copy()
        #
        # result[result < 200] = 0 # This value
        # result_1[result_1 < 100] = 0
        # result_2[result_2 < 30] = 0
        #
        # final = cv2.addWeighted(result, 1.5, result_1, 0.5, 0)
        # final = cv2.addWeighted(final, 1.5, result_2, 0.2, 0)
        # final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        # final = cv2.morphologyEx(final, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # hull = convex_hull(result_1)
        #
        # cropped = cv2.bitwise_or(result, result, mask=hull)
        # cropped_1 = cv2.bitwise_or(result_1, result_1, mask=hull)
        # cropped_2 = cv2.bitwise_or(result_2, result_2, mask=hull)
        #
        #
        # blurred_result = cv2.GaussianBlur(contour_closing(result), (3, 3), 0)
        # blurred_result = cv2.cvtColor(blurred_result, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("Hull", hull)
        # cv2.imshow("Cropped", cropped)
        # cv2.imshow("Cropped_1", cropped_1)
        # cv2.imshow("Cropped_2", cropped_2)
        # cv2.waitKey(0)

    video.release()


video_path = r'E:\PES\CDSAML\DatasetC\videos\01001fn00.avi'
sil_path = r'E:\Softwares\PyCharm\PyCharm Community Edition 2018.2.4\Projects\Gait Analysis\silhouettes\050\fn00'
median_value = median_image(video_path)
detect_roi(video_path, median_value)
cv2.destroyAllWindows()
