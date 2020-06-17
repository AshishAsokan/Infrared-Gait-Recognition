import cv2
import numpy as np
from skimage import feature, img_as_ubyte
from skimage.filters import sobel
from skimage.filters.thresholding import threshold_adaptive
import glob


def median_image(path):

    """
    Calculates the median image based on the number of images
    :param      path: Path of the infrared video
    :return:    frame: The median image when n value is encountered
    :return:    n_frames: Number of frames read from video
    """

    # Initializing video capture object
    video_obj = cv2.VideoCapture(path)

    # Details of the video frames
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

    # Finding Median of the images
    result = np.median(images, axis=2)
    result = np.uint8(result)
    return result


def contour_largest(image):

    """
    Determines the largest contour from the given image
    :param      image: Input Image
    :return:    mask: Image with the largest contour drawn
    """

    image = cv2.GaussianBlur(image, (5, 5), 3)
    # ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    # Detecting contours of the input image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Creating the output image as an array of 0s
    mask = np.zeros(image.shape[:2], np.uint8)

    # If there are contours in the image
    if len(contours) > 0:

        # Find the largest contour of the specific size
        large = max(contours, key=cv2.contourArea)
        if cv2.contourArea(large) > 2000:
            cv2.drawContours(mask, [large], 0, (255, 255, 255), 2)

    return mask


def convex_hull(image, thresh_size):

    """
    Constructs a convex hull for a given contour
    :param      image: Input Image 
    :param      thresh_size: Threshold size for the contour area
    :return:    mask: Final image with convex hull constructed
    """

    image = cv2.GaussianBlur(image, (5, 5), 3)
    # ret, thresh = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)

    # Detecting contours and initializing output image
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], np.uint8)

    hull = []
    for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))

    # Drawing the convex hull contours onto the final image
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > thresh_size:
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
        weighted = cv2.addWeighted(weighted, 1.5, list_images[len(list_images) - (i + 1)], weight, -5)
        weight -= 0.1

    return weighted


def remove_noise(image):

    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    leg_distance = 180
    mask = np.zeros(image.shape[:2], np.uint8)
    if len(contours) > 0:
        flat = lambda l: [item for sublist in l for item in sublist]
        flat_contours = flat(flat(contours))
        flat_contours = [element.tolist() for element in flat_contours]

        flat_contours.sort(key=lambda x: x[0])
        flat_contours = [i for i in flat_contours if i[1] <= (leg_distance + 20)]
        # left_point_1 = tuple([flat_contours[0][0], flat_contours[0][1] + 5])
        # left_point_2 = tuple([flat_contours[0][0] + 20, flat_contours[0][1] - 10])
        left_point_1 = tuple([flat_contours[0][0], leg_distance])
        left_point_2 = tuple([flat_contours[0][0] + 23, leg_distance + 17])

        right_point_1 = tuple([flat_contours[len(flat_contours) - 1][0], leg_distance])
        right_point_2 = tuple([flat_contours[len(flat_contours) - 1][0] - 23, leg_distance + 17])

        cv2.rectangle(mask, left_point_1, left_point_2, (255, 255, 255), cv2.FILLED)
        cv2.rectangle(mask, right_point_1, right_point_2, (255, 255, 255), cv2.FILLED)
    return mask


def threshold_image(image, mean_value):

    if mean_value < 55:
        low_range = 85
        high_range = 180

    else:
        low_range = 150
        high_range = 255

    contour_size = int(18 * mean_value)
    random_values = np.random.randint(low=low_range, high=high_range, size=(1, 5), dtype=np.uint8).tolist()
    random_values[0].sort()
    thr_images = []
    for i in random_values[0]:
        temp = image.copy()
        temp[temp < i] = 0
        thr_images.append(temp)
        del temp

    weight_image = get_weighted_sum(thr_images)
    final = cv2.morphologyEx(weight_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                             iterations=3)
    final = cv2.morphologyEx(final, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    hull = convex_hull(final, contour_size)
    person = cv2.bitwise_or(weight_image, weight_image, mask=hull)
    eroded_person = cv2.erode(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    upper = person.copy()
    upper[180:, :] = 0

    blur = cv2.GaussianBlur(eroded_person, (5, 5), 0)
    blur = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
    feet = blur.copy()
    feet[:180, :] = 0
    eroded = cv2.erode(feet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

    result = remove_noise(eroded)
    feet = cv2.bitwise_or(person, person, mask=result)
    final = cv2.add(feet, upper)
    final = cv2.bilateralFilter(final, 15, 100, 100)
    # final = cv2.GaussianBlur(final, (5, 5), 0)
    final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), iterations=1)

    return final


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
    video = cv2.VideoCapture(path)

    height, width = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_write = cv2.VideoWriter('video_binary.mp4', fourcc, 25.0, (width, height))

    while True:

        ret, frame = video.read()

        if ret is False:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(frame)
        roi = cv2.absdiff(frame, background)
        magnitude = calc_magnitude(roi, 1)

        # max_contour = contour_largest(blur)
        max_contour = contour_largest(magnitude)
        # result = cv2.bitwise_or(blur, blur, mask=max_contour)
        result = cv2.bitwise_or(magnitude, magnitude, mask=max_contour)

        result = cv2.dilate(result, diamond_kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, circle_kernel, iterations=2)
        result = cv2.erode(result, circle_kernel)

        filled = contour_closing(result)
        result = cv2.bitwise_and(magnitude, magnitude, mask=filled)
        processed_frame = threshold_image(result, mean_bright)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        video_write.write(processed_frame)

    video.release()
    video_write.release()

# Generating the binary silhouette video sequence for each gait video
# Dataset is the folder containing all the videos
contents = glob.glob(r'Dataset\*.mp4')

for path in contents:

    video_path = path
    median_value = median_image(video_path)
    detect_roi(video_path, median_value)

cv2.destroyAllWindows()
