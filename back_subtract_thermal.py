import cv2
import numpy as np
import math


def median_image(path):

    """
    Calculates the median image based on the number of images
    :param      path: Path of the infrared video
    :return:    frame: The median image when n value is encountered
    :return:    n_frames: Number of frames read from video
    """

    video_obj = cv2.VideoCapture(path)

    # Reading lesser number of frames (3 fps)
    fps = video_obj.get(cv2.CAP_PROP_FPS)
    time = video_obj.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    n_frames = int(int((fps - 3) * time))

    # Calculating value of median
    if n_frames % 2 == 1:
        n_value = int((n_frames + 1)/2)

    else:
        n_value = int(n_frames/2)

    # Maintaining a frame count to return median image
    frame_count = 0
    while video_obj.isOpened():
        ret, frame = video_obj.read()

        if ret is False:
            break

        # If frame count = median value, return the current frame
        frame_count += 1
        if frame_count == n_value:
            video_obj.release()
            return frame, n_frames


def calc_mean(path, median, sd):

    """
    Calculates the pixel wise mean of the N frames captured
    :param path:    Path of the video file
    :param median:  Median image from the N images
    :param sd:      Value of standard deviation
    :return:        mean; pixel wise mean of images
    """

    # Initialising mean and weights to 0
    mean = np.zeros((240, 320, 3))
    weights = np.zeros((240, 320, 3))

    # Creating video capture object
    video_obj = cv2.VideoCapture(path)

    while True:
        ret, frame = video_obj.read()

        if ret is False:
            break

        # Calculating difference and squaring
        diff = np.subtract(frame, median)
        term = np.square(diff)

        # Calculating exponent term and finding pixel wise exponent of image
        term = term / (-2 * math.pow(sd, 2))
        w = np.exp(term)

        # Adding terms to mean and weights arrays
        mean = np.add(mean, np.multiply(w, frame))
        weights = np.add(weights, w)

    # Calculating and return mean array
    mean = np.divide(mean, weights)
    video_obj.release()
    return mean


def calc_variance(path, median, no_of_frames, sd, mean):

    """
    Calculates pixel wise variance based on input images
    :param path: Path of input images
    :param median: Median of N images calculated
    :param no_of_frames: N images read from video
    :param sd: Standard deviation value
    :param mean: Mean image of N images
    :return: variance; pixel wise variance of the images
    """

    # Initialising variance and weights arrays to 0
    variance = np.zeros((240, 320, 3))
    weights = np.zeros((240, 320, 3))

    # Creating video capture object
    video_obj = cv2.VideoCapture(path)

    while True:
        ret, frame = video_obj.read()

        if ret is False:
            break

        # Calculating difference terms for weights and variance
        diff = np.subtract(frame, median)
        diff_variance = np.subtract(frame, mean)

        # Squaring calculated differences
        term = np.square(diff)
        term_variance = np.square(diff_variance)

        # Initialising term for pixel wise exponential
        term = term / (-2 * math.pow(sd, 2))
        w = np.exp(term)

        # Adding terms to the variance and weights arrays
        variance = np.add(variance, np.multiply(w, term_variance))
        weights = np.add(weights, w)

    # Calculating variance based on formula
    variance = (np.divide(variance, weights)) / ((no_of_frames - 1) / no_of_frames)
    video_obj.release()
    return variance


def detect_roi(path, mean, variance, threshold):

    """
    Detects the ROI of input images based on statistical model developed
    :param path: Path of video file
    :param mean: Mean image of processed N images
    :param variance: Variance of processed N images
    :param threshold: Threshold value for mahalanobis distance
    :return:
    """

    # Creating video capture object
    video_obj = cv2.VideoCapture(path)

    # Initialising kernel for dilation operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = video_obj.read()

        if ret is False:
            break

        # Calculating Mahalanobis distance to detect foreground pixels
        with np.errstate(invalid='ignore'):
            frame_roi = np.divide(np.square(np.subtract(frame, mean)), variance)
            frame[np.where((frame_roi < [threshold, threshold, threshold]).all(axis=2))] = [0, 0, 0]

        initial = frame

        # Dilation operation on background subtracted image
        frame = cv2.dilate(frame, kernel)

        # Converting to gray scale for connected components operation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        components, img, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        # Ignoring the background component
        sizes = stats[1:, -1]
        components -= 1

        # Setting threshold value of component size
        min_size = 60

        # Setting all components to 0 when size < 40
        for i in range(0, components):
            if sizes[i] < min_size:
                frame[img == i + 1] = 0

        cv2.imshow("Initial", initial)
        cv2.imshow("Final", frame)
        cv2.waitKey(30)

    video_obj.release()


video_path = r'E:\PES\CDSAML\DatasetC\videos\01001fn00.avi'

median_image, frames = median_image(video_path)

mean_value = calc_mean(video_path, median_image, 6)

variance_value = calc_variance(video_path, median_image, frames, 6, mean_value)

detect_roi(video_path, mean_value, variance_value, 23)
cv2.destroyAllWindows()






