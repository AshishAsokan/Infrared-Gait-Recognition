import cv2
import math
import numpy as np
import pywt
from scipy.spatial import distance as dist


def draw_rectangle(reading, kernel, mode):

    """
    Draws a bounding rectangle using contours for the given image
    :param  reading: image for which bounding rectangle is to be drawn
    :param  mode: Mode for processing of contours (feet or person)
    :param  kernel: kernel for morphological operations
    :return return_list; returns rectangle features or centroids depending on mode
    :return reading; final image with bounding rectangle
    """

    return_list = []
    empty_list = True
    # Converting frame to gray scale
    frame = cv2.cvtColor(reading, cv2.COLOR_BGR2GRAY)

    if mode == 'person':
        dilate = cv2.dilate(frame, kernel)
    else:
        dilate = frame

    # Finding contours for dilated image
    cont, hier = cv2.findContours(dilate.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hier is None:
        if mode == 'person':
            return reading, return_list, empty_list
        else:
            return return_list, empty_list

    else:
        empty_list = False

    for con, hi in zip(cont, hier[0]):
        if hi[3] != -1:
            cv2.drawContours(dilate, [con], 0, (0, 255, 0),  1)

    # Performing Erosion and morphological opening (erosion followed by dilation)
    image = cv2.erode(dilate, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Finding contours of processed image
    cont, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop for all contours in the final image
    for c in cont:

        # Getting coordinates of bounding rectangle for each contour
        if mode == 'person':
            x, y, w, h = cv2.boundingRect(c)
            return_list = [w, h, x, y]
            cv2.rectangle(reading, (x, y), (x + w, y + h), (0, 255, 0), 2)

        else:
            mom = cv2.moments(c)
            c_x = int(mom["m10"] / mom["m00"])
            c_y = int(mom["m01"] / mom["m00"])
            return_list.append((c_x, c_y))

    if mode == 'person':
        return reading, return_list, empty_list
    else:
        return return_list, empty_list


def calc_stride_dist(image):

    """
    Calculates the distance between the 2 feet for each input image
    :param image: Input image for distance calculation
    :return: distance; The distance between 2 feet
    """

    # Generating a mask for lower part of gait sequence
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[170:240, 0:320] = 255
    image_and = cv2.bitwise_and(image, image, mask=mask)

    image_and = cv2.erode(image_and, np.ones((4, 4), np.uint8), iterations=1)

    # Obtaining size of rectangles for input image
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # result = watershed_segmentation(image_and)
    centers, empty_list = draw_rectangle(image_and, ker, 'feet')

    if empty_list:
        return 1

    if len(centers) > 1:
        distance = dist.euclidean(centers[0], centers[1])
    else:
        distance = 1

    return distance


def calc_gait_cycle(cap):

    """
    Estimates gait cycles depending on step length
    :param   cap: Video Capture object for video sequence
    :return: step_count; number of steps in given gait sequence
    :return: gait_sizes; nested list for size of rectangles in a gait step
    :return: n_size; number of frames in each gait steps
    :return: step_lengths; list of distances between the 2 feet for each step
    """

    step_count = 0          # Number of gait steps in video sample
    threshold = 1           # Threshold value for step length
    coefficients = []       # List for storing haar wavelet coefficients for each cycle
    coeff_temp = []         # List for storing haar wavelet coefficients for each frame
    gait_sizes = []         # Nested list that stores sizes of rectangles for gait cycles
    cycle_sizes = []        # Stores size of rectangles for each gait cycle
    step_lengths = []       # List that stores step length for each gait cycle of each frame
    frame_count = 0         # Number of frames in each gait cycle
    frame_count_video = 0
    n_size = []             # List for storing number of frames in each gait cycle
    gap_legs = []           # List for storing gap between feet in each frame

    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:

        # Reading each frame from the video
        ret, reading = cap.read()

        if ret is False:
            break

        frame_count += 1
        frame_count_video += 1

        temp = cv2.cvtColor(reading, cv2.COLOR_BGR2GRAY)

        if frame_count_video <= 70:
            c_a, (c_h, c_v, c_d) = pywt.dwt2(temp, 'haar')
            coeff_temp.append((c_a, c_h, c_v, c_d))

        # Calculating step length and rectangle dimensions for image
        reading_temp, size_rect, empty_list = draw_rectangle(reading, ker, 'person')

        if empty_list:
            continue

        if len(size_rect) > 0:
            cycle_sizes.append(size_rect)

        stride_dist = calc_stride_dist(reading)

        # Checking if the maximum step length is encountered (ending of one step)
        if (len(gap_legs) > 2 and stride_dist < gap_legs[len(gap_legs) - 1]) and \
                gap_legs[len(gap_legs) - 1] > threshold:

            if step_count is 0:
                threshold = gap_legs[len(gap_legs) - 1]

            gait_sizes.append(cycle_sizes)
            n_size.append(frame_count)
            step_lengths.append(gap_legs)

            if frame_count_video <= 70:
                coefficients.append(coeff_temp)
                coeff_temp = []

            gap_legs = []
            step_count += 1
            cycle_sizes = []

        # Appending stride_length to list
        gap_legs.append(stride_dist)

    del gap_legs, cycle_sizes, threshold, coeff_temp

    return step_count, gait_sizes, n_size, step_lengths, coefficients


def calc_spatial_component(step_count, rect_sizes, num_frames):

    """
    Calculates Spatial Components from extracted binary silhouettes
    :param   step_count: Number of gait cycles
    :param   rect_sizes: Sizes of bounding rectangle for each frame
    :param   num_frames: List of number of frames in each gait cycle
    :return: mean_height; average height of rectangle
    :return: mean_width; average width of rectangle
    :return: mean_angle; average angle of diagonal of rectangle wrt horizontal
    :return: mean_ar; average aspect ratio of rectangle
    """

    # Setting all mean values to 0
    mean_height = 0
    mean_width = 0
    mean_angle = 0
    mean_ar = 0

    # Iterating over each gait step and calculating means
    for i in range(step_count):
        h = sum([element[0] for element in rect_sizes[i]]) / num_frames[i]
        w = sum([element[1] for element in rect_sizes[i]]) / num_frames[i]
        a = sum([math.atan(element[1]/element[0]) for element in rect_sizes[i]]) / num_frames[i]
        ar = sum([element[1]/element[0] for element in rect_sizes[i]]) / num_frames[i]

        # Adding cycle wise averages to overall average
        mean_height += h
        mean_width += w
        mean_angle += a
        mean_ar += ar

    # Returning average values of spatial features
    if step_count == 0:
        step_count = 1

    return [mean_height * 2/step_count, mean_width * 2/step_count, mean_angle * 2/step_count, mean_ar * 2/step_count]


def calc_temporal_component(step_lengths):

    """
    Calculates the temporal components from given step lengths
    :param   step_lengths: Distance between both feet in each frame of each step
    :return: step_len; mean step length from all steps
    :return: stride_len; mean stride length from all frames
    :return: mean_cadence; mean value of cadence in
    :return: velocity; velocity of subject in length/sec
    """

    # Cadence = number of steps / minute
    # Velocity = stride length * 0.5 * cadence

    step_len = 0
    mean_cadence = 0

    for i in range(len(step_lengths)):

        step_len += max(step_lengths[i])            # Maximum size of step from list
        no_of_frames = len(step_lengths[i])         # Number of frames in each step
        no_of_steps = 1/no_of_frames                # Number of steps in each frame
        cad = no_of_steps * cv2.CAP_PROP_FPS / 60       # Cadence = number of steps in each frame * FPS
        mean_cadence += cad                         # Mean value of cadence

    divide = len(step_lengths)
    if divide == 0:
        divide = 0.5

    mean_cadence /= divide
    step_len /= divide
    stride_len = 2 * step_len
    speed = step_len * 0.5 * mean_cadence

    return [step_len, stride_len, mean_cadence, speed]


def calc_stand_deviation(coefficient_list, frames, mean):

    """
    Calculates the standard deviation given the list of coefficients
    :param   coefficient_list: List of values of each coefficient (1 from low freq, 2 from detailed freq)
    :param   frames: number of frames in each step
    :param   mean: mean of the given coefficients
    :return: standard deviation of the list of coefficients
    """

    constant = math.pow(1/(frames - 1), 0.5)
    list_sum = [math.pow(ele - mean, 2) for ele in coefficient_list]
    sum_squares = np.sum(list_sum, axis=None)
    sd = constant * math.pow(sum_squares, 0.5)
    return sd
