import cv2
import numpy as np
import math
from scipy.spatial import distance as dist


def draw_rectangle(reading, kernel):

    """
    Draws a bounding rectangle using contours for the given image
    :param  reading: image for which bounding rectangle is to be drawn
    :param  kernel: kernel for morphological operations
    :return return_list; returns rectangle features or centroids depending on mode
    :return reading; final image with bounding rectangle
    """

    return_list = []
    list_empty = True
    # Converting frame to gray scale
    dilate = cv2.cvtColor(reading, cv2.COLOR_BGR2GRAY)

    # dilate = cv2.dilate(dilate, kernel)

    # Finding contours for dilated image
    cont, hier = cv2.findContours(dilate.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hier is None:
        return reading, return_list, list_empty
    else:
        list_empty = False

    for con, hi in zip(cont, hier[0]):
        if hi[3] != -1:
            cv2.drawContours(dilate, [con], 0, (0, 255, 0),  1)

    # Performing Erosion and morphological opening (erosion followed by dilation)
    # image = cv2.erode(dilate, kernel)
    image = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel)
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Finding contours of processed image
    cont, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop for all contours in the final image
    for c in cont:

        # Getting coordinates of bounding rectangle for each contour
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            return_list = [w, h, x, y]

    return reading, return_list, list_empty


def find_body_part(image, lower_height=0, upper_height=0):

    """
    Finds coordinates of endpoints for given body part
    :param image:         Input frame from video in grayscale
    :param lower_height:  Lower range of height for body part
    :param upper_height:  Upper limit of height for body part
    :return: left_point:  Coordinates of left most point for body part
    :return: right_point: Coordinates of right most point for body part
    """

    mask = np.ones(image.shape[:2], np.uint8) * 255

    if upper_height is 0:
        mask[0:lower_height - 3, :] = 0
        mask[lower_height + 3:, :] = 0

    else:
        mask[0:lower_height - 3, :] = 0
        mask[upper_height + 3:, :] = 0

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)

    contours, _ = cv2.findContours(result, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    flat = lambda l: [item for sublist in l for item in sublist]
    flat_contours = flat(flat(contours))
    flat_contours = [element.tolist() for element in flat_contours]

    flat_contours.sort(key=lambda x: x[0])
    left_point = flat_contours[0]
    right_point = flat_contours[len(flat_contours) - 1]
    return left_point, right_point


def fit_skeleton(x, y, w, h, image):

    """
    Fits a skeleton to the person in the given frame
    :param x: X coordinate of bounding rectangle
    :param y: Y coordinate of bounding rectangle
    :param w: width of bounding rectangle
    :param h: height of bounding rectangle
    :param image: Input frame from video
    :return:
    """

    neck_height = int(0.13 * h + y)
    # head_height = int(0.05 * h + y)
    lower_arm_range = int(0.45 * h + y)
    upper_arm_range = int(0.52 * h + y)
    waist_height = int(0.47 * h + y)
    knee_height = int(0.73 * h + y)
    feet_height = int(0.93 * h + y)
    skeleton = np.zeros(image.shape[:2], np.uint8)

    neck_point = (int(x + w/2), neck_height)
    head_point = (int(x + w/2), y)
    waist_point = (int(x + w/2), waist_height)
    image = cv2.erode(image, cv2.getStructuringElement(cv2.MORPH_CROSS, (8, 8)))

    knee_points = find_body_part(image, knee_height)
    feet_points = find_body_part(image, feet_height)
    arm_points = find_body_part(image, lower_arm_range, upper_arm_range)

    points = {"Knees": knee_points, "Feet": feet_points, "Arms": arm_points,
              "Head": (head_point, ), "Neck": (neck_point, ), "Waist": (waist_point, ), "Rectangle": (x, y, w, h)}

    for pt in points.keys():
        if pt is not "Rectangle":
            for i in points[pt]:
                cv2.circle(skeleton, tuple(i), 1, (255, 255, 255), 3)

    cv2.line(skeleton, head_point, neck_point, (255, 255, 255), 1)
    cv2.line(skeleton, neck_point, tuple(arm_points[0]), (255, 255, 255), 1)
    cv2.line(skeleton, neck_point, tuple(arm_points[1]), (255, 255, 255), 1)
    cv2.line(skeleton, neck_point, waist_point, (255, 255, 255), 1)
    cv2.line(skeleton, waist_point, tuple(knee_points[0]), (255, 255, 255), 1)
    cv2.line(skeleton, waist_point, tuple(knee_points[1]), (255, 255, 255), 1)
    cv2.line(skeleton, tuple(knee_points[0]), tuple(feet_points[0]), (255, 255, 255), 1)
    cv2.line(skeleton, tuple(knee_points[1]), tuple(feet_points[1]), (255, 255, 255), 1)
    # cv2.imshow("Skeleton", skeleton)
    # cv2.imshow("Person", image)
    # cv2.waitKey(80)
    return points


def angle_parameters(points_dict):

    parameters = []

    # 1. Angle between front thigh and vertical
    vector_1 = np.array(points_dict["Waist"][0]) - np.array(points_dict["Knees"][1])
    vector_2 = np.array(points_dict["Neck"][0]) - np.array(points_dict["Waist"][0])
    angle_thigh_front = np.degrees(np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2)))
    parameters.append(angle_thigh_front)

    # 2. Angle between rear thigh and vertical
    vector_1 = np.array(points_dict["Waist"][0]) - np.array(points_dict["Knees"][0])
    vector_2 = np.array(points_dict["Neck"][0]) - np.array(points_dict["Waist"][0])
    angle_thigh_rear = np.degrees(np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2)))
    parameters.append(angle_thigh_rear)

    # 3. Angle between front shin and vertical
    vector_1 = np.array(points_dict["Knees"][1]) - np.array(points_dict["Feet"][1])
    vector_2 = np.array(points_dict["Neck"][0]) - np.array(points_dict["Waist"][0])
    angle = np.degrees(np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2)))
    parameters.append(abs(angle))

    # 4. Angle between rear shin and vertical
    vector_1 = np.array(points_dict["Knees"][0]) - np.array(points_dict["Feet"][0])
    vector_2 = np.array(points_dict["Neck"][0]) - np.array(points_dict["Waist"][0])
    angle = np.degrees(np.math.atan2(np.linalg.det([vector_1, vector_2]), np.dot(vector_1, vector_2)))
    parameters.append(abs(angle))

    return parameters


def spatial_features(points_dict):

    parameters = []
    # 1. Distance between feet of person
    distance = abs(dist.euclidean(points_dict["Feet"][1], points_dict["Feet"][0]))
    parameters.append(distance)
    del distance

    # 2. Height of person
    distance = abs(points_dict["Head"][0][0] - points_dict["Feet"][0][0])
    parameters.append(distance)
    del distance

    # 3: Width of person
    distance = abs(dist.euclidean(points_dict["Arms"][1], points_dict["Arms"][0]))
    parameters.append(distance)
    del distance

    # 4. Angle of person
    angle = np.degrees(np.math.atan(points_dict["Rectangle"][3] / points_dict["Rectangle"][2]))
    parameters.append(angle)
    del angle

    # 5. Aspect Ratio
    aspect_ratio = points_dict["Rectangle"][3] / points_dict["Rectangle"][2]
    parameters.append(aspect_ratio)
    del aspect_ratio

    return parameters


def distance_parameters(points_dict):

    """
    Calculates distance parameters from generated skeleton
    :param points_dict: Dictionary of points 
    :return: parameters: List containing 12 distance parameters
    """
    parameters = []

    # 1. Horizontal distance of front hand from centroid
    distance = abs(points_dict["Arms"][1][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 2. Vertical distance of front hand from centroid
    distance = abs(points_dict["Arms"][1][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 3. Horizontal distance of rear hand from centroid
    distance = abs(points_dict["Arms"][0][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 4. Vertical distance of rear hand from centroid
    distance = abs(points_dict["Arms"][0][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 5. Horizontal distance of front knee from centroid
    distance = abs(points_dict["Knees"][1][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 6. Vertical distance of front knee from centroid
    distance = abs(points_dict["Knees"][1][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 7. Horizontal distance of rear knee from centroid
    distance = abs(points_dict["Knees"][0][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 8. Horizontal distance of rear knee from centroid
    distance = abs(points_dict["Knees"][0][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 9. Horizontal distance of front foot from centroid
    distance = abs(points_dict["Feet"][1][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 10. Vertical distance of front foot from centroid
    distance = abs(points_dict["Feet"][1][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 11. Horizontal distance of rear foot from centroid
    distance = abs(points_dict["Feet"][0][0] - points_dict["Waist"][0][0])
    parameters.append(distance)
    del distance

    # 12. Vertical distance of rear foot from centroid
    distance = abs(points_dict["Feet"][0][1] - points_dict["Waist"][0][1])
    parameters.append(distance)
    del distance

    # 13. Horizontal Distance between feet of person
    distance = abs(points_dict["Feet"][1][0] - points_dict["Feet"][0][0])
    parameters.append(distance)
    del distance

    # 14. Vertical Distance between feet of person
    distance = abs(points_dict["Feet"][0][1] - points_dict["Feet"][1][1])
    parameters.append(distance)
    del distance


    return parameters


# video = cv2.VideoCapture(Insert video path here to test)
# structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# train_vector = []
# frame_count = 0
#
# while True:
#     ret, frame = video.read()
#
#     if ret is False:
#         break
#
#     rect_frame, rect_size, empty_list = draw_rectangle(frame, structure)
#     if (empty_list is False and len(rect_size) > 0) and rect_size[1] > 100:
#         frame_count += 1
#         points_values = fit_skeleton(rect_size[2], rect_size[3], rect_size[0], rect_size[1], frame)
#         feature_vector = distance_parameters(points_values) + angle_parameters(points_values)
#         train_vector.append(feature_vector)
#
#     if frame_count == 80:
#         break
#
# train_vector = np.array(train_vector)
# mean_vector = np.mean(train_vector, axis=0)
# print(mean_vector)
# video.release()
# cv2.destroyAllWindows()





