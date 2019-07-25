import cv2
import skeleton_features as skf
import spatial_temporal as svt
import numpy as np
import glob


def calc_test_features():
    # contents = glob.glob(r'E:\PES\CDSAML\Gait_IR\VIDS\Train\*.mp4')
    contents = glob.glob(r'E:\Softwares\PyCharm\PyCharm Community Edition 2018.2.4\Projects\Gait Analysis\Test_videos\*.mp4')
    responses_test = []
    mean_test = []
    print("\nTesting:\n")
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for path in contents:
        video = cv2.VideoCapture(path)
        print(path)
        test_vector = []
        frame_count = 0
    
        while True:
            ret, frame = video.read()
    
            if ret is False:
                break
    
            rect_frame, rect_size, empty_list = skf.draw_rectangle(frame, structure)
            if (empty_list is False and len(rect_size) > 0) and rect_size[1] > 100:
                frame_count += 1
                points_values = skf.fit_skeleton(rect_size[2], rect_size[3], rect_size[0], rect_size[1], frame)

                distance_parameters = skf.distance_parameters(points_values)
                angle_parameters = skf.angle_parameters(points_values)

                feature_vector = angle_parameters + distance_parameters
                test_vector.append(feature_vector)
    
            if frame_count == 70:
                break

        test_vector = np.mean(test_vector, axis=0)
        spatial_temporal_parameters = svt.spatial_temporal_features(path)
        test_vector = test_vector.tolist() + spatial_temporal_parameters
        # test_vector = spatial_temporal_parameters
        mean_test.append(test_vector)
        label = int(path[(len(path) - 11): (len(path) - 8)])
        responses_test.append(label)

    mean_test = np.array(mean_test)
    responses_test = np.array(responses_test)
    return mean_test, responses_test
