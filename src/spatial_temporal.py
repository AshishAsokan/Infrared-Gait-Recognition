import cv2
import feature_extraction as f_ext
import numpy as np


def spatial_temporal_features(path):

    video = cv2.VideoCapture(path)
    print(path)
    gait_cycle_estimate = f_ext.calc_gait_cycle(video)
    # gait_cycle_estimate[0]     : Number of steps in given gait sequence
    # gait_cycle_estimate[1]     : Size of rectangles in each gait step
    # gait_cycle_estimate[2]     : Number of frames in each step
    # gait_cycle_estimate[3]     : Distance between feet in each frame
    # gait_cycle_estimate[4]     : Haar wavelet coefficients found using 2D-DWT

    spatial_feature_vector = f_ext.calc_spatial_component(gait_cycle_estimate[0], gait_cycle_estimate[1],
                                                          gait_cycle_estimate[2])
    # spatial_feature_vector[0]    : mean height of bounding rectangle
    # spatial_feature_vector[1]    : mean width of bounding rectangle
    # spatial_feature_vector[2]    : mean angle of bounding rectangle
    # spatial_feature_vector[3]    : mean aspect ratio of bounding rectangle

    temporal_feature_vector = f_ext.calc_temporal_component(gait_cycle_estimate[3])

    # temporal_feature_vector[0]   : length of each step
    # temporal_feature_vector[1]   : length of each stride (2 * step length)
    # temporal_feature_vector[2]   : cadence of subject (number of steps per second)
    # temporal_feature_vector[3]   : velocity of subject ( cadence * 0.5 * stride length)

    training_sample = np.concatenate([spatial_feature_vector, temporal_feature_vector], axis=None).tolist()
    return training_sample









