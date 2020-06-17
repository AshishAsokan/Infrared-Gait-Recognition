import cv2
import skeleton_features as skf
import spatial_temporal as svt
import svm_test as sts
import scikitplot as sckt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
import numpy as np

import glob

# Path to training set
contents = glob.glob(r'Train_videos\*.mp4')

train_vector = []
mean_vector = []
responses_train = []
structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

for path in contents:
    video = cv2.VideoCapture(path)
    print(path)
    train_vector = []
    frame_count = 0

    while True:
        ret, frame = video.read()

        if ret is False:
            break

        rect_frame, rect_size, empty_list = skf.draw_rectangle(frame, structure)

        # Calculating features only for those frames which have a silhouette
        if (empty_list is False and len(rect_size) > 0) and rect_size[1] > 100:
            frame_count += 1

            # Fitting a skeleton
            points_values = skf.fit_skeleton(rect_size[2], rect_size[3], rect_size[0], rect_size[1], frame)

            # Calculating all the features
            distance_parameters = skf.distance_parameters(points_values)
            angle_parameters = skf.angle_parameters(points_values)
            spatial_parameters = skf.spatial_features(points_values)
            feature_vector = angle_parameters + distance_parameters
            train_vector.append(feature_vector)

        if frame_count == 70:
            break

    # Averaging the data for 1 subject
    train_vector = np.mean(train_vector, axis=0)

    # Calculating spatio-temporal features
    spatial_temporal_parameters = svt.spatial_temporal_features(path)

    # Combining the features
    train_vector = train_vector.tolist() + spatial_temporal_parameters
    mean_vector.append(train_vector)

    # Setting the label for the subject
    label = int(path[(len(path) - 11): (len(path) - 8)])
    responses_train.append(label)

mean_vector = np.array(mean_vector)
print(mean_vector.shape)
responses_train = np.array(responses_train)
mean_vector = preprocessing.scale(mean_vector)

# Creating SVM model
clf = svm.SVC(C=1000.0, gamma=0.005)
print("Model Created")

clf.fit(mean_vector, responses_train)
print("Data fitted")
print("Training Successful")

test_vector, responses_test = sts.calc_test_features()
test_vector = preprocessing.scale(test_vector)
y_test = clf.predict(test_vector)

for i in range(len(responses_test)):
    print(responses_test[i], y_test[i])
print("Accuracy:", metrics.accuracy_score(responses_test, y_test) * 100, "%")
sckt.estimators.plot_learning_curve(clf, mean_vector, responses_train)
plt.show()








