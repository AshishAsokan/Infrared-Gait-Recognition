import numpy as np
from multiprocessing import Pool
import cv2
from PIL import Image


def median_calculate(frames_list):
    return np.median(frames_list[:, :, :, 0], axis=0), np.median(frames_list[:, :, :, 1], axis=0), np.median(frames_list[:, :, :, 2], axis=0)


def temporal_median_filter(frames, frame_offset, simultaneous_frames, height, width, total_frames):

    # Initialising size and setting size of median array
    size = frame_offset + simultaneous_frames + frame_offset
    median = np.zeros((size, height, width, 3), np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_write = cv2.VideoWriter('Median.mp4', fourcc, 25.0, (width, height))

    # Initialising median array with random
    for i in range(frame_offset):
        median[i, :, :, :] = np.random.randint(low=0, high=255, size=(height, width, 3))

    for i in range(simultaneous_frames + frame_offset):
        next_image = np.array(frames[i], np.uint8)
        median[frame_offset + i, :, :, :] = next_image
        del next_image

    process_obj = Pool(processes=8)
    current_frame = 0
    filter_frames = np.zeros((simultaneous_frames, height, width, 3), np.uint8)

    while current_frame < total_frames:
        if current_frame == 0:
            pass

        else:
            median = np.roll(median, -simultaneous_frames, axis=0)

            # Generating simultaneous frames for calculation
            for value in range(simultaneous_frames):
                if (current_frame + frame_offset + value) >= total_frames:
                    next_image = np.random.randint(low=0, high=255, size=(height, width, 3))

                else:
                    next_image = frames[current_frame + frame_offset + value]
                    next_image = np.array(next_image, np.uint8)

                median[frame_offset + frame_offset + value, :, :, :] = next_image

        temp_list = []
        for value in range(simultaneous_frames):
            if (value + simultaneous_frames) > total_frames:
                break

            else:
                temp_list.append(median[value: (value + frame_offset + frame_offset)])

        result = process_obj.map(median_calculate, temp_list)

        for i in range(len(result)):
            filter_frames[i, :, :, 0] = result[i][0]
            filter_frames[i, :, :, 1] = result[i][1]
            filter_frames[i, :, :, 2] = result[i][2]

            result_image = Image.fromarray(filter_frames[i, :, :, :])
            result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            video_write.write(result_image)

        current_frame += simultaneous_frames

    video_write.release()



