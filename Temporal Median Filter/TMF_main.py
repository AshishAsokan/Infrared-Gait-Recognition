import TMF
import cv2

if __name__ == '__main__':
    # video = cv2.VideoCapture(r'E:\PES\CDSAML\DatasetC\videos\01001fn00.avi')
    video = cv2.VideoCapture('video.avi')
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_frames = [video.read()[1] for i in range(frame_count)]
    print(len(video_frames))
    video.release()
    TMF.temporal_median_filter(video_frames, 50, 20, frame_height, frame_width, frame_count)

    cv2.destroyAllWindows()
