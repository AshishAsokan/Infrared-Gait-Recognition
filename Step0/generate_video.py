
'''
This program generates videos of the gait of a person
using the binary silhouettes of the person

To be run only once to get all the videos of the gait
of the person from one extracted zip file

For re-use unzip another zip file and use the steps marked as [TODO]
'''

# * Modules imported

from glob import glob
import cv2


# * Functions

def get_user():
    '''
    Return 1 or 2 to get_path() depending on if Ashish uses it or Chandratop
    '''

    while True:
        usr = int(input("Ashish[1] / Chandratop[2] : "))

        if usr == 1 or usr == 2:
            return usr

        print("Invalid user...")


def get_path():
    '''
    Returns the location of the folders where
    the frames of the gait of the person are stored
    '''

    usr = get_user()

    # [TODO]: Change the paths below to the new paths of the unzipped folder
    if usr == 1:
        return "E:\\PES\\CDSAML\\GaitDatasetC-silh\\001\\*"
    elif usr == 2:
        return "D:\\GAIT\\Extracted\\001\\001\\*"


def get_images():
    '''
    Generates a list of all the frames joined
    together that will be processed into .mp4 video
    '''

    images = list()

    for path in contents:
        walking = []
        all_frames = glob(path + r'\*.png')

        for frame in all_frames:
            frame_image = cv2.imread(frame)
            if frame_image is None:
                break

            walking.append(frame_image)

        images.append(walking)

    return images


def generate_video(images):
    '''
    Generates the .mp4 video from the list of images
    '''

    width, height, char = images[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    length = len(images)

    for i in range(length):
        out_path = origin_path[:-1] + 'Walk' + str(i + 1) + '.mp4'
        video = cv2.VideoWriter(out_path, fourcc, 20.0, (height, width))

        for j in images[i]:
            video.write(j)

    video.release()

# The path where the still frames are stored
origin_path = get_path()

# The list of all the folders inside which the frames are stored
contents = glob(origin_path)

# List of the images
images = get_images()

# Creating a video from images by joining them
generate_video(images)

cv2.destroyAllWindows()
