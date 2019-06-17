
'''
This program generates videos of the gait of a person
using the binary silhouettes of the person

To be run only once to get all the videos of the gait
of the person from one extracted zip file

For re-use unzip another zip file and use the steps marked as TODO
'''

#* Modules imported

import cv2
#? cv2 is the module for OpenCV
#? imread — read an image
#? VideoWriter — Write videos [https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html]
#? VideoWriter_fourcc — puts the file encoding format, here MP4V
#? destroyAllWindows — destroys all windows 

from glob import glob
#? glob — Unix style pathname pattern expansion. The glob module finds all the 
#? pathnames matching a specified pattern according to the rules used by the Unix shell


#* Functions

def get_user():
    ''' Return 1 or 2 to get_path() depending on if Ashish uses it or Chandratop '''
    while True:
        x = int(input("Ashish[1] / Chandratop[2] : "))
        if x == 1 or x == 2:
            return x
        print("Invalid user...")

def get_path():
    ''' Returns the location of the folders where the frames of the gait of the person are stored '''
    x = get_user()
    #TODO: Change the paths below to the new paths of the extracted/unzipped folder
    if x == 1:
        return "E:\\PES\\CDSAML\\GaitDatasetC-silh\\001\\*"
    elif x == 2:
        return "D:\\GAIT\\Extracted\\001\\001\\*"

def get_images():
    ''' Generates a list of all the frames joined together that will be processed into .mp4 video '''
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
    ''' Generates the .mp4 video from the list of images '''
    width, height, ch = images[0][0].shape
    del ch
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    for i in range(len(images)):
        out_path = origin_path[:-1] + 'Walk' + str(i + 1) + '.mp4'
        video = cv2.VideoWriter(out_path, fourcc, 20.0, (height, width))
        for j in images[i]:
            video.write(j)
    video.release()


#* MAIN

origin_path = get_path()
contents = glob(origin_path)    # The list of all the folders inside which the frames are stored
images = get_images()
generate_video(images)
cv2.destroyAllWindows()
