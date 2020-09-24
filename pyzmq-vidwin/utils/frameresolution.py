# Author - Piyush Yadav
# Insight Centre for Data Analytics
# Package- VidWIN Project

# the code gives accuracy for different image resolution on pascal voc- dataset...
# model is retrained with global avergae pooling layer...


from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image


# method to print the divisors
def get_resolution_set(aspect_width, aspect_height, resolution):
    a= resolution[0]
    b = resolution[1]
    width =[]
    height =[]
    while a >= aspect_width:
        if (a % aspect_width==0) :
            width.append(a)
        a = a - aspect_width

    while b >= aspect_height:
        if (b % aspect_height==0) :
            height.append(b)
        b = b - aspect_height

    resolutionlist = list(zip(width, height))
    return resolutionlist


def read_image_directory(directorypath):
    files = [f for f in listdir(directorypath) if isfile(join(directorypath, f))]
    return files

def prepare_keras_image(img, resolution):
    img = image.load_img(img, target_size= resolution)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# this function is susceptible to adversial attacks as how IL and OPencv
# works in reading the images.
def prepare_cv_image_2_keras_image (img, resolution):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)
    # resize the array (image) then PIL image
    im_resized = im_pil.resize(resolution)
    img_array = image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return preprocess_input(image_array_expanded)

def process_image(resolution_list, image_directory, image_files, model):
    for resolution in resolution_list:
        print('RESOLUTION ****', resolution)
        for img in image_files:
            #img = prepare_cv_image_2_keras_image(img, resolution)
            img = prepare_cv_image_2_keras_image(image_directory+img,resolution)
            pred = model.predict(img)
            print(pred)

def process_video(resolution_list, video_path, model):
    for resolution in resolution_list:
        print('RESOLUTION ****', resolution)
        cap = cv2.VideoCapture(video_path)
        # get frame dimension
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = prepare_cv_image_2_keras_image(frame,resolution)
            pred = model.predict(img)
            print(pred)


# set the video path
video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2/Jacksonhole9804.mp4'

# image directory path
image_directory = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/voc_validation/car/'
# get the list of the image files
image_files = read_image_directory(image_directory)
# the get the list of the resolution
resolution_list= get_resolution_set(16,9, (1920,1080))
#load model
model = load_model('mobilenet_model_voc.h5')

#process_image(resolution_list, image_directory,image_files, model)
process_video(resolution_list, video_path, model)