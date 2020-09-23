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
def printDivisors(n) :
    i = 1
    while i <= n :
        if (n % i==0) :
            print(i)
        i = i + 1


def read_image_directory(directorypath):
    files = [f for f in listdir(directorypath) if isfile(join(directorypath, f))]
    return files

# this function is susceptible to adversial attacks as how IL and OPencv
# works in reading the images.
def prepare_cv_image_2_keras_image (img):
    # convert the color from BGR to RGB then convert to PIL array
    cvt_image =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cvt_image)

    # resize the array (image) then PIL image
    im_resized = im_pil.resize((224, 224))
    img_array = image.img_to_array(im_resized)
    image_array_expanded = np.expand_dims(img_array, axis = 0)
    return preprocess_input(image_array_expanded)

#load model
model = load_model('mobilenet_model_voc.h5')

# image directory path
#image_directory = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/voc_validation/car/2008_000027.jpg'
image_directory = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/voc_validation/car/2008_002875.jpg'
# image_files = read_image_directory(image_directory)
#
# for images in image_files:
#     img = cv2.imread(image_directory+images)
#     img = cv2.resize(img, (224, 224))
#     pred = model.predict(img)
#     print(pred)


img = image.load_img(image_directory, target_size=(224, 224))
x = image.img_to_array(img)
x= np.expand_dims(x, axis=0)
x= preprocess_input(x)


img1 = cv2.imread(image_directory)
img1 = prepare_cv_image_2_keras_image(img1)

# finally check if it is working
# if np.array_equal(z, img):
#     print('TRUE>>>>>>>>>>>>>>>')

pred = model.predict(x)
pred1 = model.predict(img1)
print(pred, pred1)





