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
import matplotlib.pyplot as plt


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

# decode predictions
def decode_predictions(pred,k):
    predict_list =[]
    class_labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    #for top-k score
    # index1= pred[0].argsort()[::-1][: k]
    # print('Index1',index1)
    index = pred[0].argsort()[-k:][::-1]
    #print('Index', index)

    #print("Prdicted label for Image: ", k)
    for i in range(0,k):
        predict_list.append({class_labels[index[i]]: pred[0][index[i]]})
        #print(class_labels[index[i]], " : ", pred[0][index[i]])

    return predict_list

def process_image(resolution_list, image_directory, image_files, model):
    for resolution in resolution_list:
        print('RESOLUTION ****', resolution)
        for img in image_files:
            #img = prepare_cv_image_2_keras_image(img, resolution)
            img = cv2.imread(image_directory+img)
            img = prepare_cv_image_2_keras_image(img, resolution)
            pred = model.predict(img)

def calculate_top_k_accuracy(acc_value):
    top1 =[]
    top2= []
    top3 = []
    top4 =[]
    top5 = []
    for data in acc_value:
        for key, value in data.items():
            if any('car' in d for d in value[:1]):
                top1.append(1)
            if any('car' in d for d in value[:2]):
                top2.append(1)
            if any('car' in d for d in value[:3]):
                top3.append(1)
            if any('car' in d for d in value[:4]):
                top4.append(1)
            if any('car' in d for d in value[:5]):
                top5.append(1)

    top1_accurcacy = len(top1)/len(acc_value)
    top2_accurcacy = len(top2) / len(acc_value)
    top3_accurcacy = len(top3) / len(acc_value)
    top4_accurcacy = len(top4) / len(acc_value)
    top5_accurcacy = len(top5) / len(acc_value)

    return top1_accurcacy, top2_accurcacy, top3_accurcacy, top4_accurcacy, top5_accurcacy


# function to plot live graph
def plot_accuracy(final_resolution):
    batch_x_ticks_res = []
    top_1 =[]
    top_2 =[]
    top_3 =[]
    top_4= []
    top_5= []
    for data in final_resolution:
        for key, value in data.items():
            batch_x_ticks_res.append(str(key))
            a,b,c,d,e = calculate_top_k_accuracy(value)
            top_1.append(a)
            top_2.append(b)
            top_3.append(c)
            top_4.append(d)
            top_5.append(e)

    fig = plt.figure()
    #t_y = np.array([1,5,10,20,30,40,50,60,70,80,90,100])
    t_y = np.array(batch_x_ticks_res)
    plt.plot(t_y, top_1, 'r^--',t_y, top_2, 'mD-',t_y, top_3, '#92D050',t_y, top_4, 'bo--',t_y, top_4, 'b^--')
    plt.legend(['top-1','top-2', 'top-3' , 'top-4', 'top-5'], loc='lower right',prop={'size':10},labelspacing=0.2)
    plt.xlabel('Resolution')
    plt.ylabel('Accuracy (%)')
    plt.xticks(batch_x_ticks_res)
    #plt.savefig('throughput6.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

def process_video(resolution_list, video_path, model):
    final_resolution =[]
    for resolution in resolution_list:
        all_predictions = []
        print('RESOLUTION ****', resolution)
        cap = cv2.VideoCapture(video_path)
        # get frame dimension
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                final_resolution.append({resolution: all_predictions})
                print('OK')
                break
            img = prepare_cv_image_2_keras_image(frame,resolution)
            pred = model.predict(img)
            predictions = decode_predictions(pred,5)
            dict_predict = {i: predictions}
            all_predictions.append(dict_predict)
            #print(pred)
            i = i+1

    plot_accuracy(final_resolution)



# set the video path
video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2/Jacksonhole9804.mp4'

# image directory path
image_directory = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/voc_validation/car/'
# get the list of the image files
image_files = read_image_directory(image_directory)
# the get the list of the resolution
resolution_list= get_resolution_set(320,180, (1920,1080))
#load model
model = load_model('mobilenet_model_voc_20class_ep_50_sgd.h5')

#process_image(resolution_list, image_directory,image_files, model)
process_video(resolution_list, video_path, model)
