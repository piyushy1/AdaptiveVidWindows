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
from datetime import datetime
from sys import getsizeof

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

def get_pred_score(data_list):
    for data in data_list:
        if 'car' in data:
            return data['car']



def calculate_avg_accuracy(acc_value):
    top1 =[]
    top2= []
    top3 = []
    top4 =[]
    top5 = []
    for data in acc_value:
        for key, value in data.items():
            if any('car' in d for d in value[:1]):
                top1.append(get_pred_score(value[:1]))
            if any('car' in d for d in value[:2]):
                top2.append(get_pred_score(value[:2]))
            if any('car' in d for d in value[:3]):
                top3.append(get_pred_score(value[:3]))
            if any('car' in d for d in value[:4]):
                top4.append(get_pred_score(value[:4]))
            if any('car' in d for d in value[:5]):
                top5.append(get_pred_score(value[:5]))

    top1_accurcacy = sum(top1) / len(top1)
    top2_accurcacy = sum(top2) / len(top2)
    top3_accurcacy = sum(top3) / len(top3)
    top4_accurcacy = sum(top4) / len(top4)
    top5_accurcacy = sum(top5) / len(top5)

    return top1_accurcacy, top2_accurcacy, top3_accurcacy, top4_accurcacy, top5_accurcacy

# function to plot live graph
def plot_top_k_accuracy(final_resolution):
    batch_x_ticks_res = []
    top_1 =[]
    top_2 =[]
    top_3 =[]
    top_4= []
    top_5= []
    for data in final_resolution:
        for key, value in data.items():
            batch_x_ticks_res.append(str(key))
            #a,b,c,d,e = calculate_top_k_accuracy(value)
            a, b, c, d, e = calculate_avg_accuracy(value)
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
    plt.ylabel('Average Accuracy (%)')
    plt.xticks(batch_x_ticks_res)
    #plt.savefig('avgaccuracy_asp.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

def plot_throughput(throughput_list):
    resolution_x_ticks = []
    throughput = []
    for data in throughput_list:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            throughput.append(value)

    fig = plt.figure()
    #t_y = np.array([1,5,10,20,30,40,50,60,70,80,90,100])
    t_y = np.array(resolution_x_ticks)
    plt.plot(t_y, throughput, color=get_colorlist()[0],linestyle = '--', marker = get_marker_list()[0],markersize = 7)
    plt.legend(['throughput'], loc='lower right',prop={'size':10},labelspacing=0.2)
    plt.xlabel('Resolution')
    plt.ylabel('Throughput (fps)')
    plt.xticks(resolution_x_ticks)
    #plt.savefig('throughput_asp.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

def get_colorlist():
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]
    return color_list

def get_marker_list():
    marker_list =['o','*','s','v','D','1','d','3','4']
    return marker_list

def plot_latency(latency_list):
    resolution_x_ticks = []
    latency = []
    for data in latency_list:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            latency.append(value)

    x_pos = [i for i in range(1,len(resolution_x_ticks)+1)]
    #data1 = [np.random.normal(0, std, size=100) for std in x_pos]
    # print(data)
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))
    plt.violinplot(latency, x_pos, points=20, widths=0.3,
                   showmeans=True, color=get_colorlist()[0], showextrema=True, showmedians=True)
    plt.title('Latency', fontsize=10)
    plt.xlabel('Resolution')
    plt.ylabel('Latency Distribution (ms) ')
    plt.xticks(x_pos,resolution_x_ticks)
    #plt.savefig('latency_asp.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_memory_consumption(memory_list):
    resolution_x_ticks = []
    latency = []
    for data in memory_list:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            latency.append(value)

    x_pos = [i for i in range(1, len(resolution_x_ticks) + 1)]
    # data1 = [np.random.normal(0, std, size=100) for std in x_pos]
    # print(data)
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))
    plt.violinplot(latency, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)
    plt.title('Memory', fontsize=10)
    plt.xlabel('Resolution')
    plt.ylabel('Memory Bytes')
    plt.xticks(x_pos, resolution_x_ticks)
    #plt.savefig('memorycons_asp.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()


def process_video(resolution_list, video_path, model):
    final_resolution =[]
    final_throughput = []
    final_latency =[]
    final_memory =[]
    for resolution in resolution_list:
        all_predictions = []
        all_latency =[]
        all_memory = []
        print('RESOLUTION ****', resolution)
        cap = cv2.VideoCapture(video_path)
        # get frame dimension
        width = cap.get(3)  # float
        height = cap.get(4)  # float
        i = 0
        dt1 = datetime.now()
        while True:
            ret, frame = cap.read()
            if not ret:
                dt2 = datetime.now()
                time_diff = (dt2-dt1).total_seconds()
                throughput = (i+1)/time_diff
                final_resolution.append({resolution: all_predictions})
                final_throughput.append({resolution: throughput})
                final_latency.append({resolution: all_latency})
                final_memory.append({resolution: all_memory})
                print('OK')
                break
            img = prepare_cv_image_2_keras_image(frame,resolution)
            all_memory.append(img.nbytes)
            dt3 = datetime.now()
            pred = model.predict(img)
            dt4 = datetime.now()
            time_diff = (dt4-dt3).total_seconds()*1000
            all_latency.append(time_diff)
            predictions = decode_predictions(pred,5)
            print(predictions)
            dict_predict = {i: predictions}
            all_predictions.append(dict_predict)
            #print(pred)
            i = i+1

    plot_top_k_accuracy(final_resolution)
    #plot_throughput(final_throughput)
    #plot_memory_consumption(final_memory)
    #plot_latency(final_latency)



# set the video path
video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/videoplaybackclip.mp4'
#video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test3.mp4'

# image directory path
image_directory = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/voc_validation/car/'
# get the list of the image files
image_files = read_image_directory(image_directory)
# the get the list of the resolution
#resolution_list= get_resolution_set(320,180, (1920,1080))
resolution_list1= get_resolution_set(200,200, (400,400))
#load model
from memory_profiler import profile
from tensorflow.keras.applications import ResNet50

model = load_model('mobilenet_model_voc_20class_ep_40_sgd_layer_83.h5')

# @profile
# def load_model_m():
#     model = load_model('mobilenet_model_voc_20class_ep_50_sgd.h5')
#     #base_model = ResNet50()  # imports the mobilenet model
#
# load_model_m()

#process_image(resolution_list, image_directory,image_files, model)
#process_video(resolution_list, video_path, model)
process_video(resolution_list1, video_path, model)