# Author - Piyush Yadav
# Insight Centre for Data Analytics
# Package- VidWIN Project
from __future__ import division
import cv2
from time import sleep
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import keras
from datetime import datetime
import math
from keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet121
from keras.applications.imagenet_utils import decode_predictions


# decode predictions
# def decode_predictions(pred,k):
#     class_labels = ['car','person']
#     #for top-k score
#     index = pred.argsort()[-2:][::-1]
#
#     print("Prdicted label for Image: ", k)
#     for i in range(index[0].shape[0]):
#         print(class_labels[index[0][i]], " : ", pred[0][index[0][i]])


# Generate frame by frame predcitions
def frame_by_frame_prediction(batch_holder,model):
  # log the time
  dt1 = datetime.now()
  # read each image frame by frame
  for frame in range(0,batch_holder.shape[0]):
    image = batch_holder[frame]
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    pred = model.predict(image)
    #decode_predictions(pred,frame)

  # log the final time
  dt2 = datetime.now()
  time_diff_frame = dt2-dt1
  return time_diff_frame.total_seconds()*1000

# generate batch predictions
def batch_prediciton(batch_holder,model):
  # get the intial date time processing
  dt1 = datetime.now()
  #image = preprocess_input(batch_holder)
  pred = model.predict(batch_holder)
  #print('Predicted:', decode_predictions(pred, top=3))
  #print('Pred Shape: ')
  # for k in range(0,batch_holder.shape[0]):
  #     print('Shape of a: ',a.shape)
  #     print('Predicted:', decode_predictions(pred, top=3)[0])

  # get final time of batch prediction
  dt2 = datetime.now()
  time_diff_batch = dt2-dt1
  return time_diff_batch.total_seconds()*1000


# function 2 to read the images
def load_image(img_path, show=False):

    #img = image.load_img(img_path, target_size=(200, 200))
    #img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_path, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    #print("Image Tensor SHape:" , img_tensor.shape)
    return img_tensor

# batch holder: function to hold the batch size
def batch_of_images(frame_list,batch_size, model):
    #batch_size = 3
    batch_holder = np.zeros((batch_size, 224, 224, 3))
    i=0
    for frame in frame_list:
        batch_holder[i]= frame
        i +=1

    #frame_time = frame_by_frame_prediction(batch_holder,model)
    batch_time =  batch_prediciton(batch_holder, model)
    return batch_time
    #return frame_time,batch_time


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

#load the video publisher
def load_video_into_frames(publisher_path, batch_size, model):
    # read the video file
    cap = cv2.VideoCapture(publisher_path)
    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS for video: ", " : ", fps )
    # time to sleep the process
    sleep_time = 1/(fps) # 2 is added to make the reading frame time and video time equivalnet. this is an empirical value may change for others.
    print("Sleep Time : ", sleep_time )
    # to check the frame count
    frame_count_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count_num/fps
    print("Duration of video ", " ", duration%60)
    frame_list =[]
    tt1 = datetime.now()
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        # need to set the frame rate:
        #sleep(sleep_time)
        #np_frame = cv2.imread('video',frame)
        resize_np_frame =cv2.resize(frame, (224,224))
        frame_list.append(resize_np_frame)
        if len(frame_list)== batch_size:
            #frame_prcess_time , batch_process_time = batch_of_images(frame_list, batch_size, model)
            batch_process_time = batch_of_images(frame_list, batch_size, model)
            #print(frame_prcess_time, batch_process_time)
            #print(batch_process_time)
            frame_list.clear()

    diff = datetime.now()-tt1
    #print("Total time", diff.total_seconds() )
    throughput = frame_count_num/diff.total_seconds()
    #print("throughput: ", throughput)
    return throughput


def load_video_frame(videoFile):
    count = 0
    #videoFile = "Tom and jerry.mp4"
    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
    frameRate = cap.get(25) #frame rate
    x=1
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        #if (frameId % math.floor(frameRate) == 0):
        filename ="frame%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
    cap.release()
    #print ("Done!")


# function to plot live graph
def plotgraph(throughput_list):
    fig = plt.figure()
    batch_x_ticks = [1,5,10,20,30,40,50,60,70,80,90,100]
    t_y = np.array([1,5,10,20,30,40,50,60,70,80,90,100])
    #batch_x_ticks = [1,5]
    #t_y = np.array([1,5])
    colorlist = get_colorlist()
    marker_list = get_marker_list()
    for i in range(0,len(throughput_list)):
        plt.plot(t_y, throughput_list[i],color = colorlist[i], linestyle = '-', marker = marker_list[i], markersize = 7)

    #plt.plot(t_y, throughput_list[0], 'r^--',t_y, throughput_list[1], 'mD-',t_y, throughput_list[2], '#92D050',t_y, throughput_list[3], 'bo--')
    plt.legend(['Resnet50','VGG16', 'Mobilenet' , 'Mobilenetv2', 'DenseNet121'], loc='lower right',prop={'size':10},labelspacing=0.2)
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Frames/sec)')
    plt.xticks(batch_x_ticks)
    plt.savefig('dnn_throughput.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

#load model
#model = load_model('mobilenet_model.h5')

# load resnet
model_resnet = ResNet50(weights='imagenet')

# load VGG 16
model_VGG = VGG16(weights='imagenet')

# load mobilenet
model_mobilenet = MobileNet(weights='imagenet')

# load mobilenetv2
model_mobilenetv2 = MobileNetV2(weights='imagenet')

# load Densenet121
model_densenet121 = DenseNet121(weights='imagenet')

#model_list =[model_resnet,model_mobilenet]
model_list =[model_resnet,model_VGG,model_mobilenet,model_mobilenetv2, model_densenet121] # define model list
batch_size = [1,5,10,20,30,40,50, 60, 70,80,90,100]  # define batch size
#batch_size = [1,5]  # define batch size
total_throughput =[]

for model in range(0,len(model_list)):
    model_throughput_time = []
    for i in range(0,len(batch_size)):
        #print(batch_size[i])
        th_put = load_video_into_frames('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar.mp4',batch_size[i], model_list[model])
        model_throughput_time.append(th_put)

    total_throughput.append(model_throughput_time)


#total_throughput =[[10,8,3,6,5,4,6,8,3,6,5,4],[1,2,3,4,5,4,6,4,6,8,3,6],[2,5,6,7,5,4,6,4,6,8,3,6],[5,6,4,5,5,4,6,4,6,8,3,6]]
plotgraph(total_throughput)