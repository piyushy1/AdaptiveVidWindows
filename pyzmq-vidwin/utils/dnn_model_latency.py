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
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from statistics import mean


# decode predictions
def decode_predictions(pred,k):
    class_labels = ['car','person']
    #for top-k score
    index = pred.argsort()[-2:][::-1]

    print("Prdicted label for Image: ", k)
    # for i in range(index[0].shape[0]):
    #     print(class_labels[index[0][i]], " : ", pred[0][index[0][i]])


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
  # for k in range(0,batch_holder.shape[0]):
  #     print(k)
  #     #decode_predictions(pred,k)

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

    frame_time = frame_by_frame_prediction(batch_holder,model)
    batch_time =  batch_prediciton(batch_holder, model)
    #return batch_time
    return frame_time,batch_time



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
    avg_frame_process_time =[]
    avg_batch_process_time =[]

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
            frame_prcess_time , batch_process_time = batch_of_images(frame_list, batch_size, model)
            avg_frame_process_time.append(frame_prcess_time)
            avg_batch_process_time.append(batch_process_time)
            #batch_process_time = batch_of_images(frame_list, batch_size, model)
            #print(frame_prcess_time, batch_process_time)
            #print(batch_process_time)
            frame_list.clear()

    #print("Total time", diff.total_seconds() )
    #throughput = frame_count_num/diff.total_seconds()
    # if not (batch_size ==100):
    #     x = avg_frame_process_time[1: len(avg_frame_process_time)-1]
    #     y = avg_batch_process_time[1: len(avg_batch_process_time)-1]
    #     return mean(x), mean(y)
    #
    # if (batch_size== 100):
    #     return mean(avg_frame_process_time), mean(avg_batch_process_time)
    return mean(avg_frame_process_time), mean(avg_batch_process_time)
    #return sum(avg_frame_process_time),sum(avg_batch_process_time)



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
def plotgraph(frame_latency, batch_latency):
    fig = plt.figure()
    barWidth = 0.30
    # Set position of bar on X axis
    r1 = np.arange(len(frame_latency))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Amber = '#F5B14C'
    # Make the plot
    plt.bar(r1, frame_latency, color=CB91_Green, width=barWidth, edgecolor='white', label='Frame By Frame Processing')
    plt.bar(r2, batch_latency, color=CB91_Amber, width=barWidth, edgecolor='white', label='Batch Processing')
    #plt.bar(r3, ObjAT, color='#2d7f5e', width=barWidth, edgecolor='white', label='Objects+Attributes+Tracking' )

    # Add xticks on the middle of the group bars
    plt.xlabel('Video Frames')
    plt.ylabel('Average Latency(ms)')
    plt.xticks([r +0.15 for r in range(len(frame_latency))], ['5', '10','25','50','100'])

    # Create legend & Show graphic
    plt.legend()
    #plt.grid()
    plt.savefig('Latency6.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

    # batch_x_ticks =[5,10,25,50,100]
    # t_y = np.array([5,10,25,50,100])
    # plt.plot(t_y, throughput_list[0], 'r--',t_y, throughput_list[1], 'yD-',t_y, throughput_list[2], 'g^--',t_y, throughput_list[3], 'bo--')
    # plt.legend(['Resnet50','VGG16', 'Mobilenet' , 'Mobilenetv2'], loc='lower right',prop={'size':10},labelspacing=0.2)
    # plt.xlabel('Batch Size')
    # plt.ylabel('Throughput (Frames/sec)')
    # plt.xticks(batch_x_ticks)
    # plt.savefig('throughput.svg', format='svg', dpi=1000,bbox_inches='tight')
    # plt.show()



#load model
#model = load_model('mobilenet_model.h5')

# # load resnet
# model_resnet = ResNet50(weights='imagenet')
#
# # load VGG 16
# model_VGG = VGG16(weights='imagenet')

# load mobilenet
model_mobilenet = MobileNet()

# # load mobilenetv2
# model_mobilenetv2 = keras.applications.mobilenet_v2.MobileNetV2()

#model_list =[model_resnet]
model_list =[model_mobilenet] # define model list

batch_size = [5,10,25,50,100]  # define batch size
frame_latency_time =[]
batch_latency_time =[]

for model in range(0,len(model_list)):
    for i in range(0,len(batch_size)):
        #print(batch_size[i])
        frame_latency, batch_latency = load_video_into_frames('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2.mp4',batch_size[i], model_list[model])
        frame_latency_time.append(frame_latency)
        batch_latency_time.append(batch_latency)

#
# frame_latency_time =[2,3,4,5,6]
# batch_latency_time =[1,2,3,4,5]
plotgraph(frame_latency_time,batch_latency_time)

