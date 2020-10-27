from keras.preprocessing import image
import numpy as np
#import matplotlib.pyplot as plt
import keras
from datetime import datetime
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions
import psutil
import csv
#from keras.applications import mobilenet
#from tensorflow.keras.applications.imagenet_utils import decode_predictions

def get_cpu():
    cpu = ''
    for x in range(3):
        cpu = cpu + str(psutil.cpu_percent(interval=1))+' '
    return cpu

def get_cpu_percent():
    cpu = psutil.cpu_percent(interval=1)
    return cpu


def get_used_mem_bytes():
    return psutil.virtual_memory().used /(1024*1024)

def get_used_mem_percentage():
    return psutil.virtual_memory().percent

def get_available_memory():
    return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total


def load_DNN_model(model_name):
    # load customized model
    if model_name == 'mobilenet_custom':
        # load model
        model = load_model('mobilenet_model.h5')
        return model

    # load resnet
    if model_name == 'ResNet50':
        model_resnet = ResNet50(weights='imagenet')
        return model_resnet

    # load VGG 16
    if model_name == 'VGG16':
        model_VGG = VGG16(weights='imagenet')
        return model_VGG

    # load resnet
    if model_name == 'InceptionResNet50':
        model_resnet = InceptionResNetV2(weights='/app/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
        return model_resnet

    # load mobilenet
    if model_name == 'MobileNet':
        model_mobilenet = MobileNet(weights='imagenet')
        return model_mobilenet

    # load mobilenetv2
    if model_name == 'MobileNetV2':
        model_mobilenetv2 = keras.applications.mobilenet_v2.MobileNetV2()
        return model_mobilenetv2

# batch holder: function to hold the batch size
def batch_of_images(frame_list, model):
    # get the resolution
    x,y,z = frame_list[0][0].shape
    #batch_size = 3
    batch_holder = np.zeros((len(frame_list), x, y, z))
    other_metrics = []
    i=0
    for frame in frame_list:
        batch_holder[i]= frame[0] # [0]because each frame is attached with time and other metrics
        other_metrics.append(frame[1:])
        i +=1

    #frame_time = frame_by_frame_prediction(batch_holder,model)
    batch_time, pred =  batch_prediciton(batch_holder, other_metrics,model)
    return batch_time, pred
    #return frame_time,batch_time

# evaluation function to log memory and cpu
def eval_memory_cpu(batchsize):
    data = [batchsize, get_used_mem_percentage(), get_cpu_percent()]
    with open('batch_data_cpumemory5.csv', mode='a') as batch_data:
        batch_writer = csv.writer(batch_data, delimiter=',')
        batch_writer.writerow(data)

# generate batch predictions
def batch_prediciton(batch_holder,other_metrics,model):
  # get the intial date time processing
  dt1 = datetime.now()
  #image = preprocess_input(batch_holder)
  pred = model.predict(batch_holder)
  eval_memory_cpu(batch_holder.shape[0])
  predict_labels = decode_predictions_voc(pred,2)
  #print('Predicted:', decode_predictions(pred, 3)[0])
  # for predict in predict_labels:
  #     print('Predicted:',predict)

  # get final time of batch prediction
  dt2 = datetime.now()
  time_diff_batch = dt2-dt1
  processed_framedata_with_other_metric = []
  #print('Other Metrics***********', other_metrics)
  for i in range(0,len(predict_labels)):
      data =[]
      data.append(predict_labels[i])
      data[1:] = other_metrics[i]
      processed_framedata_with_other_metric.append(data)

  return time_diff_batch.total_seconds()*1000 , processed_framedata_with_other_metric

# decode predictions
def decode_predictions_voc(pred, k):
    predict_list_full = []
    class_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor']
    # for top-k score
    # index1= pred[0].argsort()[::-1][: k]
    # print('Index1',index1)
    for i in range(0, len(pred)):
        index = pred[i].argsort()[-k:][::-1]
        predict_list =[]
        for i in range(0, k):
            predict_list.append({class_labels[index[i]]: pred[0][index[i]]})
            # print(class_labels[index[i]], " : ", pred[0][index[i]])
        predict_list_full.append(predict_list)
    return predict_list_full


