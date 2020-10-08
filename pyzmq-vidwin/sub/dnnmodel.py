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
#from keras.applications import mobilenet
#from tensorflow.keras.applications.imagenet_utils import decode_predictions


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
        model_resnet = InceptionResNetV2(weights='imagenet')
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
    #batch_size = 3
    batch_holder = np.zeros((len(frame_list), 299, 299, 3))
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


# generate batch predictions
def batch_prediciton(batch_holder,other_metrics,model):
  # get the intial date time processing
  dt1 = datetime.now()
  #image = preprocess_input(batch_holder)
  pred = model.predict(batch_holder)
  predict_labels = decode_predictions(pred,3)
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
def decode_predictions_custom(pred,k):
    class_labels = ['car','person']
    #for top-k score
    index = pred.argsort()[-2:][::-1]
    print("Prdicted label for Image: ", k)
    for i in range(index[0].shape[0]):
        print(class_labels[index[0][i]], " : ", pred[0][index[0][i]])



