from keras.preprocessing import image
import numpy as np
#import matplotlib.pyplot as plt
import keras
from datetime import datetime
import math
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications import mobilenet
from keras.applications.imagenet_utils import decode_predictions


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

    # load mobilenet
    if model_name == 'MobileNet':
        model_mobilenet = keras.applications.mobilenet.MobileNet()
        return model_mobilenet

    # load mobilenetv2
    if model_name == 'MobileNetV2':
        model_mobilenetv2 = keras.applications.mobilenet_v2.MobileNetV2()
        return model_mobilenetv2

# batch holder: function to hold the batch size
def batch_of_images(frame_list, model):
    #batch_size = 3
    batch_holder = np.zeros((len(frame_list), 224, 224, 3))
    i=0
    for frame in frame_list:
        batch_holder[i]= frame[0] # [0]because each frame is attached with time and other metrics
        i +=1

    #frame_time = frame_by_frame_prediction(batch_holder,model)
    batch_time =  batch_prediciton(batch_holder, model)
    return batch_time
    #return frame_time,batch_time


# generate batch predictions
def batch_prediciton(batch_holder,model):
  # get the intial date time processing
  dt1 = datetime.now()
  #image = preprocess_input(batch_holder)
  pred = model.predict(batch_holder)
  print('Predicted:', decode_predictions_custom(pred, 3))

  # for k in range(0,batch_holder.shape[0]):
  #     #print(k)
  #     decode_predictions(pred,k)

  # get final time of batch prediction
  dt2 = datetime.now()
  time_diff_batch = dt2-dt1
  return time_diff_batch.total_seconds()*1000


# decode predictions
def decode_predictions_custom(pred,k):
    class_labels = ['car','person']
    #for top-k score
    index = pred.argsort()[-2:][::-1]
    print("Prdicted label for Image: ", k)
    for i in range(index[0].shape[0]):
        print(class_labels[index[0][i]], " : ", pred[0][index[0][i]])



