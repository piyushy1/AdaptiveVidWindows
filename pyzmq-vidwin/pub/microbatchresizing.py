import queue
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image
import math

CANDIDATE_RESOLUTION_SET = [ (100,100), (250,250),(500,500),(1000,1000),(1920,1080)]

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


def predict_label_binary_search(keyframe,model,query_predicates, resolution_set):
    # get the middle resolution
    resolution_middle = math.trunc(len(resolution_set) / 2)  # get middle element for binary search
    keyframe = prepare_cv_image_2_keras_image(keyframe, resolution_set[resolution_middle])
    pred = model.predict(keyframe)
    predict_labels = decode_predictions(pred, query_predicates['ACCURACY'])
    #print(predict_labels)
    object_present = []  # temporary presence of number of objects available
    for objects in query_predicates['object']:
        for i in range(0, len(predict_labels)):
            if objects in predict_labels[i]:
                #print(objects, predict_labels[i])
                object_present.append(predict_labels[i])
    # for objects in query_predicates['object']:
    #     if any(objects in d for d in predict_labels):
    #         object_present.append('True')

    return resolution_middle, object_present


def get_sum(object_list):
    score_val =[]
    for d in object_list:
        for key in d:
            score_val.append(d[key])

    return sum(score_val)

def get_ratio(object_list):
    score_val = []
    ratio =[]
    for d in object_list:
        for key in d:
            score_val.append(d[key])

    for i in range(0, len(score_val) - 1):
        ratio.append(score_val[i + 1] / score_val[i])

    return sum(ratio)/len(ratio)

def keyframe_resizer(keyframe,model,query_predicates, resolution_set):
    prev_res_flag = None
    #prev_res = None
    prev_object_present = 0
    # resolution_middle = None
    # object_present = []
    resolution_middle, object_present = predict_label_binary_search(keyframe, model, query_predicates, resolution_set)
    prev_res = resolution_set[resolution_middle]

    while True:
        try:
            #resolution_middle = math.trunc(len(resolution_set) / 2)  # get middle element for binary search

            ###################################################
            #CACHE CONCEPT
            ###################################################

            # if object not present increase the resolution
            if len(object_present) == 0:
                # increase the resolution of image to check
                prev_object_present = 0
                if prev_res_flag == True:
                    return prev_res
                else:
                    if resolution_set[resolution_middle] != CANDIDATE_RESOLUTION_SET[-1]:
                        prev_res_flag = False
                        prev_res = resolution_set[resolution_middle]
                        resolution_middle, object_present = predict_label_binary_search(keyframe,model,query_predicates,resolution_set[resolution_middle:])
                    else:
                        return CANDIDATE_RESOLUTION_SET[0] # if resolution not find till end return lowest resolution

            # if object present then decrease the resolution
            else:
                if len(object_present)==1:
                    if prev_object_present > len(object_present):
                        return prev_res
                    else:
                        prev_object_present = len(object_present)
                        # simply resize
                        if resolution_set[resolution_middle] == CANDIDATE_RESOLUTION_SET[0]:
                            return CANDIDATE_RESOLUTION_SET[0]
                        else:
                            prev_res_flag = True
                            prev_res = resolution_set[resolution_middle]
                            resolution_middle, object_present = predict_label_binary_search(keyframe, model, query_predicates, resolution_set[:resolution_middle])

                if len(object_present)>1:
                    if prev_object_present > len(object_present):
                        return prev_res
                    else:
                        prev_object_present = len(object_present)
                        # simply resize
                        if resolution_set[resolution_middle] == CANDIDATE_RESOLUTION_SET[0]:
                            return CANDIDATE_RESOLUTION_SET[0]
                        else:
                            if get_sum(object_present)> 0.5 and get_ratio(object_present)>0.5:
                                prev_res_flag = True
                                prev_res = resolution_set[resolution_middle]
                                resolution_middle, object_present = predict_label_binary_search(keyframe, model,
                                                                                                query_predicates,resolution_set[:resolution_middle])
                            else:
                                return prev_res

                #     # resize with ration
        except Exception as e:
            print('Resizing Exception',str(e))


#'resize the full batch as per recieved keyframe resolution'
def resize_micro_batch(microbatch, keyframe_resolution):
    for frames in microbatch:
        frames[0] = cv2.resize(frames[0],keyframe_resolution)

    return microbatch


def resizer(inp_q, out_q, query_predicates):
    #model = load_model('mobilenet_model_voc_20class_ep_200_sgd_layer_59.h5')
    model = load_model('mobilenet_model_voc_20class_ep_40_sgd_layer_83.h5')
    while True:
        try:
            new_micro_batch = inp_q.get(timeout= None)
            resolution = keyframe_resizer(new_micro_batch[0][0],model,query_predicates,CANDIDATE_RESOLUTION_SET)
            print('The RESOLUTION IS *********************************', resolution)
            #resize full microbatch
            resized_microbatch = resize_micro_batch(new_micro_batch,resolution)
            out_q.put(resized_microbatch)

        except queue.Empty:
            pass

# for EVALUATION
def fixed_resizer(inp_q, out_q, query_predicates):

    while True:
        try:
            new_micro_batch = inp_q.get(timeout= None)
            out_q.put(new_micro_batch)

        except queue.Empty:
            pass