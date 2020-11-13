import torch
import numpy as np
import torchvision
import pathlib
from PIL import Image
import torchvision.transforms as TF
from operator import itemgetter
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from datetime import datetime

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# model.cuda()
# model.eval()

# image = Image.open('/home/dhaval/Desktop/Car-Image.jpg')
#
# x = TF.to_tensor(image)
# x.unsqueeze_(0)
# print(x.shape)

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

# predictions = model(x)
#
# print(predictions)


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def load_faster_rcnn_model():
    detection_model = load_model('faster_rcnn_resnet50_coco_2018_01_28')
    print('LOAD MODEL*********************************')
    return detection_model

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# batch holder: function to hold the batch size
def batch_of_images_fr(frame_list, model):
    # get the resolution
    x,y,z = frame_list[0][0].shape
    #batch_size = 3
    batch_holder = []
    other_metrics = []
    i=0
    for frame in frame_list:
        if type(frame) == list:
            batch_holder.append(frame[0]) # [0]because each frame is attached with time and other metrics
            other_metrics.append(frame[1:])
            i +=1

    #frame_time = frame_by_frame_prediction(batch_holder,model)
    batch_time, pred =  get_prediction(batch_holder, other_metrics,model, threshold =0.2)
    return batch_time, pred
    #return frame_time,batch_time

def get_prediction(img_batch, other_metrics, detection_model, threshold):
    dt1 = datetime.now()
    car =[]
    person=[]
    pred_boxes_full =[]
    pred_class_full=[]
    pred_scores_full =[]
    if len(img_batch)>=50:
        idx = int(len(img_batch)/2)
        input_tensor =None
        for i in range(0,2):
            if i==0:
                input_tensor = [tf.convert_to_tensor(frame) for frame in img_batch[:idx]]
            else:
                input_tensor = [tf.convert_to_tensor(frame) for frame in img_batch[idx:]]
            predictions = detection_model(tf.stack(input_tensor))
            pred_classes = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(frame_classes)] for frame_classes
                            in list(predictions['detection_classes'].numpy().astype(np.int64))]  # Get the Prediction Score
            pred_boxes = [pred for pred in list(predictions['detection_boxes'].numpy())]  # Bounding boxes
            pred_score = [list(scores) for scores in list(predictions['detection_scores'].numpy())]
            pred_thresholds = [[index for index, value in enumerate(score) if value > threshold] for score in
                      pred_score]  # Get list of index with score greater than threshold.

            print('length*****',len(pred_classes))
            pred_boxes = [pred_box[pred_thresholds[index]] for index, pred_box in enumerate(pred_boxes) if len(pred_thresholds[index])>0]
            pred_class = [itemgetter(*pred_thresholds[index])(pred_class) for index, pred_class in enumerate(pred_classes) if len(pred_thresholds[index])>0]
            pred_scores = [itemgetter(*pred_thresholds[index])(pred_score) for index, pred_score in enumerate(pred_score) if len(pred_thresholds[index])>0]

            pred_boxes_full.append(pred_boxes)
            pred_class_full.append(pred_class)
            pred_scores_full.append(pred_scores)
    #print('rediction class', pred_class)
    # get final time of batch prediction
    else:
        input_tensor = [tf.convert_to_tensor(frame) for frame in img_batch]
        predictions = detection_model(tf.stack(input_tensor))
        pred_classes = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(frame_classes)] for frame_classes
                        in list(predictions['detection_classes'].numpy().astype(np.int64))]  # Get the Prediction Score
        pred_boxes = [pred for pred in list(predictions['detection_boxes'].numpy())]  # Bounding boxes
        pred_scores = [list(scores) for scores in list(predictions['detection_scores'].numpy())]
        pred_thresholds = [[index for index, value in enumerate(score) if value > threshold] for score in
                  pred_scores]  # Get list of index with score greater than threshold.

        #print('length*****',len(pred_classes))
        pred_boxes = [pred_box[pred_thresholds[index]] for index, pred_box in enumerate(pred_boxes) if len(pred_thresholds[index])>0]
        pred_class = [itemgetter(*pred_thresholds[index])(pred_class) for index, pred_class in enumerate(pred_classes) if len(pred_thresholds[index])>0]
        pred_scores = [itemgetter(*pred_thresholds[index])(pred_score) for index, pred_score in enumerate(pred_scores) if len(pred_thresholds[index])>0]
        pred_boxes_full.append(pred_boxes)
        pred_class_full.append(pred_class)
        pred_scores_full.append(pred_scores)

    dt2 = datetime.now()
    time_diff_batch = dt2 - dt1

    processed_framedata_with_other_metric = []
    pred_boxes_full = sum(pred_boxes_full, [])
    pred_class_full = sum(pred_class_full, [])
    pred_scores_full = sum(pred_scores_full, [])
    # print('Other Metrics***********', other_metrics)
    for i in range(0, len(pred_class_full)):
        data = []
        data.append([pred_boxes_full[i],pred_class_full[i],pred_scores_full[i]])
        data[1:] = other_metrics[i]
        processed_framedata_with_other_metric.append(data)

    return time_diff_batch.total_seconds() * 1000, processed_framedata_with_other_metric


def scale_up_coordinates(global_bbox, original_res, batch_res):
    original_res = np.array(original_res)
    batch_res = np.array(batch_res)

    original_coordinate_1 = np.array((global_bbox[0], global_bbox[1]))
    new_coordinate_1 = (original_coordinate_1 / (batch_res / original_res))

    original_coordinate_2 = np.array((global_bbox[2], global_bbox[3]))
    new_coordinate_2 = (original_coordinate_2 / (batch_res / original_res))

    return [int(new_coordinate_1[0]), int(new_coordinate_1[1]), int(new_coordinate_2[0]), int(new_coordinate_2[1])]


def update_global_bounding_box(global_bbox, bounding_box):
    if bounding_box[0] < global_bbox[0]:
        global_bbox[0] = bounding_box[0]
    if bounding_box[1] < global_bbox[1]:
        global_bbox[1] = bounding_box[1]
    if bounding_box[2] > global_bbox[2]:
        global_bbox[2] = bounding_box[2]
    if bounding_box[3] > global_bbox[3]:
        global_bbox[3] = bounding_box[3]


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3, res=None):
    global_batch_bbox = [9999, 9999, 0, 0]
    pred_boxes, pred_classes = get_prediction(img, threshold, res)  # Get predictions

    for frame_bounding_box, frame_pred_class in zip(pred_boxes, pred_classes):
        for bounding_box, obj_class in zip(list(frame_bounding_box), frame_pred_class):
            update_global_bounding_box(global_batch_bbox, bounding_box)

    return global_batch_bbox


def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float
    print('Original width, height:', width, height)

    # (height, width)
    resolutions = [(100, 100), (250, 250), (500, 500)]

    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    i = 0
    original_image = None
    frame_batch = []
    res = random.choice(resolutions)
    global_video_bbox = [99999, 99999, 0, 0]
    while True:
        # Capture frame-by-frame

        ret, frame = cap.read()
        if i >= 1:
            if frame is None:
                break
            print('frame: ' + str(i))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # thi is done as openc v read bgr while pil read rgb
            original_image = frame
            frame = cv2.resize(frame, res)
            # frame = Image.fromarray(frame)
            frame_batch.append(frame)

            if len(frame_batch) == 50:
                print('Batch:' + str(i) + ' width, height:', res)
                global_batch_bbox = object_detection_api(frame_batch, threshold=0.5, res=res)
                global_batch_bbox = scale_up_coordinates(global_batch_bbox, original_res=(height, width), batch_res=res)
                update_global_bounding_box(global_video_bbox, global_batch_bbox)
                frame_batch = []
                res = random.choice(resolutions)
                print('Global bbox: ', global_video_bbox)
        i = i + 1
        if not ret:
            break
    cap.release()

    pt1, pt2 = (global_video_bbox[1], global_video_bbox[0]), (
        global_video_bbox[3], global_video_bbox[2])
    cv2.rectangle(original_image, pt1, pt2, color=(0, 255, 0), thickness=3)
    plt.imsave('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/images/final.png', original_image)

