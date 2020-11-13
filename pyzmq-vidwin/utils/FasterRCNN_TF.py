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
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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


detection_model = load_model('faster_rcnn_resnet50_coco_2018_01_28')

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


def get_prediction(img_batch, threshold, res):
    width, height = res
    input_tensor = [tf.convert_to_tensor(frame) for frame in img_batch]
    predictions = detection_model(tf.stack(input_tensor))
    pred_classes = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(frame_classes)] for frame_classes
                    in list(predictions['detection_classes'].numpy().astype(np.int64))]  # Get the Prediction Score
    pred_boxes = [pred * [height, width, height, width] for pred in
                  list(predictions['detection_boxes'].numpy())]  # Bounding boxes

    pred_score = [list(scores) for scores in list(predictions['detection_scores'].numpy())]
    pred_t = [[index for index, value in enumerate(score) if value > threshold] for score in
              pred_score]  # Get list of index with score greater than threshold.
    pred_boxes = [pred_box[pred_t[index]] for index, pred_box in enumerate(pred_boxes)]
    pred_class = [itemgetter(*pred_t[index])(pred_class) for index, pred_class in enumerate(pred_classes)]
    return pred_boxes, pred_class


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
            # cv2.rectangle(img[0], (int(bounding_box[1]), int(bounding_box[0])),
            #               (int(bounding_box[3]), int(bounding_box[2])),
            #               color=(0, 255, 0), thickness=rect_th)  # Draw Rectangle with the coordinates
            # cv2.putText(img[0], obj_class, (int(bounding_box[1]), int(bounding_box[0])), cv2.FONT_HERSHEY_SIMPLEX,
            #             text_size, (0, 255, 0),
            #             thickness=text_th)  # Write the prediction class
            # plt.figure(figsize=(20, 30))  # display the output image
            # plt.imsave('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/images/batch.png', img[0])
            # plt.close()
            # print()
    return global_batch_bbox


def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float
    print('Original width, height:', width, height)

    # (height, width)
    #resolutions = [(100, 100), (250, 250), (500, 500)]
    resolutions = [(500, 500)]
    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    i = 0
    original_image = None
    frame_batch = []
    res = random.choice(resolutions)
    global_video_bbox = [99999, 99999, 0, 0]
    time1 = datetime.now()
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

            if len(frame_batch) == 40:
                #print('Batch:' + str(i) + ' width, height:', res)
                time = datetime.now()
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
    print('Total Time', (datetime.now()-time1).total_seconds())
    pt1, pt2 = (global_video_bbox[1], global_video_bbox[0]), (
        global_video_bbox[3], global_video_bbox[2])
    cv2.rectangle(original_image, pt1, pt2, color=(0, 255, 0), thickness=3)
    plt.imsave('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/images/final.png', original_image)


# object_detection_api('/home/dhaval/Desktop/Car-Image.jpg')
stream('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar_clip.mp4')
