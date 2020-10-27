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


def get_prediction(img_batch, threshold):
    # transform = TF.Compose([TF.ToTensor()])  # Defing PyTorch Transform
    # frames = [transform(Image.fromarray(frame)).cuda() for frame in img_batch]
    # predictions = model(frames)  # Pass the image to the model
    # del frames
    # del transform
    # import gc
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except Exception as e:
    #         pass

    # torch.cuda.empty_cache()
    detection_model = load_model('faster_rcnn_resnet50_coco_2018_01_28')
    input_tensor = tf.convert_to_tensor(img_batch[0])
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_model(input_tensor)
    print(detections)

    pred_classes = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(pred['labels'].cpu().data.numpy())] for pred
                    in predictions]  # Get the Prediction Score
    pred_boxes = [pred['boxes'].cpu().data.detach().numpy() for pred in predictions]  # Bounding boxes
    pred_score = [list(pred['scores'].cpu().data.detach().numpy()) for pred in predictions]
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

    # x, y = int(new_x[0]), int(new_x[1])
    original_coordinate_2 = np.array((global_bbox[2], global_bbox[3]))
    new_coordinate_2 = (original_coordinate_2 / (batch_res / original_res))
    # x1, y1 = int(new_y[0]), int(new_y[1])

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


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    global_batch_bbox = [9999, 9999, 0, 0]
    pred_boxes, pred_class = get_prediction(img, threshold)  # Get predictions

    for frame_bounding_box, frame_pred_class in zip(pred_boxes, pred_class):
        for bounding_box, obj_class in zip(list(frame_bounding_box), frame_pred_class):
            update_global_bounding_box(global_batch_bbox, bounding_box)
    # cv2.rectangle(img[0], (global_bbox[0],global_bbox[1]), (global_bbox[2],global_bbox[3]), color=(0, 255, 0),
    #               thickness=rect_th)  # Draw Rectangle with the coordinates
    # # cv2.putText(img[0], obj_class, (bounding_box[0],bounding_box[1]), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
    # #             thickness=text_th)  # Write the prediction class
    # plt.figure(figsize=(20, 30))  # display the output image
    # plt.imsave('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/images/batch_'+str(c)+'.png', img[0])
    return global_batch_bbox


def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float
    print('Original width, height:', width, height)

    resolutions = [(1280, 720)]

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

            if len(frame_batch) == 18:
                print('Batch:' + str(i) + ' width, height:', res)
                global_batch_bbox = object_detection_api(frame_batch, threshold=0.5)
                global_batch_bbox = scale_up_coordinates(global_batch_bbox, original_res=(width, height), batch_res=res)
                update_global_bounding_box(global_video_bbox, global_batch_bbox)
                frame_batch = []
                res = random.choice(resolutions)
                print('Global bbox: ', global_video_bbox)
        i = i + 1
        if not ret:
            break
    cap.release()

    pt1, pt2 = (global_video_bbox[0], global_video_bbox[1]), (global_video_bbox[2], global_video_bbox[3])
    cv2.rectangle(original_image, pt1, pt2, color=(0, 255, 0), thickness=3)
    plt.imsave('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/images/final.png', original_image)


# object_detection_api('/home/dhaval/Desktop/Car-Image.jpg')
stream('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar_clip.mp4')
