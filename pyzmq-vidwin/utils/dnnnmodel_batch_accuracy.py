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
import matplotlib.patches as mpatches
import tensorflow as tf
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'



def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.compat.v1.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


detection_model = load_model('faster_rcnn_resnet50_coco_2018_01_28')
tf.compat.v1.global_variables_initializer()

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
    car =[]
    person=[]
    input_tensor = [tf.convert_to_tensor(frame) for frame in img_batch]
    # input_tensor = tf.convert_to_tensor(img_batch[0])
    # input_tensor = input_tensor[tf.newaxis, ...]
    predictions = detection_model(tf.stack(input_tensor))
    pred_classes = [[COCO_INSTANCE_CATEGORY_NAMES[index] for index in list(frame_classes)] for frame_classes
                    in list(predictions['detection_classes'].numpy().astype(np.int64))]  # Get the Prediction Score
    pred_boxes = [pred for pred in list(predictions['detection_boxes'].numpy())]  # Bounding boxes
    pred_score = [list(scores) for scores in list(predictions['detection_scores'].numpy())]
    pred_t = [[index for index, value in enumerate(score) if value > threshold] for score in
              pred_score]  # Get list of index with score greater than threshold.
    pred_boxes = [pred_box[pred_t[index]] for index, pred_box in enumerate(pred_boxes)]
    pred_class = [itemgetter(*pred_t[index])(pred_class) for index, pred_class in enumerate(pred_classes)]
    pred_scores = [itemgetter(*pred_t[index])(pred_score) for index, pred_score in enumerate(pred_score)]

    # get all the car and person in the list with accuracy
    for i in range(0,len(pred_class)):
        for index, val in enumerate(pred_class[i]):
            if val == 'car':
                car.append(pred_scores[i][index])
            if val == 'person':
                person.append(pred_scores[i][index])

    #print(pred_scores)
    return car,person



def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1, text_th=3):
    global_batch_bbox = [9999, 9999, 0, 0]
    pred_car, pred_person = get_prediction(img, threshold)  # Get predictions
    return pred_car, pred_person



car_accuracy = []
person_accuracy =[]

def stream(video_path, batchsize, resolutions):
    cap = cv2.VideoCapture(video_path)
    frame_count_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Frame Count Number******* ', frame_count_num)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float
    #print('Original width, height:', width, height)

    #resolutions = [(1280, 720)]

    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    i = 0
    original_image = None
    frame_batch = []
    res = random.choice(resolutions)
    global_video_bbox = [99999, 99999, 0, 0]
    carlist = []
    person_list = []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if i <= 300:
            if not ret:
                break
            #print('frame: ' + str(i))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # thi is done as openc v read bgr while pil read rgb
            original_image = frame
            frame = cv2.resize(frame, res)
            # frame = Image.fromarray(frame)
            frame_batch.append(frame)

            if len(frame_batch) == batchsize:
                #print('Batch:' + str(i) + ' width, height:', res)
                pred_car, pred_person = object_detection_api(frame_batch, threshold=0.2)
                carlist.append(pred_car)
                person_list.append(pred_person)
                frame_batch = []
                res = random.choice(resolutions)
                #print('Global bbox: ', global_video_bbox)
        else:
            return carlist, person_list
        i = i + 1


    #return carlist, person_list
def plot_resolution_accuracy(latency_list, personaccuracy):
    resolution_x_ticks = []
    latency = []
    latencyperson =[]

    for data in latency_list:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            latency.append(value)

    for data in personaccuracy:
        for key, value in data.items():
            #resolution_x_ticks.append(str(key))
            latencyperson.append(value)

    # since 100 batch is out of memory
    # resolution_x_ticks.append('100')
    # latency.append(latency[0])
    # latencyperson.append(latencyperson[0])
    x_pos = [i for i in range(1,len(resolution_x_ticks)+1)]
    #data1 = [np.random.normal(0, std, size=100) for std in x_pos]
    # print(data)
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))

    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))


    parts = plt.violinplot(latency, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#2CBDFE')
        #pc.set_edgecolor('black')
        pc.set_alpha(1)

    part2 = plt.violinplot(latencyperson, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)
    for pc in part2['bodies']:
        pc.set_facecolor('#F5B14C')
        #pc.set_edgecolor('black')
        pc.set_alpha(0.85)

    add_label(parts,'Car')

    add_label(part2,'Person')


    #plt.title('Latency', fontsize=10)
    plt.xlabel('Resolution')
    plt.ylabel('Latency Distribution (ms) ')
    plt.xticks(x_pos,resolution_x_ticks)
    # Create legend & Show graphic
    plt.legend(*zip(*labels), loc=2)
    plt.savefig('dnn_resolution_accuracy.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()

def get_colorlist():
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue,CB91_Amber,CB91_Purple,CB91_Violet, CB91_Pink, CB91_Green]
    return color_list

def get_marker_list():
    marker_list =['o','*','s','v','D','1','d','3','4']
    return marker_list

import statistics
def plot_resolution_accuracy_bar_graph(video_list):

    resolution_x_ticks = []
    sandy = []
    jackson = []
    southampton =[]
    sandymedian= []
    jacksonmedian =[]
    southamptonmedian =[]
    a = video_list[0]
    b = video_list[1]
    c = video_list[2]
    for data in a:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            sandy.append(sum(value) / len(value))
            sandymedian.append(statistics.median(value))

    for data in b:
        for key, value in data.items():
            # resolution_x_ticks.append(str(key))
            jackson.append(sum(value) / len(value))
            jacksonmedian.append(statistics.median(value))

    for data in c:
        for key, value in data.items():
            # resolution_x_ticks.append(str(key))
            southampton.append(sum(value) / len(value))
            southamptonmedian.append(statistics.median(value))


    colorlist = get_colorlist()
    marker_list = get_marker_list()
    accuracylist = [sandy, jackson,southampton, sandymedian, jacksonmedian, southamptonmedian]
    fig = plt.figure()
    # t_y = np.array([1,5,10,20,30,40,50,60,70,80,90,100])
    t_y = np.array(resolution_x_ticks)
    for i in range(0, len(accuracylist)):
        plt.plot(t_y, accuracylist[i], color=colorlist[i], linestyle='-', marker=marker_list[i], markersize=7)

    # plt.plot(t_y, top_1, 'r^--',t_y, top_2, 'mD-',t_y, top_3, '#92D050',t_y, top_4, 'bo--',t_y, top_4, 'b^--')
    plt.legend(['Sandy Lane Mean','Jackson Hole Mean', 'Southampton Mean', 'Sandy Lane Median','Jackson Hole Median', 'Southampton Median'], loc='lower right', prop={'size': 10}, labelspacing=0.2)
    plt.xlabel('Resolution')
    plt.ylabel('Average Accuracy')
    plt.xticks(resolution_x_ticks)
    plt.ylim(0, 1)
    plt.savefig('asp_median_resolution.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()


def plot_batch_accuracy(latency_list, personaccuracy):
    resolution_x_ticks = []
    latency = []
    latencyperson =[]

    for data in latency_list:
        for key, value in data.items():
            resolution_x_ticks.append(str(key))
            latency.append(value)

    for data in personaccuracy:
        for key, value in data.items():
            #resolution_x_ticks.append(str(key))
            latencyperson.append(value)

    # since 100 batch is out of memory
    resolution_x_ticks.append('100')
    latency.append(latency[0])
    latencyperson.append(latencyperson[0])
    x_pos = [i for i in range(1,len(resolution_x_ticks)+1)]
    #data1 = [np.random.normal(0, std, size=100) for std in x_pos]
    # print(data)
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 6))

    labels = []

    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))


    parts = plt.violinplot(latency, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#2CBDFE')
        #pc.set_edgecolor('black')
        pc.set_alpha(1)

    part2 = plt.violinplot(latencyperson, x_pos, points=20, widths=0.3,
                   showmeans=True, showextrema=True, showmedians=True)
    for pc in part2['bodies']:
        pc.set_facecolor('#F5B14C')
        #pc.set_edgecolor('black')
        pc.set_alpha(0.85)

    add_label(parts,'Car')

    add_label(part2,'Person')


    #plt.title('Latency', fontsize=10)
    plt.xlabel('Resolution')
    plt.ylabel('Latency Distribution (ms) ')
    plt.xticks(x_pos,resolution_x_ticks)
    # Create legend & Show graphic
    plt.legend(*zip(*labels), loc=2)
    #plt.savefig('dnn_batch_accuracy.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.show()

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

# #######BATCHING FUNCTINALITY
# batch_size = [1,5,10,25,50]  # define batch size
#
# #batch_size = [5]
# for i in range(0,len(batch_size)):
#     carlist, person_list = stream('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar_clip.mp4',batch_size[i])
#     car_accuracy.append({str(batch_size[i]):[item for sublist in carlist for item in sublist]})
#     person_accuracy.append({str(batch_size[i]):[item for sublist in person_list for item in sublist]})
#
# plot_batch_accuracy(car_accuracy,person_accuracy)

#########RESOLUTION FUNCTIONALITY
#resolution_list1= get_resolution_set(300,300, (1800,1800))
#resolution_list1.append((100,100))
resolution_list1= get_resolution_set(320,180, (1920,1080))
#resolution_list1= [(1200,1200), (500,500),(300,300), (100,100)]
videolist = ['personcar_clip.mp4','test.mp4','soutampton_clip.mp4']
#videolist = ['personcar_clip.mp4','test.mp4']
sandy_lane =[]
for video in videolist:
    car_ac= []
    for i in range(0,len(resolution_list1)):
        print('Loop Update', video,resolution_list1[i])
        carlist, person_list = stream('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/'+video,1,[resolution_list1[i]])
        car_ac.append({str(resolution_list1[i]):[item for sublist in carlist for item in sublist]})
        #person_accuracy.append({str(resolution_list1[i]):[item for sublist in person_list for item in sublist]})

    sandy_lane.append(car_ac)
#plot_resolution_accuracy(car_accuracy,person_accuracy)
plot_resolution_accuracy_bar_graph(sandy_lane)