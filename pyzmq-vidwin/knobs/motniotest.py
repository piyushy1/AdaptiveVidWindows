import cv2
import pybgs
import time
import matplotlib.pyplot as plt
import numpy as np

def extract_motion(video_path):
    time_start = time.time()
    motion_ratio = []
    cap = cv2.VideoCapture(str(video_path))
    index = 0
    global bgs
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (500,500), interpolation = cv2.INTER_AREA)
        if index == 100:
            return motion_ratio
        if index ==0:
            bgs = pybgs.WeightedMovingMean()
        if not ret:
            cap.release()
            break
        index += 1
        if index !=0:
            motion_ratio.append(cal_motion_ratio(frame,bgs))
    time_end = time.time()
    #video_name = Path(video_path).stem
    #logging.info(f'{video_name},motion,{self.name},{time_start},{time_end},{time_end - time_start}')
    return motion_ratio


def cal_motion_ratio(frame,bgs):
    fgmask = bgs.apply(frame)
    motion_ratio = cv2.countNonZero(fgmask) / (frame.shape[0] * frame.shape[1])
    return motion_ratio

def plotgraph(throughput_list):
    fig = plt.figure()
    # batch_x_ticks = [1,5,10,20,30,40,50,60,70,80,90,100]
    # t_y = np.array([1,5,10,20,30,40,50,60,70,80,90,100])
    batch_x_ticks = [frame for frame in range(len(throughput_list[0])) ]
    t_y = np.array([frame for frame in range(len(throughput_list[0])) ])
    # plt.plot(t_y, throughput_list[0], 'r^--',t_y, throughput_list[1], 'mD-',t_y, throughput_list[2], '#92D050',t_y, throughput_list[3], 'bo--')
    plt.plot(t_y, throughput_list[0], 'r^--',t_y, throughput_list[1], 'mD-',t_y, throughput_list[2],'#92D050',t_y, throughput_list[3],'bo--')
    #plt.legend(['Resnet50','VGG16', 'Mobilenet' , 'Mobilenetv2'], loc='lower right',prop={'size':10},labelspacing=0.2)
    plt.xlabel('Moving Ratio')
    plt.ylabel('Frames')
    #plt.xticks(batch_x_ticks)
    #plt.savefig('throughput6.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

videolist = ['nomotion.mp4', 'somemotion.mp4', 'highmotion.mp4', 'veryhighmotion.mp4']

motion_data = []

for video in videolist:
    print('video:', video)
    motion_data.append(extract_motion('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Test Images/'+video))

plotgraph(motion_data)
#print(a)
