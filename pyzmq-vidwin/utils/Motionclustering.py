import cv2
import pybgs
import time
import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
import os
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

def extract_motion(video_path):
    time_start = time.time()
    motion_ratio = []
    cap = cv2.VideoCapture(str(video_path))
    index = 0
    global bgs
    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (500,500), interpolation = cv2.INTER_AREA)
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
    plt.plot(t_y, throughput_list[0], 'r^--',t_y, throughput_list[1], 'm*-',t_y, throughput_list[2], 'gD-',t_y, throughput_list[3], 'bo-')
    # for i in range(len(throughput_list)):
    #     plt.plot(t_y,throughput_list[i])
    #plt.plot(t_y, throughput_list[0], 'r^--',t_y, throughput_list[1], 'mD-',t_y, throughput_list[2],'#92D050',t_y, throughput_list[3],'bo--')
    plt.legend(['Objects with no motion','Objects with motion at start', 'Objects with continuous motion','Sudden burst of objects with motion'], loc='upper right',prop={'size':10},labelspacing=0.2)
    plt.xlabel('Frames')
    plt.ylabel('Moving Ratio')
    plt.ylim(-4, 4)
    #plt.xticks(batch_x_ticks)
    #plt.savefig('throughput6.svg', format='svg', dpi=1000,bbox_inches='tight')
    plt.show()

#videolist = ['nomotion.mp4', 'somemotion.mp4', 'highmotion.mp4', 'veryhighmotion.mp4']
videolist = os.listdir('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Videosonbasisofmotion')

motion_data = []

for video in videolist:
    print('video:', video)
    motion_data.append(np.array(extract_motion('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Videosonbasisofmotion/'+video)))

#plotgraph(motion_data)

data = [data.reshape((data.size,1)) for data in motion_data]

newarray = np.dstack(data)
print(newarray.shape)
# To get the shape to be Nx10x10, you could  use rollaxis:
newarray = np.rollaxis(newarray,-1)
print(newarray.shape)
seed = 0
# Keep only 50 time series
X_train = TimeSeriesScalerMeanVariance().fit_transform(newarray[:280])
# Make time series shorter
#X_train = TimeSeriesResampler(sz=40).fit_transform(X_train)
sz = X_train.shape[1]


# Euclidean k-means
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=4, verbose=True, random_state=seed)
y_pred = km.fit_predict(X_train)

plt.figure()
for yi in range(4):
    #plt.subplot(2, 2, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.subplot(2, 2, yi + 1)
        plt.plot(xx.ravel(), color='#F5B14C', linestyle='-', alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), color='#2CBDFE', linestyle='-')
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    # plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
    #          transform=plt.gca().transAxes)
    if yi == 1:
        #plt.title("Euclidean $k$-means")
        print('OK')

# # DBA-k-means
# print("DBA k-means")
# dba_km = TimeSeriesKMeans(n_clusters=3,
#                           n_init=2,
#                           metric="dtw",
#                           verbose=True,
#                           max_iter_barycenter=10,
#                           random_state=seed)
# y_pred = dba_km.fit_predict(X_train)
#
# for yi in range(3):
#     plt.subplot(3, 3, 4 + yi)
#     for xx in X_train[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("DBA $k$-means")

# # Soft-DTW-k-means
# print("Soft-DTW k-means")
# sdtw_km = TimeSeriesKMeans(n_clusters=3,
#                            metric="softdtw",
#                            metric_params={"gamma": .01},
#                            verbose=True,
#                            random_state=seed)
# y_pred = sdtw_km.fit_predict(X_train)
#
# for yi in range(3):
#     plt.subplot(3, 3, 7 + yi)
#     for xx in X_train[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
#     plt.xlim(0, sz)
#     plt.ylim(-4, 4)
#     plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
#              transform=plt.gca().transAxes)
#     if yi == 1:
#         plt.title("Soft-DTW $k$-means")

#plt.tight_layout()
plt.savefig('movingcluster.svg',format='svg', dpi=1000, bbox_inches='tight')
plt.show()