from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import random

def clip_video(start, end, number):
    ffmpeg_extract_subclip("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar.mp4",
                       start, end, targetname = "/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar_clip.mp4")


clip_video(23,35,1)


# random_time = random.sample(range(10, 20000), 118)
#
# for index, time in enumerate(random_time):
#
#     str_number=''
#     if index < 10:
#         str_number = '00'+str(index)
#     elif index < 99:
#         str_number = '0'+str(index)
#     else:
#         str_number = str(index)
#
#     clip_video(time, time+10, str_number)

# import cv2
#
# cap = cv2.VideoCapture('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test3.mp4')
# i =0
# while True:
#     ret, frame = cap.read()
#     if i ==1:
#         break
#     frame = cv2.resize(frame, (900, 900))
#     cv2.imwrite('resol900.jpg', frame)
#     i = i+1


# import numpy as np
#
# def create_diff_batch(frames):
#     diff_batch = []
#     keyframe = None
#     i = 0
#     for frame in frames:
#         if i==0:
#             keyframe =frame[0]
#             diff_batch.append(frame)
#         else:
#             match_mask = (keyframe == frame)
#             idx_unmatched = np.argwhere(~match_mask)
#             idx_values = frame[tuple(zip(*idx_unmatched))]
#             frame[0] =[idx_unmatched, idx_values]
#             diff_batch.append(frame)
#         i=i+1
#
#     return diff_batch
#
#
# a = np.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])
# b = np.array([[[1, 2,3], [3, 4,8]], [[5, 6,7], [7, 8,10]]])
# c = np.array([[[1, 2,3], [3, 4,8]], [[5, 12,7], [7, 11,10]]])
#
# frames = [[a,1,2], [b,1,2], [c,1,2]]
#
# create_diff_batch(frames)

#match_mask = (base_array == array_1)
#idx_unmatched = np.argwhere(~match_mask)

#print(idx_unmatched)


# unmatched values, e.g.:
#print(base_array[tuple(idx_unmatched[0])])

# import math
#
# def calculate_entropy_score(list_event_score):
#
#     output = math.log2(4)


from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np

def get_marker_list():
    marker_list =['o','*','s','v','D','1','d','3','4']
    return marker_list

def get_colorlist():
    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
                  CB91_Purple, CB91_Violet]
    return color_list

def micro_batch_utility(window_length):
    entropylist =[]
    batchlist = [1000]
    for batch in batchlist:
        batch_entropy =[]
        for i in range(0, len(window_length), batch):
            win = window_length[i:i+batch]
            win_position = 1-(i/len(window_length))
            batch_size = 1- (len(win)/ len(window_length[i:]))
            batch_entropy.append(entropy([win_position, batch_size], base=2))
            entropylist.append(batch_entropy)
            #print('Window_Position: ', win_position, 'Batch_Size: ',batch_size, 'Entropy: ', entropy([win_position, batch_size], base=2))

    #t_y = np.array(batchlist)
    a = get_colorlist()
    b = get_marker_list()
    for i in range(0,2):
        plt.plot(entropylist[0], color = a[i], linestyle = '-', marker = b[i], markersize = 7)
    #plt.plot(entropylist[1], 'b+--')
    #plt.plot(entropylist[2])
    #plt.plot(entropylist[3])
    plt.title('Batch Size:' + str(batchlist[0])+' Window Size:'+ str(10000))
    plt.show()




win = []
for i in range(0,10000):
    win.append('ok')

#micro_batch_utility(win)

#print(entropy([1/2,1/1000], base=2))



# a = [{'person': 0.3494016}, {'car': 0.3192894},{'car1': 0.2192894},{'car2': 0.1192894}]
# b =['car','person']
#
# score = []
# for d in a:
#     for key in d:
#         score.append(d[key])
# ratio=[]
# for i in range(0,len(score)-1):
#     print(i)
#     ratio.append(score[i+1]/score[i])
#
# print(ratio)

# from pympler.asizeof import asizeof
# import sys
# import numpy as np
# b = np.full((10000,1000),10)
# a = [b,b,b,b]
# c= [1]
#
# print('Size of',sys.getsizeof(a), sys.getsizeof(c),sys.getsizeof(b), asizeof(a))
#
# indexes = [i for i in range(1,20,2)]
# print(indexes)

def create_diff_batch(frames):
    try:
        diff_batch = []
        keyframe = None
        i = 0
        for frame in frames:
            if i == 0:
                keyframe = frame[0]
                diff_batch.append(frame)
            else:
                #print('Memory of Frame********************8', asizeof(frame[0])/(1024*1024))
                match_mask = (keyframe == frame[0])
                idx_unmatched = np.argwhere(~match_mask).astype('uint8')
                #print('Data TYPE***********',idx_unmatched.dtype)
                idx_values = frame[0][tuple(zip(*idx_unmatched))]
                #print('Memory of DIFF********************8', asizeof([idx_unmatched, idx_values])/(1024*1024))
                frame[0] = [idx_unmatched, idx_values]
                #frame[0] = [np.c_[idx_unmatched,idx_values].astype('int8')]
                #frame[0] = [idx_unmatched]
                #frame = frame[1:]d
                diff_batch.append(frame)
            i = i + 1
        # print('Length********', len(frames), len(diff_batch))
        return diff_batch
    except Exception as e:
        print('Exception**********************'+str(e))


def recreate_diff_batch(frames):
    try:
        reconst_batch = []
        keyframe = None
        i = 0
        for frame in frames:
            if i == 0:
                keyframe = frame[0]
                reconst_batch.append(frame)
            else:
                idx, vals = frame[0]
                new_frame = keyframe
                new_frame[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
                frame[0] = new_frame
                reconst_batch.append(frame)
            i = i + 1
        # print('Length********', len(frames), len(diff_batch))
        return reconst_batch
    except Exception as e:
        print('Exception**********************'+str(e))



import cv2
def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float

    print('width, height:', width, height)
    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    i =0
    frame_batch = []
    while True:
        framel = []
        # Capture frame-by-frame

        ret, frame = cap.read()
        frame = cv2.resize(frame,(224,224))
        framel.append(frame)
        if i >= 1:
            print('frame: '+str(i))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # thi is done as openc v read bgr while pil read rgb
            #frame = Image.fromarray(frame)
            frame_batch.append(framel)

            if len(frame_batch) == 5:
                diff = create_diff_batch(frame_batch)
                reconst_batch = recreate_diff_batch(diff)
                if frame_batch == reconst_batch:
                    print("The lists are identical")
                else:
                    print("The lists are not identical")
                #print(diff,reconst_batch)
                frame_batch =[]
        i=i+1
        if not ret:
            break
    cap.release()

#object_detection_api('/home/dhaval/Desktop/Car-Image.jpg')
#stream('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar.mp4')