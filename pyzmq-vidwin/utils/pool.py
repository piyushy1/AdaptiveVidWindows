import numpy as np
import queue
import _pickle as cPickle
import zlib
import datetime
import queue
from multiprocessing import Process, Queue,Pool
import cv2
import threading
import multiprocessing

def get_time_milliseconds():
    time = (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time


def do_in_new_process(keyframe, frame, index):
    #print('batch:'+str(batch_no)+', frame:'+str(frame_no))
    match_mask = (keyframe[0] == frame[0])
    idx_unmatched = np.argwhere(~match_mask).astype('uint8')  # get index of unmatched value
    idx_values = frame[0][tuple(zip(*idx_unmatched))]  # get corresponding value of index
    frame[0] = [idx_unmatched, idx_values]
    return (index, frame)


# this function takes the resized batch as argument and return a differences while maintaining
# keyframes.
def create_diff_batch(inp_q, out_q):
    batch_no = 0
    with multiprocessing.Pool(processes=10) as pool:
        while True:
            try:
                frame_batch = inp_q.get(timeout=0.01)
                frame_batch = [(frame_batch[0], frame, index) for index, frame in enumerate(frame_batch)]
                diff_batch = []
                batch_no = batch_no + 1
                a = get_time_milliseconds()
                # i = 0
                keyframe = frame_batch[0][0]
                diff_batch.append(keyframe)
                # frame_pool_batch.append((batch_no, i, frame_batch[0], frame_batch[1:]))
                # for frame in frame_batch:
                #     if i == 0:
                #         keyframe = frame[0]
                #         diff_batch.append(frame)
                #     else:
                #         frame_pool_batch.append((batch_no, i, keyframe, frame))
                #     i = i + 1
                processed_batch = pool.starmap(do_in_new_process, frame_batch[1:])
                for ff in processed_batch:
                    print(ff[0])
                #diff_batch = diff_batch + processed_batch
                #print('Batch:' + str(batch_no) + ', Comparison time*******************************',(get_time_milliseconds() - a) / 1000)
                #out_q.put(zlib.compress(cPickle.dumps(diff_batch)))
            except queue.Empty:
                pass


batch_queue = Queue()
batch_differencer_queue = Queue()

# start differencer
# pool = Pool()
# pool.map(create_diff_batch, filter_output_queue)
# pool.close()

batcher_process = Process(name='Difference',target=create_diff_batch, args=(batch_queue, batch_differencer_queue,))
batcher_process.start()

# give video path
cap = cv2.VideoCapture('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar_clip.mp4')

frame_batch = []
while True:
    ret, frame = cap.read()

    if frame is None:
        break

    frame = cv2.resize(frame, (500, 500))
    frame_batch.append([frame,0])

    if len(frame_batch) >= 20:
        batch_queue.put(frame_batch)
        frame_batch = []
