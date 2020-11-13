import numpy as np
import queue
import _pickle as cPickle
import zlib
import datetime
import multiprocessing
from sys import getsizeof


def get_time_milliseconds():
    time = (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time


def do_in_new_process(keyframe, frame):
    match_mask = (keyframe[0] == frame[0])
    idx_unmatched = np.argwhere(~match_mask).astype('uint8')  # get index of unmatched value
    idx_values = frame[0][tuple(zip(*idx_unmatched))]  # get corresponding value of index
    frame[0] = [idx_unmatched, idx_values]
    #frame[0] = [np.c_[idx_unmatched, idx_values].astype('int8')]
    #frame[0] = [idx_unmatched]
    return frame


def create_diff_batch(inp_q, out_q):
    batch_no = 0
    with multiprocessing.Pool(processes=8) as pool:
        while True:
            try:
                frame_batch = inp_q.get(timeout=0.01)
                print('Frame Batch Size ', getsizeof(frame_batch))
                frame_batch = [(frame_batch[0], frame) for frame in frame_batch]
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
                diff_batch[1:] = processed_batch
                print('Diff Batch Size ', getsizeof(diff_batch))
                #print('processed ATCH**********', processed_batch)
                # print('Batch:' + str(batch_no) + ', Comparison time*******************************',
                #       (get_time_milliseconds() - a) / 1000)
                out_q.put(cPickle.dumps(diff_batch)) # without compression
                #out_q.put(zlib.compress(cPickle.dumps(diff_batch)))  # with compression
            except queue.Empty:
                pass

# for EVALUATION... do fixed filtering or no filtering
def fixed_differencer(inp_q, out_q):
    while True:
        try:
            new_micro_batch = inp_q.get(timeout=None)
            out_q.put(zlib.compress(cPickle.dumps(new_micro_batch))) # with compression
            #out_q.put(cPickle.dumps(new_micro_batch)) # without

        except queue.Empty:
            pass