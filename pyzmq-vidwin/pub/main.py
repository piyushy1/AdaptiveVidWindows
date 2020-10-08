# Name Piyush Yadav

import os
import cv2
import zmq
import time
import queue
import psutil
import json
import argparse
import datetime
import numpy as np
import pickle as pk
from window import sliding
from probe import get_i_frames
from videostreamer import stream
from multiprocessing import Process, Queue

# os.system('hostname -I')

## PSUTIL functions#########################
# Note: psutil fxn dont work inside docker container....the below fxns are kept for future ref.

def get_cpu():
    cpu = ''
    for x in range(3):
        cpu = cpu + str(psutil.cpu_percent(interval=1))+' '
    return cpu

def get_cpu_percent():
    cpu = psutil.cpu_percent(interval=1)
    return cpu


def get_used_mem_bytes():
    return psutil.virtual_memory().used /(1024*1024)

def get_used_mem_percentage():
    return psutil.virtual_memory().percent

def get_available_memory():
    return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

#################################################################################


def calculate_container_CPU_Percent():
    # fetch running container id
    container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
    try:
        f = open("/docker_stats/" + container_id.rstrip() + '.json', "r")
        data = json.loads(f.read())
        cpuDelta = data['cpu_stats']['cpu_usage']['total_usage'] - data['precpu_stats']['cpu_usage']['total_usage']
        systemDelta = data['cpu_stats']['system_cpu_usage'] - data['precpu_stats']['system_cpu_usage']
        if systemDelta > 0.0 and cpuDelta > 0.0:
            cpuPercent = (cpuDelta / systemDelta) * float(len(data['cpu_stats']['cpu_usage']['percpu_usage'])) * 100.0
            return cpuPercent
            #print('CPU Usage ==> ' + str(cpuPercent) + ' %')
    except Exception as e:
        # print(str(e))
        print('File not created yet. Retrying...')


def calculate_container_memory():
    # fetch running container id
    container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
    try:
        mem_percent = []
        f = open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", "r")
        with open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", 'r') as infile:
            mem_usage =float(infile.read()) / (1024 * 1024) # mem in MB
            mem_percent.append(mem_usage)
            #print('MEM USAGE', mem_usage)

        with open("/mem/" + container_id.rstrip() + "/memory.limit_in_bytes", 'r') as infile:
            mem_limit = float(infile.read())/(1024 * 1024) # max memory limit of container
            mem_percent.append(mem_limit)
            #print('MEM LIMIT', mem_limit)

        mem_usage_percent = (mem_percent[0]/mem_percent[1])*100
        return mem_usage_percent
        #print('MEM PERCENT %', mem_usage_percent)

    except Exception as e:
        print('container id not found.....'+str(e))


def socket_send(frame, socket):
    # md = dict(
    #         dtype = str(frame.dtype)
    #         shape = frame.shape,
    #     )
    # socket.send_json(md)
    socket.send(pk.dumps(frame))

def socket_send_window(q, url):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.connect(url)

    while True:
        # print('sended..')
        try:
            new_block = q.get(timeout=0.1)
            if type(new_block) == str and new_block == 'END':
                break
            print(len(new_block))
            socket_send(new_block, socket)
            # socket.send(new_block)
        except queue.Empty:
            pass

import random

# this function takes the resized microbatch as argument and return a differences while maintaining
# keyframes.
def create_diff_batch(frames):
    diff_batch = []
    keyframe = None
    i = 0
    for frame in frames:
        if i == 0:
            keyframe = frame[0]
            diff_batch.append(frame)
        else:
            match_mask = (keyframe == frame)
            idx_unmatched = np.argwhere(~match_mask)
            idx_values = frame[tuple(zip(*idx_unmatched))]
            frame[0] = [idx_unmatched, idx_values]
            diff_batch.append(frame)
        i = i + 1

    return diff_batch


def batcher(inp_q, out_q):
    frames = []
    while True:
        try:
            new_frame = inp_q.get()
            # print(new_frame)
            if len(frames) == 5 or new_frame[2] == 1:
                # put random batches
                idx = random.randint(5,15)
                #out_q.put(frames[:int(idx/2)] + frames[int(3*idx/2):])
                frame_diff_batch = create_diff_batch(frames)
                out_q.put(frame_diff_batch)
                print('put')
                frames = []
            frames.append(new_frame)
        except queue.Empty:
            pass


g = None

def publisher(ip="0.0.0.0", port=5551):
    # ZMQ connection
    url = f"tcp://{ip}:{port}"
    print("Going to connect to: {}".format(url))
    print("Pub connected to: {}\nSending data...".format(url))

    batch_input_queue = Queue()
    batch_output_queue = Queue()
    batcher_process = Process(name='Batcher',target=batcher, args=(batch_input_queue, batch_output_queue,))
    batcher_process.start()

    socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(batch_output_queue,url,))
    # socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(sliding_window_output_queue,url,))
    socket_send_window_process.start()
    # time.sleep(5)

    wind = []
    import time
    ctr = 1
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2.mp4'
    video_path = '/app/video/test2.mp4'
    iframes_list = get_i_frames(video_path)
    print('The iframe list***************', iframes_list)

    for frame in stream(video_path):

        if ctr in iframes_list:
            batch_input_queue.put([frame,ctr,1,datetime.datetime.now(),calculate_container_CPU_Percent(),calculate_container_memory()])  # an iframe
        else:
            batch_input_queue.put([frame,ctr,0,datetime.datetime.now(),calculate_container_CPU_Percent(),calculate_container_memory()])

        #calculate_container_CPU_Percent()
        #calculate_container_memory()


        ctr += 1
        #time.sleep(0.1)


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=argparse.SUPPRESS,
                        help="IP of (Docker) machine")
    parser.add_argument("--port", default=argparse.SUPPRESS,
                        help="Port of (Docker) machine")

    args, leftovers = parser.parse_known_args()
    print("The following arguments are used: {}".format(args))
    print("The following arguments are ignored: {}\n".format(leftovers))

    # call function and pass on command line arguments
    try:
        publisher(**vars(args))
    except KeyboardInterrupt:
        print("Exiting publisher...")
        pass
