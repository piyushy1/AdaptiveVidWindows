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
from microbatching import batcher
from microbatchresizing import resizer
from microbatching import fixed_batcher
from microbatchresizing import fixed_resizer
from window import sliding
from probe import get_i_frames
from videostreamer import stream
from videoquery import parse_query
from multiprocessing import Process, Queue
import sys

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

import _pickle as cPickle
import zlib


def socket_send(frame, socket):
    # md = dict(
    #         dtype = str(frame.dtype)
    #         shape = frame.shape,
    #     )
    # socket.send_json(md)
    #a = np.array(frame, dtype=object)
    #print('rame size***********', a.nbytes)
    #print('MEMORY SIZE************************************************',asizeof(frame)/(1024*1024),asizeof(a)/(1024*1024),asizeof(zlib.compress(cPickle.dumps(frame)))/(1024*1024),asizeof(zlib.compress(cPickle.dumps(a)))/(1024*1024))
    print('Data Send')
    socket.send(pk.dumps(frame))
    #socket.send(frame)

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
            #print(len(new_block))
            socket_send(new_block, socket)
            # socket.send(new_block)
        except queue.Empty:
            pass

import random

# this function takes the resized microbatch as argument and return a differences while maintaining
# keyframes.
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

#from pympler.asizeof import asizeof

g = None

def get_time_milliseconds():
    time = (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time

def publisher(ip="0.0.0.0", port=5551):
    # ZMQ connection
    #url = f"tcp://{ip}:{port}"
    url = "tcp://{}:{}".format(ip, port)
    print("Going to connect to: {}".format(url))
    print("Pub connected to: {}\nSending data...".format(url))

    # fetch the query predicates
    query_predicates = parse_query()

    batch_input_queue = Queue()
    batch_output_queue = Queue()
    resizer_output_queue = Queue()
    # start micro-batcher
    batcher_process = Process(name='Batcher',target=batcher, args=(batch_input_queue, batch_output_queue,query_predicates,))
    batcher_process.start()
    # start micro-batch resizer
    batch_resizer_process = Process(name='Resizer',target=resizer, args=(batch_output_queue, resizer_output_queue,query_predicates,))
    batch_resizer_process.start()
    # send data to socket
    socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(resizer_output_queue,url,))
    # socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(sliding_window_output_queue,url,))
    socket_send_window_process.start()
    # time.sleep(5)

    wind = []
    import time


    ctr = 1
    video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2.mp4' #absolute path
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Jacksonhole2391.mp4'  # absolute path
    #video_path = '/app/video/test2.mp4' # docker volume

    # get the list of i frames..
    iframes_list = get_i_frames(video_path)
    #iframes_list = [1, 151, 301, 451, 601, 751, 901, 1051] #temp fix for conntainer
    print('The iframe list***************', iframes_list)

    for frame in stream(video_path):
        #get_bw_from_server()
        # if frame is i frame put i frame info
        if ctr in iframes_list:
            #batch_input_queue.put([frame,ctr,1,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory()])  # an iframe
            batch_input_queue.put([frame, ctr, 1,get_time_milliseconds()])  # an iframe
        else:
            #batch_input_queue.put([frame,ctr,0,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory()])
            batch_input_queue.put([frame, ctr, 0,get_time_milliseconds()])

        #calculate_container_CPU_Percent()
        #calculate_container_memory()


        ctr += 1
        #time.sleep(0.1)


def get_bw_from_server():
    import iperf3

    client = iperf3.Client()
    client.server_hostname = '10.5.0.5'
    client.port = 6969
    client.json_output = True
    print('Fetching Bandwidth...')
    result = client.run()
    print('  Gigabits per second  (Gbps)  {0}'.format(result.sent_Mbps / 1024))


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
