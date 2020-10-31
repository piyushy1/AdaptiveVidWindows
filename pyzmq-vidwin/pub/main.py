# Name Piyush Yadav

import os
from multiprocessing.context import Process

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
from microbatchfiltering import lazyfilter
from microbatching import fixed_batcher
from microbatchresizing import fixed_resizer
from window import sliding
from probe import get_i_frames
from videostreamer import stream
from videoquery import parse_query
from microbatchfiltering import fixed_filter
from microbatchdifferencer import create_diff_batch, fixed_differencer
from microbatchfiltering import calculate_container_CPU_Percent, calculate_container_memory_Percent
from multiprocessing import Process, Queue
import _pickle as cPickle
import zlib
from pympler import asizeof
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


def socket_send(frame, socket,ctr):

    #print('MEMORY SIZE************************************************',asizeof(frame)/(1024*1024),asizeof(a)/(1024*1024),asizeof(zlib.compress(cPickle.dumps(frame)))/(1024*1024),asizeof(zlib.compress(cPickle.dumps(a)))/(1024*1024))
    # send the difference
    # a = get_time_milliseconds()
    # frame = create_diff_batch(frame)
    # print('Data Send***************************************************',ctr, (get_time_milliseconds()-a)/1000)
    print('Data Send***************************************************', ctr)
    #compress
    #socket.send(pk.dumps(frame))
    #socket.send(zlib.compress(cPickle.dumps(frame)))
    socket.send(cPickle.dumps(frame))
    # socket.send(frame)
    #socket.send_string(frame, zmq.NOBLOCK)
    #socket.send(frame)

def socket_send_window(q, url):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(url)
    snd_ctr= 1
    while True:
        # print('sended..')
        try:
            new_block = q.get(timeout=None)
            if type(new_block) == str and new_block == 'END':
                break
            #print(len(new_block))
            socket_send(new_block, socket,snd_ctr)

            snd_ctr+=1

            # socket.send(new_block)
        except queue.Empty:
            pass


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
    filter_output_queue = Queue()
    batch_differencer_queue = Queue()
    # start micro-batcher
    batcher_process = Process(name='Batcher',target=fixed_batcher, args=(batch_input_queue, batch_output_queue,query_predicates,))
    batcher_process.start()
    # start micro-batch resizer
    batch_resizer_process = Process(name='Resizer',target=fixed_resizer, args=(batch_output_queue, resizer_output_queue,query_predicates,))
    batch_resizer_process.start()

    #batch filter process

    batch_filter_process = Process(name='LazyFilter', target=fixed_filter,
                                             args=(resizer_output_queue, filter_output_queue, query_predicates,))
    batch_filter_process.start()

    #batch differencer and compression process
    # batch_differencer_process = Process(name='FrameDifferencer', target=fixed_differencer,
    #                                          args=(filter_output_queue, batch_differencer_queue,))
    # batch_differencer_process.start()

    # send data to socket
    socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(filter_output_queue,url,))
    # socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(sliding_window_output_queue,url,))
    socket_send_window_process.start()
    # time.sleep(5)

    wind = []
    import time

    ctr = 1
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar.mp4' #absolute path
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Jacksonhole2391.mp4'  # absolute path
    video_path = '/app/video/auburn_clip3min.mp4' # docker volume
    #video_path = '/app/video/personcar_clip.mp4'  # docker volume
    # video_path = '/app/video/soutampton_clip.mp4'

    # get the list of i frames..
    iframes_list = get_i_frames(video_path)
    #iframes_list = [1, 151, 301, 451, 601, 751, 901, 1051] #temp fix for conntainer
    print('The iframe list***************', iframes_list)

    for frame in stream(video_path):
        #get_bw_from_server()
        # if frame is i frame put i frame info
        if ctr in iframes_list:
            #batch_input_queue.put([frame,ctr,1,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory()])  # an iframe
            batch_input_queue.put([frame, ctr, 1,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory_Percent()])  # an iframe
            #print('Netowrk Data MEMORY', calculate_packet_transfered())
        else:
            #batch_input_queue.put([frame,ctr,0,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory()])
            batch_input_queue.put([frame, ctr, 0,get_time_milliseconds(),calculate_container_CPU_Percent(),calculate_container_memory_Percent()])
            #print('Netowrk Data MEMORY', calculate_packet_transfered())
        #print('Batch input queue******', batch_input_queue.qsize())
        #print('Resizer Input queue******', batch_output_queue.qsize())
        #print('Filter input queue******', resizer_output_queue.qsize())
        #print('DIfferencer  queue******', filter_output_queue.qsize())
        #print('Socket input
        # queue******', batch_differencer_queue.qsize())

        ctr += 1



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
