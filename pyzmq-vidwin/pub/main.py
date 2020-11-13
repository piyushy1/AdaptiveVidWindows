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
from microbatchresizing import fixed_resizer, cloudseg_resizer
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
    print('Data Send**********', ctr)
    #compress
    #socket.send(pk.dumps(frame))
    #socket.send(zlib.compress(cPickle.dumps(frame)))
    #socket.send(cPickle.dumps(frame))
    socket.send(frame)
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
    batcher_process = Process(name='Batcher',target=batcher, args=(batch_input_queue, batch_output_queue,query_predicates,))
    batcher_process.start()
    # start micro-batch resizer
    batch_resizer_process = Process(name='Resizer',target=resizer, args=(batch_output_queue, resizer_output_queue,query_predicates,))
    batch_resizer_process.start()

    #batch filter process

    batch_filter_process = Process(name='LazyFilter', target=lazyfilter,
                                             args=(resizer_output_queue, filter_output_queue, query_predicates,))
    batch_filter_process.start()

    #batch differencer and compression process
    batch_differencer_process = Process(name='FrameDifferencer', target=fixed_differencer,
                                             args=(filter_output_queue, batch_differencer_queue,))
    batch_differencer_process.start()

    # send data to socket
    socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(batch_differencer_queue,url,))
    # socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(sliding_window_output_queue,url,))
    socket_send_window_process.start()
    # time.sleep(5)

    wind = []
    import time

    ctr = 1
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/personcar.mp4' #absolute path
    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Jacksonhole2391.mp4'  # absolute path
    video_path = '/app/video/segment114.mp4' # docker volume
    #video_path = '/app/video/personcar_clip.mp4'  # docker volume
    # video_path = '/app/video/soutampton_clip.mp4'

    # get the list of i frames..
    iframes_list = get_i_frames(video_path)

    #iframes_list = [1, 151, 300, 450, 600, 752, 901, 1052, 1201, 1351, 1502, 1652, 1801, 1952, 2101, 2253, 2402, 2551, 2702, 2852, 3001, 3152, 3304, 3453, 3603, 3753, 3902, 4052, 4203, 4353, 4502, 4653, 4803, 4952, 5102, 5253, 5403, 5554, 5703, 5855, 6003, 6154, 6304, 6454, 6603, 6753, 6904, 7054, 7204, 7354, 7504, 7653, 7804, 7955, 8105, 8254, 8404, 8555, 8705, 8854, 9005, 9155, 9304, 9455, 9605, 9755, 9904, 10055, 10205, 10356, 10506, 10655, 10805, 10956, 11105, 11255, 11406, 11556, 11705, 11855, 12006, 12155, 12306, 12455, 12606, 12758, 12907, 13056, 13207, 13357, 13506, 13657, 13807, 13957, 14106, 14256, 14407, 14556, 14706, 14857, 15007, 15158, 15308, 15458, 15607, 15757, 15908, 16058, 16207, 16359, 16508, 16657, 16807, 16958, 17107, 17257, 17409, 17559, 17708, 17859, 18009, 18158, 18308, 18459, 18608, 18758, 18909, 19059, 19208, 19359, 19509, 19660, 19810, 19959, 20109, 20260, 20409, 20559, 20710, 20859, 21010, 21160, 21309, 21461, 21610, 21759, 21910, 22060, 22211, 22360, 22511, 22661, 22810, 22961, 23110, 23260, 23411, 23561, 23710, 23861, 24011, 24160, 24310, 24461, 24612, 24762, 24911, 25061, 25212, 25361, 25511, 25662, 25813, 25962, 26112, 26261, 26412, 26562, 26713, 26863, 27012, 27163, 27313, 27462, 27612, 27763, 27912, 28062, 28214, 28363, 28513, 28662, 28813, 28963, 29114, 29264, 29413, 29564, 29714, 29864, 30013, 30164, 30314, 30463, 30613, 30764, 30914, 31063, 31214, 31364, 31515, 31664, 31814, 31965, 32115, 32264, 32415, 32565, 32714, 32864, 33015, 33164, 33314, 33465, 33615, 33764, 33916, 34065, 34215, 34366, 34515, 34665, 34816, 34966, 35115, 35265, 35415, 35566, 35715, 35865, 36016, 36166, 36317, 36467, 36616, 36766, 36917, 37067, 37216, 37367, 37517, 37666, 37816, 37967, 38117, 38266, 38417, 38567, 38718, 38867, 39018, 39168, 39317, 39467, 39619, 39768, 39917, 40067, 40218, 40368, 40517, 40667, 40818, 40969, 41119, 41268, 41418, 41569, 41719, 41868, 42019, 42169, 42318, 42469, 42618, 42769, 42919, 43068, 43220, 43369, 43519, 43670, 43819, 43970, 44119, 44269, 44420, 44570, 44719, 44869, 45020, 45170, 45319, 45470, 45620, 45771, 45921, 46070, 46221, 46370, 46521, 46671, 46820, 46970, 47120, 47270, 47420, 47571, 47720, 47872, 48021, 48172, 48322, 48472, 48621, 48772, 48922, 49071, 49221, 49372, 49522, 49671, 49822, 49972, 50122, 50274, 50423, 50572, 50724, 50873, 51023, 51172, 51323, 51473, 51622, 51772, 51923, 52073, 52222, 52373, 52523, 52673, 52823, 52974, 53124, 53273, 53425, 53574, 53723, 53874, 54024, 54173, 54324, 54474, 54624, 54773, 54924, 55074, 55225, 55375, 55524, 55674, 55825, 55974, 56124, 56275, 56425, 56574, 56725, 56875, 57024, 57174, 57325, 57475, 57626, 57776, 57925, 58075, 58226, 58375, 58525, 58676, 58826, 58975, 59126, 59276, 59425, 59575, 59726, 59876] #temp fix for conntainer

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
