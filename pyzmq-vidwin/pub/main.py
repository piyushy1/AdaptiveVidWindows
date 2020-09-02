# Name Piyush Yadav

import os
import cv2
import zmq
import time
import queue
import psutil
import argparse
import datetime
import numpy as np
import pickle as pk
from window import sliding
from probe import get_i_frames
from videostreamer import stream
from multiprocessing import Process, Queue
# os.system('hostname -I')

def get_cpu():
    return psutil.cpu_percent()

def get_vram():
    return psutil.virtual_memory().percent

def get_available_memory():
    return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

def socket_send(frame, socket):
    # md = dict(
    #         dtype = str(frame.dtype),
    #         shape = frame.shape,
    #     )
    # socket.send_json(md)
    socket.send(pk.dumps(frame))

def socket_send_window(q, url):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PAIR)
    socket.connect(url)
    # print('sendin..')
    # socket.send_string("omg here")
    # socket.send_string("omg here")

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

def batcher(inp_q, out_q):
    frames = []
    while True:
        try:
            new_frame = inp_q.get()
            # print(new_frame)
            if len(frames) == 50 or new_frame[2] == 1:
                # put random batches
                idx = random.randint(5,15)
                #out_q.put(frames[:int(idx/2)] + frames[int(3*idx/2):])
                out_q.put(frames)
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

    video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2.mp4'
    #video_path = '/data1/test2.mp4'
    iframes_list = get_i_frames(video_path)
    print('The iframe list***************', iframes_list)
    for frame in stream(video_path):
        if ctr in iframes_list:
            batch_input_queue.put([frame,ctr,1])  # an iframe
        else:
            batch_input_queue.put([frame,ctr,0])
        print('CPU, MEMORY: ',get_cpu(), get_vram(), get_available_memory())
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
