# Name Piyush Yadav

import argparse
import zmq
import time
import os
os.system('hostname -I')
import psutil
import numpy as np
import cv2
from videostreamer import stream
from window import sliding
from multiprocessing import Process, Queue
import queue
import pickle as pk

def get_cpu():
    return psutil.cpu_percent()

def get_vram():
    return psutil.virtual_memory().percent

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
            if len(frames) == 25:
                idx = random.randint(5,15)
                out_q.put(frames[:int(idx/2)] + frames[int(3*idx/2):])
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
    # ctx = zmq.Context()
    # socket = ctx.socket(zmq.PUB)
    # socket = ctx.socket(zmq.PAIR)
    # socket.connect(url)  # publisher connects to subscriber
    print("Pub connected to: {}\nSending data...".format(url))
    i = 0

    # socket.send_string("Testing")

    # sliding_window_input_queue = Queue()
    # sliding_window_output_queue = Queue()
    # sliding_process = Process(name='Slider',target=sliding, args=(sliding_window_input_queue, sliding_window_output_queue,5,2,))
    # sliding_process.start()

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
    ctr = 0
    for frame in stream('midas.mp4'):
        # sliding_window_input_queue.put(frame)
        global g
        g = frame
        batch_input_queue.put([frame,ctr])
        # socket_send( (frame,ctr) , socket)
        ctr += 1
        # if len(wind) == 100:
        #     st = time.time()
        #     socket_send(np.array(wind), socket)
        #     wind = []
        #     break
        # wind.append(frame)
        time.sleep(0.1)
    # print(time.time() - st)
    # st = time.time()
    # for i in wind:
    #     socket_send(i, socket)
    

        # print(gc.get_stats())
        # gc.collect()

    # sliding_window_input_queue.put('END')
    # sliding_process.join()
    # socket_send_window_process.join()


    # socket.send_json(md, 0|zmq.SNDMORE)

    # while True:
    #     topic = 'foo'.encode('ascii')
    #     msg = f'test {i}'.encode('ascii')
        
    #     usage_topic = 'usage'.encode('ascii')
    #     socket.send_multipart([topic, msg])  # 'test'.format(i)
    #     time.sleep(.5)
    #     i += 1
    #     print(i)



    #     # publish data
    #     socket.send_multipart([topic, msg*200])  # 'test'.format(i)
    #     socket.send_multipart([usage_topic, f'CPU - {get_cpu()}'.encode('ascii')])  # 'test'.format(i)
    #     socket.send_multipart([usage_topic, f'VRAM - {get_vram()}'.encode('ascii')])  # 'test'.format(i)
    #     print(f"On topic {topic}, send data: {msg}")
    #     # print(f"On topic {usage_topic}, send data: {}")
    #     # print(f"On topic {usage_topic}, send data: {f'{get_vram()}'.encode('ascii')}")


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
