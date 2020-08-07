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

def get_cpu():
    return psutil.cpu_percent()

def get_vram():
    return psutil.virtual_memory().percent

def socket_send_window(q, url):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(url)
    # print('sendin..')
    # socket.send_string("omg here")
    # socket.send_string("omg here")

    # print('sended..')
    while True:
        try:
            new_block = np.array(q.get(timeout=0.1))
            if type(new_block) == str and new_block == 'END':
                break
            print(new_block.shape)
            md = dict(
                dtype = str(new_block.dtype),
                shape = new_block.shape,
            )
            socket.send_json(md)
            socket.send(new_block)
        except queue.Empty:
            pass


def publisher(ip="0.0.0.0", port=5551):
    # ZMQ connection
    url = "tcp://{}:{}".format(ip, port)
    print("Going to connect to: {}".format(url))
    # ctx = zmq.Context()
    # socket = ctx.socket(zmq.PUB)
    # socket.connect(url)  # publisher connects to subscriber
    print("Pub connected to: {}\nSending data...".format(url))
    i = 0



    sliding_window_input_queue = Queue()
    sliding_window_output_queue = Queue()
    sliding_process = Process(name='Slider',target=sliding, args=(sliding_window_input_queue, sliding_window_output_queue,5,2,))
    sliding_process.start()
    socket_send_window_process = Process(name='Socket Sender ',target=socket_send_window, args=(sliding_window_output_queue,url,))
    socket_send_window_process.start()
    # time.sleep(5)
    for frame in stream('midas.mp4'):
        sliding_window_input_queue.put(frame)
        time.sleep(0.2)
    sliding_window_input_queue.put('END')
    sliding_process.join()
    socket_send_window_process.join()


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
