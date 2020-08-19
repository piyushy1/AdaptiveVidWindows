# Piyush Yadav

import time
import argparse
import zmq
import os
import numpy
import queue
from window import sliding
from multiprocessing import Process, Queue
import datetime
# os.system('hostname -I')
# a = []
import pickle as pk
# gc.set_debug(gc.DEBUG_LEAK)

def block(inp_q):
    # import gc
    while True:
        try:
            frame = inp_q.get(timeout=0.1)
            # print(frame, len(frame))
            # print(gc.get_stats())
            if type(frame) == str and frame == 'END':
                # out_q.put('END')
                break
            print(f'New block len- {len(frame)} and time start = {frame[0][1]} and end is = {frame[-1][1]}')
            del frame
            # if len(slide_window) == time_segment:
            #     out_q.put(slide_window)
            #     slide_window = slide_window[slide_time:]
            # else:
            #     slide_window.append(frame)
        except queue.Empty:
            pass

latency =[]
def measure_latency(batch,time):
    # only transmission latency
    latency.append((time-batch[-1][2]).total_seconds()*1000)


    # avg latency of batch plus transmission
    # for i in batch:
    #     latency.append((time-i[2]).total_seconds()*1000)

packs = []

def subscriber(ip="0.0.0.0", port=5551):
    # ZMQ connection
    url = f"tcp://{ip}:{port}"
    # url = f"tcp://localhost:{port}"
    print(f"Going to bind to: {url}")
    ctx = zmq.Context()
    # socket = ctx.socket(zmq.SUB)
    # socket.bind(url)  # subscriber creates ZeroMQ socket
    # socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))  # any topic

    socket = ctx.socket(zmq.PAIR)
    socket.bind(url)  # connects to pub server

    # print(socket.recv_string())
    print("Sub bound to: {}\nWaiting for data...".format(url))

    # print(socket.recv_string())
    # print(socket.recv_string())   
    # socket.send_json({"this":"and that"})
    # time.sleep(3)
    rc = 1
    
    sliding_window_input_queue = Queue()
    sliding_window_output_queue = Queue()
    sliding_process = Process(name='Slider',target=sliding, args=(sliding_window_input_queue, sliding_window_output_queue,5,2,))
    sliding_process.start()

    # block_process_input_queue = Queue()
    block_process = Process(name='Blocker', target=block, args=(sliding_window_output_queue,))
    block_process.start()

    while True:
        # md = socket.recv_json()
        # print(md)
        # continue
        msg = socket.recv()
        A = pk.loads(msg)
        #measure_latency(A,datetime.datetime.now())

        for i in A:
            sliding_window_input_queue.put(i)

        # packs.append(A)
        # for i in A:
        #     print(i[1])
        # buf = memoryview(msg)
        # A = numpy.frombuffer(buf, dtype=md['dtype']).reshape(md['shape'])
        # a.append(A.reshape(md['shape']))
        # print(A.reshape(md['shape']))
        # sliding_window_input_queue.put(A)
        # print(gc.get_stats())
        # gc.collect()
        print(f'Receive count = {rc}')
        rc += 1

        # if rc%1 == 0:
        #     print('Average_Latency******', sum(latency) / len(latency) )
        #     latency.clear()


        # wait for publisher data
        # print(socket.recv())
        # topic, msg = socket.recv_multipart()
        # print("On topic {}, received data: {}".format(topic, msg))


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
    print('test')

    # call function and pass on command line arguments
    subscriber(**vars(args))
