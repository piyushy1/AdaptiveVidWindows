# Piyush Yadav

import os
import zmq
import time
import queue
import argparse
import datetime
import pickle as pk
from window import sliding
from multiprocessing import Process, Queue
import numpy
from dnnmodel import load_DNN_model
from dnnmodel import batch_of_images


# os.system('hostname -I')

def block(inp_q):
    try:
        while True:
            try:
                frame = inp_q.get(timeout=0.1)
                # print(frame, len(frame))
                if type(frame) == str and frame == 'END':
                    # out_q.put('END')
                    break
                #print(f'New block len- {len(frame)} and time start = {frame[0][1]} and end is = {frame[-1][1]}')
                print('New block length:', len(frame))
                del frame
                # if len(slide_window) == time_segment:
                #     out_q.put(slide_window)
                #     slide_window = slide_window[slide_time:]
                # else:
                #     slide_window.append(frame)
            except queue.Empty:
                pass
    except Exception as e:
        print(e)



def measure_batch_transmission_latency(batch,time):
    # only transmission latency
    trans_latency = (time-batch[-1][3]).total_seconds()*1000
    batch_plus_transmission_latency = (time-batch[0][3]).total_seconds()*1000
    return trans_latency, batch_plus_transmission_latency


def get_avg_edge_cpu_usage_batch(batch):
    cpu_usage = []
    for frame in batch:
        cpu_usage.append(frame[4])

    avgcpu = sum(cpu_usage)/ len(cpu_usage)
    return avgcpu


def get_avg_mem_usage_batch(batch):
    mem_usage = []
    for frame in batch:
        mem_usage.append(frame[5])

    avgmem = sum(mem_usage) / len(mem_usage)
    return avgmem


packs = []

def subscriber(ip="172.17.0.1", port=5551):
    # ZMQ connection
    #url = f"tcp://{ip}:{port}"
    url = "tcp://{}:{}".format(ip,port)
    # url = f"tcp://localhost:{port}"
    #print(f"Going to bind to: {url}")
    print("Going to bind to: {url}", url)
    ctx = zmq.Context()
    # socket = ctx.socket(zmq.SUB)
    # socket.bind(url)  # subscriber creates ZeroMQ socket
    # socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))  # any topic

    socket = ctx.socket(zmq.PAIR)
    socket.bind(url)  # connects to pub server

    print("Sub bound to: {}\nWaiting for data...".format(url))

    rc = 1
    
    sliding_window_input_queue = Queue()
    sliding_window_output_queue = Queue()
    sliding_process = Process(name='Slider',target=sliding, args=(sliding_window_input_queue, sliding_window_output_queue,5,2,))
    sliding_process.start()

    # block_process_input_queue = Queue()
    block_process = Process(name='Blocker', target=block, args=(sliding_window_output_queue,))
    block_process.start()

    # load the DNN Model
    model = load_DNN_model('mobilenet_custom')

    while True:
        msg = socket.recv()
        A = pk.loads(msg)

        if len(A) !=0:
            trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(A, datetime.datetime.now())
            batch_latency = batch_of_images(A,model)
            avg_cpu = get_avg_edge_cpu_usage_batch(A)
            avg_mem = get_avg_mem_usage_batch(A)
            print('The batch time is: ',rc, len(A), batch_latency)

        for i in A:
            if i[2] == 1:
                print("i frame received")
            sliding_window_input_queue.put(i)

        #print(f'Receive count = {rc}')
        print('Recieve Count:',rc)
        rc += 1

        # if rc%1 == 0:
        #     print('Average_Latency******', sum(latency) / len(latency) )
        #     latency.clear()


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
