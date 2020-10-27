# Piyush Yadav

import os
import zmq
import time
import queue
import threading
import argparse
import datetime
import pickle as pk
from window import sliding
from multiprocessing import Process, Queue
import numpy
from dnnmodel import load_DNN_model
from dnnmodel import batch_of_images
from faster_rcnn import load_fasterrcnn_model
from faster_rcnn import get_prediction
from matcher import cepmatcher
import csv
import _pickle as cPickle
import zlib
import psutil
# os.system('hostname -I')


def dnn_input_object_detection(inp_q, out_q):
    # load the DNN Model
    model = load_fasterrcnn_model()
    #model = load_DNN_model('InceptionResNet50')
    try:
        while True:
            try:
                frame_batch = inp_q.get(timeout=0.1)
                batch_process_time, pred = get_prediction(frame_batch, model)
                for processed_frame in pred:
                    out_q.put(processed_frame)

            except queue.Empty:
                pass
    except Exception as e:
        print(e)

def dnn_input_object_classification(inp_q, out_q):
    # load the DNN Model
    model = load_DNN_model('mobilenet_custom')
    #model = load_DNN_model('InceptionResNet50')
    try:
        while True:
            try:
                frame_batch = inp_q.get(timeout=0.1)
                #print('ok')
                #out_q.put(frame_batch)
                batch_process_time, pred = batch_of_images(frame_batch, model)
                for processed_frame in pred:
                    out_q.put(processed_frame)

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

def get_metrics(rc, A, model):
    data = []
    if rc == 0:
        throughput_time = (A[0][3]-datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
        trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(A, datetime.datetime.now())
        dnn_batch_latency = batch_of_images(A, model)
        avg_cpu = get_avg_edge_cpu_usage_batch(A)
        avg_mem = get_avg_mem_usage_batch(A)
        data.extend([rc, trans_lat, batch_plus_trans_lat, dnn_batch_latency, avg_cpu, avg_mem, throughput_time])
    else:
        trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(A, datetime.datetime.now())
        dnn_batch_latency = batch_of_images(A, model)
        avg_cpu = get_avg_edge_cpu_usage_batch(A)
        avg_mem = get_avg_mem_usage_batch(A)
        throughput_time = (datetime.datetime.now()-datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
        data.extend([rc, trans_lat, batch_plus_trans_lat, dnn_batch_latency, avg_cpu, avg_mem, throughput_time])

    with open('batch_data.csv', mode='a') as batch_data:
        batch_writer = csv.writer(batch_data, delimiter=',')
        batch_writer.writerow(data)

def recreate_diff_batch(frames):
    try:
        reconst_batch = []
        keyframe = None
        i = 0
        for frame in frames:
            if i == 0:
                keyframe = frame[0]
                reconst_batch.append(frame)
            else:
                idx, vals = frame[0]
                new_frame = keyframe
                new_frame[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
                frame[0] = new_frame
                reconst_batch.append(frame)
            i = i + 1
        # print('Length********', len(frames), len(diff_batch))
        return reconst_batch
    except Exception as e:
        print('Exception**********************'+str(e))


packs = []

def subscriber(ip="172.17.0.1", port=5551):
    # ZMQ connection
    #url = f"tcp://{ip}:{port}"
    url = "tcp://{}:{}".format(ip,port)
    # url = f"tcp://localhost:{port}"
    #print(f"Going to bind to: {url}")
    print("Going to bind to: {url}", url)
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.bind(url)  # subscriber creates ZeroMQ socket
    socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))  # any topic

    #socket = ctx.socket(zmq.PAIR)
    #socket.bind(url)  # connects to pub server
    #socket.connect(url)  # connects to pub server

    print("Sub bound to: {}\nWaiting for data...".format(url))

    #model = load_DNN_model('InceptionResNet50')

    rc = 1
    
    dnn_input_queue = Queue()
    sliding_window_input_queue = Queue()
    sliding_window_output_queue = Queue()
    dnn_input_queue_process = Process(name='DNN',target=dnn_input_object_classification, args=(dnn_input_queue,sliding_window_input_queue,))
    sliding_process = Process(name='Slider',target=sliding, args=(sliding_window_input_queue, sliding_window_output_queue,5,2,))
    dnn_input_queue_process.start()
    sliding_process.start()

    # block_process_input_queue = Queue()
    mathcer_process = Process(name='Blocker', target=cepmatcher, args=(sliding_window_output_queue,))
    mathcer_process.start()


    while True:
        msg = socket.recv()
        A = cPickle.loads(zlib.decompress(msg))
        #A = pk.loads(msg)

        if len(A) !=0:
            #print(A)
            # recreate the diff batch to original
            #A = recreate_diff_batch(A)
            dnn_input_queue.put(A)
            #print('Queuesize********', dnn_input_queue.qsize())
            print('Recieve Count:', rc)
            rc += 1

        # if len(A) !=0:
        #     get_metrics(rc, A, model)
        #
        # for i in A:
        #     if i[2] == 1:
        #         print("i frame received")
        #     sliding_window_input_queue.put(i)

        #print(f'Receive count = {rc}')
        # print('Recieve Count:',rc)
        # rc += 1

        # if rc%1 == 0:
        #     print('Average_Latency******', sum(latency) / len(latency) )
        #     latency.clear()


def start_bw_stats_server():
    import iperf3

    server = iperf3.Server()
    server.port = 6969
    server.verbose = False
    server.json_output = False
    while True:
        server.run()


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

    # start_bw_stats_server()
    # start bandwidth server
    #bw_thread = threading.Thread(target=start_bw_stats_server)
    #bw_thread.start()
    #print("Bandwidth stats server started...")
    # call function and pass on command line arguments
    subscriber(**vars(args))


