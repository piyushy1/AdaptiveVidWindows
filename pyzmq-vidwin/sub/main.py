# Piyush Yadav

import os
from pathlib import Path
from collections import namedtuple
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION']='false'
import zmq
import time
from sub.cloudseg import run_carn, load_model
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
from faster_rcnn import load_faster_rcnn_model
from faster_rcnn import batch_of_images_fr
from matcher import cepmatcher
import csv
import numpy as np
import _pickle as cPickle
import zlib
import psutil
# os.system('hostname -I')


def dnn_input_object_detection(inp_q, out_q):
    # load the DNN Model
    model = load_faster_rcnn_model()
    rc =1
    # model = None
    #model = load_DNN_model('InceptionResNet50')
    try:
        while True:
            try:
                acc =[]
                frame_batch = inp_q.get(timeout=0.1)
                # log evalauation metrics
                trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(frame_batch, get_time_milliseconds())
                batch_process_time, pred = batch_of_images_fr(frame_batch, model)
                # put frame one by one in queue
                for processed_frame in pred:
                    #print('Data', processed_frame[0][1])
                    indices = [i for i, x in enumerate(processed_frame[0][1]) if x == "car"]
                    if len(indices) > 0:
                        a = [processed_frame[0][2][index] for index in indices]
                        acc.append(sum(a)/len(a))
                    out_q.put(processed_frame)
                # log evalaution metrics
                if len(acc)>0:
                    get_metrics(len(frame_batch), rc, frame_batch, trans_lat, batch_plus_trans_lat, batch_process_time,sum(acc)/len(acc))
                else:
                    get_metrics(len(frame_batch), rc, frame_batch, trans_lat, batch_plus_trans_lat, batch_process_time,0)
                rc+=1

            except queue.Empty:
                pass
    except Exception as e:
        print(e)

def dnn_input_object_classification(inp_q, out_q):
    # load the DNN Model
    model = load_DNN_model('ResNet101')
    #model = load_DNN_model('InceptionResNet50')
    rc =1
    try:
        while True:
            try:
                acc = []
                frame_batch = inp_q.get(timeout=0.1)
                # log evalauation metrics
                trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(frame_batch, get_time_milliseconds())
                batch_process_time, pred = batch_of_images(frame_batch, model)
                # log evalaution metrics
                #get_metrics(len(frame_batch),rc, frame_batch,trans_lat,batch_plus_trans_lat,batch_process_time)
                for processed_frame in pred:
                    indices = [i for i, x in enumerate(processed_frame[0][1]) if x == "car"]
                    if len(indices) > 0:
                        a = [processed_frame[0][2][index] for index in indices]
                        acc.append(sum(a) / len(a))
                    out_q.put(processed_frame)
                    # log evalaution metrics
                if len(acc) > 0:
                    get_metrics(len(frame_batch), rc, frame_batch, trans_lat, batch_plus_trans_lat,
                                batch_process_time, sum(acc) / len(acc))
                else:
                    get_metrics(len(frame_batch), rc, frame_batch, trans_lat, batch_plus_trans_lat,
                                batch_process_time, 0)
                rc+=1

            except queue.Empty:
                pass
    except Exception as e:
        print(e)


def measure_batch_transmission_latency(batch,time):
    # only transmission latency
    trans_latency = time-batch[-1][3]
    batch_plus_transmission_latency = time-batch[0][3]
    return trans_latency, batch_plus_transmission_latency


def get_avg_edge_cpu_usage_batch(batch):
    cpu_usage = []
    for frame in batch:
        if isinstance(frame[4], float):
            #print('CPU instance',frame[4])
            cpu_usage.append(frame[4])
    if sum(cpu_usage) >0:
        avgcpu = sum(cpu_usage)/ len(cpu_usage)
        return avgcpu
    else:
        return 0

def get_avg_edge_network_usage_batch(batch):
    ntw_usage = []
    for frame in batch:
        ntw_usage.append(frame[6])

    avgntw = sum(ntw_usage)/ len(ntw_usage)
    return ntw_usage


def get_avg_mem_usage_batch(batch):
    mem_usage = []
    for frame in batch:
        if isinstance(frame[5], float):
            mem_usage.append(frame[5])

    if sum(mem_usage) >0:
        avgmem = sum(mem_usage) / len(mem_usage)
        return avgmem
    else:
        return 0


def get_time_milliseconds():
    time = (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time

def get_metrics(batchsize, rc, A,trans_lat,batch_plus_trans_lat,dnn_batch_latency,accuracy_batch):
    data = []
    if rc == 0:
        throughput_time = A[0][3]
        # trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(A, get_time_milliseconds())
        #dnn_batch_latency, pred = batch_of_images(A, model) #object classificato
        # dnn_batch_latency, pred = batch_of_images_fr(A,model) # object detection
        avg_ntw= get_avg_edge_network_usage_batch(A)
        avg_cpu = get_avg_edge_cpu_usage_batch(A)
        avg_mem = get_avg_mem_usage_batch(A)
        data.extend([batchsize, rc, trans_lat, batch_plus_trans_lat, dnn_batch_latency, avg_cpu, avg_mem,throughput_time,A[0][0].shape,accuracy_batch])
    else:
        # trans_lat, batch_plus_trans_lat = measure_batch_transmission_latency(A, get_time_milliseconds())
        #dnn_batch_latency, pred = batch_of_images(A, model) #object classificato
        # dnn_batch_latency, pred = batch_of_images_fr(A, model)  # object detection
        #avg_ntw = get_avg_edge_network_usage_batch(A)
        avg_cpu = get_avg_edge_cpu_usage_batch(A)
        avg_mem = get_avg_mem_usage_batch(A)
        throughput_time = get_time_milliseconds()
        data.extend([batchsize, rc, trans_lat, batch_plus_trans_lat, dnn_batch_latency, avg_cpu, avg_mem,throughput_time,A[0][0].shape,accuracy_batch])

    with open('cloudseg_batch1_thlatency_jacksonhole3min_clip_resnet101_new.csv', mode='a') as batch_data:
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


def cloud_seg_sr(frame_batch, scale, net, device):
    # super resolution
    SR_dir = f'/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/sub/output/frames_x{scale}'
    Path(SR_dir).mkdir(parents=True, exist_ok=True)
    carn_args = {
        'model': 'carn',
        'ckpt_path': './checkpoint/carn.pth',
        'SR_dir': SR_dir,
        'frame_batch': frame_batch,
        'scale': scale,
        'group': 1,
        'shave': 20,
        'cuda': True,
        'batch_size': len(frame_batch),
        'with_bar': False,
        'desc': 'test',
    }
    carn_args = namedtuple('args', ' '.join(list(carn_args.keys())))(**carn_args)
    return run_carn(carn_args, net, device)

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

    # load model for evalauation purpose ####COMMENT IT
    # model = load_faster_rcnn_model()
    #model = load_DNN_model('mobilenet_custom')
    # for cloud seg model
    net, device = load_model('./checkpoint/carn.pth')
    while True:
        msg = socket.recv()
        A =cPickle.loads(msg)
        #A = cPickle.loads(zlib.decompress(msg))
        #A = pk.loads(msg)

        if len(A) !=0:
            #print(A)
            # recreate the diff batch to original
            #A = recreate_diff_batch(A)
            # cloudseg model for supere resolution***************
            frame_batch = [A[0][0]]
            frame_batch = cloud_seg_sr(frame_batch, scale=4, net=net, device=device)
            A[0][0] = np.transpose(frame_batch, (2,1, 0))
            #****************************************************8
            dnn_input_queue.put(A)
            print('Queuesize********', dnn_input_queue.qsize())
            print('Recieve Count:', rc)
            rc += 1

        # if len(A) !=0:
        #     # a = A[-1]
        #     # get_metrics(a, rc, A[:-1], model) # first element is batch size
        #     get_metrics(200, rc, A, model)  # first element is batch size
        #     print('Recieve Count:', rc)
        #     rc += 1

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


