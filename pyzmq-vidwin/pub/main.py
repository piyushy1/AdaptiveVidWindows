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
import docker
# os.system('hostname -I')

def get_docker_stats():
    client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
    for i in client.containers.list():
        print(i.stats(stream=False))

def get_cpu():
    cpu = ''
    for x in range(3):
        cpu = cpu + str(psutil.cpu_percent(interval=1))+' '
    return cpu

def get_used_mem():
    return psutil.virtual_memory()

def get_used_mem_percentage():
    return psutil.virtual_memory().percent

def get_available_memory():
    return psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

def calculateCPUPercent(previousCPU, previousSystem, container_id):
    cpuPercent = 0
    f_container = open("/cpu/" + container_id.rstrip() + "/cpuacct.usage", "r")
    cont_stat = f_container.read().split()

    f_system = open("/system_cpu/" + "/cpuacct.usage", "r")
    system_stat = f_system.read().split()
    cpuDelta = int(cont_stat[0])-int(previousCPU[0])
    systemDelta = int(system_stat[0])- int(previousSystem[0])
    if systemDelta > 0.0 and cpuDelta > 0.0 :
        cpuPercent = (cpuDelta / systemDelta) * 2 * 100.0
        print('*******CPU_PERCENT_USAGE************', cpuPercent)




def socket_send(frame, socket):
    # md = dict(
    #         dtype = str(frame.dtype)
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

    #video_path = '/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test2.mp4'
    video_path = '/app/video/test2.mp4'
    iframes_list = get_i_frames(video_path)
    print('The iframe list***************', iframes_list)

    container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
    print(container_id.rstrip())

    f_container = open("/cpu/" + container_id.rstrip() + "/cpuacct.usage", "r")
    cont_stat = f_container.read().split()
    print('Container CPU Stat: ', cont_stat)

    f_system = open("/system_cpu/" + "/cpuacct.usage", "r")
    system_stat = f_system.read().split()
    print('System CPU Stat: ', system_stat)

    for frame in stream(video_path):
        if ctr in iframes_list:
            batch_input_queue.put([frame,ctr,1])  # an iframe
        else:
            batch_input_queue.put([frame,ctr,0])

        #print('CPU, Mem Use, Limit, MEM%: ',get_cpu(), get_used_mem(), get_available_memory(), get_used_mem_percentage())

        # fetch running container id
        # container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
        # print(container_id.rstrip())
        #
        # f_container = open("/cpu/" + container_id.rstrip() + "/cpuacct.usage", "r")
        # cont_stat = f_container.read().split()
        # print('Container CPU Stat: ', cont_stat)
        #
        # f_system = open("/system_cpu/" + "/cpuacct.usage", "r")
        # system_stat = f_system.read().split()
        # print('System CPU Stat: ', system_stat)
        calculateCPUPercent(cont_stat, system_stat, container_id)
        #print('PSUTIL CPU USAGE****', get_cpu())

        #user_time = int(stat[1])
        #sys_time = int(stat[3])

        f = open("/host_proc/uptime", "r")
        host_uptime = f.read()
        host_uptime = float(host_uptime.split()[0])
        print('Host uptime*** ', host_uptime)

        #container = os.popen('uptime').read()
        #print(container)

        #pid = open("/device/73c7549c9bbd8743b527471031c87e1e11d2bea4d73c059cfe49ec97abe91b99/cgroup.procs", "r")
        #print(pid.read())

        #f = open("/mem/" + container_id.rstrip() + "/cgroup.procs", "r")
        #print(f.read())
        #container_uptime = int(procs_id.split()[0])
        #print(container_uptime)
        #f = open("/host_proc/28808/stat", "r")
        #container_uptime = f.read()
        #container_uptime = int(container_uptime.split()[21])
        #print(container_uptime)
        #print(container_uptime)

        #cpu_usage = 100 * (((user_time + sys_time)/100) / (host_uptime - container_uptime))
        #print("CPU USAGE: " + str(cpu_usage) + " %")

        f = open("/mem/"+ container_id.rstrip()+"/memory.usage_in_bytes", "r")
        print("MEM USAGE: " + str(int(f.read())/(1024*1024))+" MB")
        f = open("/mem/" + container_id.rstrip() + "/memory.limit_in_bytes", "r")
        print("MEM LIMIT: " + str(int(f.read())/(1024*1024))+" MB")
        # f = open("/cpu/" + container_id.rstrip() + "/cpuacct.usage_percpu", "r")
        # stat = f.read().split()
        # print('CPU Statics****: ', stat)
        # print('Docker Stats: ', get_docker_stats())
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
