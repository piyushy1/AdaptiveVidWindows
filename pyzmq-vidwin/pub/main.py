# Name Piyush Yadav

import argparse
import zmq
import time
import os
os.system('hostname -I')
import psutil

def get_cpu():
    return psutil.cpu_percent()

def get_vram():
    return psutil.virtual_memory().percent

def publisher(ip="0.0.0.0", port=5551):
    # ZMQ connection
    url = "tcp://{}:{}".format(ip, port)
    print("Going to connect to: {}".format(url))
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(url)  # publisher connects to subscriber
    print("Pub connected to: {}\nSending data...".format(url))

    i = 0

    while True:
        topic = 'foo'.encode('ascii')
        msg = f'test {i}'.encode('ascii')
        
        usage_topic = 'usage'.encode('ascii')

        # publish data
        socket.send_multipart([topic, msg])  # 'test'.format(i)
        socket.send_multipart([usage_topic, f'CPU - {get_cpu()}'.encode('ascii')])  # 'test'.format(i)
        socket.send_multipart([usage_topic, f'VRAM - {get_vram()}'.encode('ascii')])  # 'test'.format(i)
        print(f"On topic {topic}, send data: {msg}")
        # print(f"On topic {usage_topic}, send data: {}")
        # print(f"On topic {usage_topic}, send data: {f'{get_vram()}'.encode('ascii')}")
        time.sleep(.5)

        i += 1


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
