# Piyush Yadav

import argparse
import zmq
import os
import numpy
os.system('hostname -I')

def subscriber(ip="0.0.0.0", port=5551):
    # ZMQ connection
    url = "tcp://{}:{}".format(ip, port)
    print("Going to bind to: {}".format(url))
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.bind(url)  # subscriber creates ZeroMQ socket
    socket.setsockopt(zmq.SUBSCRIBE, ''.encode('ascii'))  # any topic
    print("Sub bound to: {}\nWaiting for data...".format(url))


    # md = socket.recv_json()
    rc = 1
    while True:
        md = socket.recv_json()
        msg = socket.recv(copy=True)
        buf = memoryview(msg)
        A = numpy.frombuffer(buf, dtype=md['dtype'])
        # print(A.reshape(md['shape']))

        print(f'Receive count = {rc}, md = {md}')
        rc += 1

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
