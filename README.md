# VID-WIN AdaptiveWindows

This is repo page for paper 'VID-WIN: Query Aware Data-Driven Windowing in Complex Event Processing for Fast Video Inference in Edge Cloud Paradigm'

### Requirements
VID-WIN prototype is implemented in python.

**Software Requirements**
You will need the following to be installed before running the system:
1. Python 3
2. Cuda 10
3. Tensorflow 2.0
4. Pytorch
5. Docker

**Hardware Requirements**
1. GPU (tested on Nvidia RTX 2080 Ti with 10 GB RAM)
2. Atleast 64 GB RAM

# Important File Information
[pub/main.py](https://github.com/piyushy1/AdaptiveVidWindows/blob/master/pyzmq-vidwin/pub/main.py)
[pub/microbatching.py](https://github.com/piyushy1/AdaptiveVidWindows/blob/master/pyzmq-vidwin/pub/microbatching.py)

# Steps to run the process
1. There are two folder sub and pub. Pub emulates the edge device.
2. To run pub run the docker-compose-cpu.yml

```
docker-compose -f docker-compose-cpu.yml build
```
The CPU and memory can be set from the dockercompose file.
```
    mem_limit: 2000m
    mem_reservation: 1000m
    cpuset: 0-4
```


### Contact
In case of any queries or issue please connect with me at piyush.yadav@insight-centre.org

