import os
import json
import queue
from scipy.stats import entropy
from pympler.asizeof import asizeof
import traceback
# partial match cache
PM_CACHE = {}
import subprocess

# current container CPU usage
# RUN docker stats writer running to keep these functions available
def calculate_container_CPU_Percent():
    try:
        container_id = os.popen("cat /etc/hostname").read()
        files = os.listdir('/docker_stats/')
        container_stats_file = ''
        for file in files:
            if file.startswith(container_id.strip()):
                container_stats_file = file
                break

        f = open("/docker_stats/" + container_stats_file, "r")
        data = json.loads(f.read())
        cpuDelta = data['cpu_stats']['cpu_usage']['total_usage'] - data['precpu_stats']['cpu_usage']['total_usage']
        systemDelta = data['cpu_stats']['system_cpu_usage'] - data['precpu_stats']['system_cpu_usage']
        if systemDelta > 0.0 and cpuDelta > 0.0:
            cpuPercent = (cpuDelta / systemDelta) * float(len(data['cpu_stats']['cpu_usage']['percpu_usage'])) * 100.0
            return cpuPercent
            # print('CPU Usage ==> ' + str(cpuPercent) + ' %')
    except Exception as e:
        traceback.print_exc()
        pass

# def calculate_packet_transfered():
#     container_id = os.popen("cat /etc/hostname").read()
#     files = os.listdir('/docker_stats/')
#     container_stats_file = ''
#     for file in files:
#         if file.startswith(container_id.strip()):
#             container_stats_file = file
#             break
#
#     try:
#         f = open("/docker_stats/" + container_stats_file, "r")
#         data = json.loads(f.read())

#         if 'networks' in data:
#             packet_transfered = (data['networks']['eth0']['tx_bytes'])/1024
#             return packet_transfered
#         else:
#             return 0
#         #print('Data  ==> ' + str(cpuPercent) + ' %')
#     except Exception as e:
#         traceback.print_exc()


def calculate_container_memory_Percent():
    # fetch running container id
    #container_id = os.popen("cat").read()
    container_id = os.popen("cat /etc/hostname").read()
    files = os.listdir('/docker_stats/')
    container_stats_file = ''
    for file in files:
        if file.startswith(container_id.strip()):
            container_stats_file = file
            break

    try:
        f = open("/docker_stats/" + container_stats_file, "r")
        data = json.loads(f.read())
        memUsagePercent = ((data['memory_stats']['stats']['cache']+data['memory_stats']['stats']['rss'])/data['memory_stats']['stats']['hierarchical_memory_limit'])*100
        return memUsagePercent
            # print('CPU Usage ==> ' + str(cpuPercent) + ' %')
    except Exception as e:
        traceback.print_exc()

# # current container memory usage
# def calculate_container_memory():
#     # fetch running container id
#     container_id = os.popen("/proc/self/cgroup | grep 'cpu:/' | sed 's/\([0-9]\):cpu:\/docker\///g'").read()
#     try:
#         mem_percent = []
#         f = open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", "r")
#         with open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", 'r') as infile:
#             mem_usage = float(infile.read()) / (1024 * 1024)  # mem in MB
#             mem_percent.append(mem_usage)
#             # print('MEM USAGE', mem_usage)
#
#         with open("/mem/" + container_id.rstrip() + "/memory.limit_in_bytes", 'r') as infile:
#             mem_limit = float(infile.read()) / (1024 * 1024)  # max memory limit of container
#             mem_percent.append(mem_limit)
#             # print('MEM LIMIT', mem_limit)
#
#         mem_usage_percent = (mem_percent[0] / mem_percent[1]) * 100
#         print('MEM PERCENT %', mem_usage_percent)
#         return mem_usage_percent
#         # print('MEM PERCENT %', mem_usage_percent)
#
#     except Exception as e:
#         print('container id not found.....' + str(e))


def initialize_cache(query_predicates):
    for object in query_predicates['object']:
        PM_CACHE[object] = 0
    # print('CACHE????????????????', PM_CACHE)


def drop_entropy_frames(microbatch, entropy):
    utilitymb_remove= asizeof(microbatch)*entropy
    # remove random frames:
    indexes = [i for i in range(1,len(microbatch),2)]
    for index in sorted(indexes, reverse=True):
        if asizeof(microbatch) > utilitymb_remove:
            del microbatch[index]
    else:
        return microbatch


def get_microbatch_utility(new_resized_micro_batch, window_counter, win_type, query_predicates):
    #print('INDOW COUNTER************', window_counter)
    micro_batch_accuracy = []
    frame_processed = 0
    micro_batch_relative_position = 0
    micro_batch_size = 0
    frame_left_window = 0
    cache_bool = False
    # access the identified objects of the microbatch
    micro_batch_objects = new_resized_micro_batch[-1]
    for i in range(1, len(micro_batch_objects) + 1): # 1 because key frame is kept
        for key, value in micro_batch_objects[i - 1].items():
            #### udpate PARTIAL CACHE and if object not present drop it
            ##########################################################
            if key in query_predicates['object']:
                if value > PM_CACHE[key]:
                    PM_CACHE[key] = value
                    cache_bool = True
            # calculate the cache score...
            micro_batch_accuracy.append(value / i)

    # micro batch relative position
    if isinstance(new_resized_micro_batch[-2], dict):
        frame_processed = new_resized_micro_batch[-3][1] - window_counter
        micro_batch_size = len(new_resized_micro_batch) - 2
    else:
        frame_processed = new_resized_micro_batch[-3][1] - window_counter
        micro_batch_size = len(new_resized_micro_batch) - 2


    if win_type == 'RANGE':
        micro_batch_relative_position = 1 - (frame_processed / (query_predicates['RANGE'] * 30))  # 30fps
        frame_left_window = (query_predicates['RANGE'] * 30) - frame_processed
        #print('Remaining data in window RAAANAGE*********', micro_batch_relative_position, frame_left_window)
    if win_type == 'SLIDE':
        micro_batch_relative_position = 1 - (frame_processed / (query_predicates['SLIDE'] * 30))
        frame_left_window = (query_predicates['SLIDE'] * 30) - frame_processed
        #print('Remaining data in window SLIIDEEEEEEE*********', micro_batch_relative_position, frame_left_window)

    remaining_mb_on_win = 1- (micro_batch_size / frame_left_window)
    #print('Remaining data in window*********', micro_batch_relative_position, remaining_mb_on_win)

    entropy_val = entropy([micro_batch_relative_position, remaining_mb_on_win], base=2)
    #entropy_val1 = entropy([sum(micro_batch_accuracy),micro_batch_relative_position, remaining_mb_on_win], base=2)
    entropy_val2 = entropy([sum(micro_batch_accuracy), entropy_val], base=2)

    #print('Micro Batch accuracy, Entropy', sum(micro_batch_accuracy), entropy_val,entropy_val2)

    # return the accuracy and entropy value with cache bool value...
    return sum(micro_batch_accuracy), entropy_val2, cache_bool


# lazy filter
def lazyfilter(inp_q, out_q, query_predicates):
    # initalise partial match cache with
    initialize_cache(query_predicates)
    win_counter = 0
    win_type = 'RANGE'
    i = 0
    while True:
        try:
            new_resized_micro_batch = inp_q.get(timeout=0.01)
            # update the window to get frame number
            if isinstance(new_resized_micro_batch[-2], dict):
                if i == 0:
                    win_counter = new_resized_micro_batch[-2]['WINDOW'] + 1
                    initialize_cache(query_predicates)
                    i=1
                else:
                    win_counter = new_resized_micro_batch[-2]['WINDOW'] + 1
                    win_type = 'SLIDE'
                    #i=1
                    initialize_cache(query_predicates)
            # calculate the microbatch utility
            microbatch_accuracy, entropy, cache_bool = get_microbatch_utility(new_resized_micro_batch, win_counter,
                                                                              win_type, query_predicates)
            #print('CACHE BOOL*************', cache_bool)
            if cache_bool== True:
                # calculate the memory and cpu consumption and filter the batch///
                if calculate_container_memory_Percent() > 20 or calculate_container_CPU_Percent() > 20:
                    # remove the object list and range dictionary
                    del new_resized_micro_batch[-1]  # remove the object list
                    if isinstance(new_resized_micro_batch[-1], dict):
                        del new_resized_micro_batch[-1]  # remove the window list
                    #print('Data Send ENTROPY FILTERING BEFORE', asizeof(new_resized_micro_batch))
                    new_resized_micro_batch = drop_entropy_frames(new_resized_micro_batch,entropy)
                    #print('Data Send ENTROPY FILTERING', asizeof(new_resized_micro_batch))
                    #new_resized_micro_batch = create_diff_batch(new_resized_micro_batch)
                    out_q.put(new_resized_micro_batch)
                else:
                    # remove the object list and range dictionary
                    del new_resized_micro_batch[-1] # remove the object list
                    if isinstance(new_resized_micro_batch[-1], dict):
                        del new_resized_micro_batch[-1]  # remove the window list
                    #print('Data Send WITHOUT ENTROPY FILTERING')
                    #new_resized_micro_batch = create_diff_batch(new_resized_micro_batch)
                    out_q.put(new_resized_micro_batch)
            else:
                del new_resized_micro_batch


        except queue.Empty:
            pass


# for EVALUATION... do fixed filtering or no filtering
def fixed_filter(inp_q, out_q, query_predicates):
    while True:
        try:
            new_micro_batch = inp_q.get(timeout=None)
            # drop the frames if you want....
            # idx = random.randint(5, 15)
            # out_q.put(new_micro_batch[:int(idx/2)] + frames[int(3*idx/2):])
            out_q.put(new_micro_batch)

        except queue.Empty:
            pass