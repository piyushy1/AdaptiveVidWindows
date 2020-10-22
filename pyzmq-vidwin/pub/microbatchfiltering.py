import os
import json
import queue

# current container CPU usage
# RUN docker stats writer running to keep these functions available
def calculate_container_CPU_Percent():
    # fetch running container id
    container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
    try:
        f = open("/docker_stats/" + container_id.rstrip() + '.json', "r")
        data = json.loads(f.read())
        cpuDelta = data['cpu_stats']['cpu_usage']['total_usage'] - data['precpu_stats']['cpu_usage']['total_usage']
        systemDelta = data['cpu_stats']['system_cpu_usage'] - data['precpu_stats']['system_cpu_usage']
        if systemDelta > 0.0 and cpuDelta > 0.0:
            cpuPercent = (cpuDelta / systemDelta) * float(len(data['cpu_stats']['cpu_usage']['percpu_usage'])) * 100.0
            return cpuPercent
            #print('CPU Usage ==> ' + str(cpuPercent) + ' %')
    except Exception as e:
        # print(str(e))
        print('File not created yet. Retrying...')


# current container memory usage
def calculate_container_memory():
    # fetch running container id
    container_id = os.popen('head -1 /proc/self/cgroup|cut -d/ -f3').read()
    try:
        mem_percent = []
        f = open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", "r")
        with open("/mem/" + container_id.rstrip() + "/memory.usage_in_bytes", 'r') as infile:
            mem_usage =float(infile.read()) / (1024 * 1024) # mem in MB
            mem_percent.append(mem_usage)
            #print('MEM USAGE', mem_usage)

        with open("/mem/" + container_id.rstrip() + "/memory.limit_in_bytes", 'r') as infile:
            mem_limit = float(infile.read())/(1024 * 1024) # max memory limit of container
            mem_percent.append(mem_limit)
            #print('MEM LIMIT', mem_limit)

        mem_usage_percent = (mem_percent[0]/mem_percent[1])*100
        return mem_usage_percent
        #print('MEM PERCENT %', mem_usage_percent)

    except Exception as e:
        print('container id not found.....'+str(e))


# lazy filter
def lazyfilter(inp_q, out_q, query_predicates):

    while True:
        try:
            new_micro_batch = inp_q.get(timeout= None)

            out_q.put(new_micro_batch)

        except queue.Empty:
            pass