import requests
import json
import time

c=1
time.sleep(1)
while True:

    containers = requests.get('http://localhost:2376/containers/json')
    containers = containers.json()
    # print(containers)
    for container in containers:
        stats = requests.get('http://localhost:2376/containers/' + container['Id'] + '/stats?stream=0')
        stats = stats.json()
        with open('/home/dhaval/piyush/ViIDWIN/Code/pyzmq-vidwin/stats/'+container['Id']+".json", "w") as outfile:
            data = dict()
            data['cpu_stats'] = stats['cpu_stats']
            data['precpu_stats'] = stats['precpu_stats']
            data['memory_stats'] = stats['memory_stats']
            # if 'networks' in stats:
            #     data['networks'] = stats['networks']
            #print(stats['networks'])
            #print(json.dumps(data))
            outfile.write(json.dumps(data))
            #print('Updating File: ' + container['Id'] + '.json, ' + 'Try: ' +str(c))
            c += 1

    if len(containers) == 0:
        break
