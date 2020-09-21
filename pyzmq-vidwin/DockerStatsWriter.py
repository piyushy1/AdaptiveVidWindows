import requests
import json


c=1
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
            #print(json.dumps(data))
            outfile.write(json.dumps(data))

    if len(containers) == 0:
        break
    else:
        print('Updating File: ' + str(c))
        c+=1