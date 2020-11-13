import queue
from videoquery import parse_query
import datetime
import csv

#get the query predicates...
querypredicates = parse_query()

def log_data(data):
    with open('esperwindowtime3sec.csv', mode='a') as batch_data:
        batch_writer = csv.writer(batch_data, delimiter=',')
        batch_writer.writerow(data)


def get_time_milliseconds():
    time = (datetime.datetime.now() - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time

# the above functions have been written on basis of faster -rcnn output
# resnet out will require some datastrucutre changes # use custom finetune resnet on voc
def objectlevel_matching(win_data):
    event_detected = []
    event = {}
    # first selection and consumtion policy
    for data in win_data:
        if 'car' in data[0][1]:
            event['car'] = 0
        if len(event) ==1:
            event_detected.append('Object Event Detected')
            event = {}
    print('Total Event Detected', len(event_detected))


def conj_query(win_data):
    event_detected = []
    event ={}
    # first selection and consumtion policy
    for data in win_data:
        #print('data*******', data[0][1])
        if 'car' in data[0][1]:
            event['car'] = 0
        if 'person' in data[0][1]:
            event['person'] =0
        if len(event) ==2:
            event_detected.append('CONJ Event Detected')
            event = {}
    print('Total Event Detected', len(event_detected))
    event_detected = []
    #print('ok')

def cepmatcher(inp_q):
    try:
        while True:
            try:
                frame = inp_q.get(timeout=0.1)
                data = get_time_milliseconds()
                log_data([data])
                # print(frame, len(frame))
                if type(frame) == str and frame == 'END':
                    # out_q.put('END')
                    break
                #print(f'New block len- {len(frame)} and time start = {frame[0][1]} and end is = {frame[-1][1]}')
                #print('MATCHER FRAME*****************')
                # if querypredicates['operator'] == 'CONJ':
                #     conj_query(frame)
                #
                # if querypredicates['operator'] == 'OBJ':
                #     objectlevel_matching(frame)
                # del frame
                # if len(slide_window) == time_segment:
                #     out_q.put(slide_window)
                #     slide_window = slide_window[slide_time:]
                # else:
                #     slide_window.append(frame)
            except queue.Empty:
                pass
    except Exception as e:
        print(e)