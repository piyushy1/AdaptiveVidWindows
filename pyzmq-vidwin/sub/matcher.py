import queue
from videoquery import parse_query

#get the query predicates...
querypredicates = parse_query()


# the above functions have been written on basis of faster -rcnn output
# resnet out will require some datastrucutre changes # use custom finetune resnet on voc
def objectlevel_matching(win_data):
    print('ok')


def conj_query(win_data):
    print('ok')

def cepmatcher(inp_q):
    try:
        while True:
            try:
                frame = inp_q.get(timeout=0.1)
                # print(frame, len(frame))
                if type(frame) == str and frame == 'END':
                    # out_q.put('END')
                    break
                #print(f'New block len- {len(frame)} and time start = {frame[0][1]} and end is = {frame[-1][1]}')
                #print('MATCHER FRAME*****************', frame)
                if querypredicates['operator'] == 'CONJ':
                    conj_query(frame)

                if querypredicates['operator'] == 'object':
                    objectlevel_matching(frame)
                del frame
                # if len(slide_window) == time_segment:
                #     out_q.put(slide_window)
                #     slide_window = slide_window[slide_time:]
                # else:
                #     slide_window.append(frame)
            except queue.Empty:
                pass
    except Exception as e:
        print(e)