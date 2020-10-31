import queue
from videoquery import parse_query

#get the query predicates...
querypredicates = parse_query()


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
                # print('New WINDOW length:', len(frame))
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