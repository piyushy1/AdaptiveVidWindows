import queue
from videoquery import parse_query
from datetime import datetime
from multiprocessing import Queue

# def get_len(slide_window):

fps = 15


def get_time_milliseconds():
    time = (datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0
    return time


def sliding(inp_q, out_q, time_segment, slide_time):
    query_predicates = parse_query()

    def get_slide_time_window(slide_window,delay):
        begin = slide_window[0][3]  # begin time index
        for frame_idx in range(1, len(slide_window)):
            if isinstance(slide_window[frame_idx][2],str):
                return frame_idx

            if (slide_window[frame_idx][3] - begin) >= query_predicates['SLIDE']*1000:
                return frame_idx

    slide_window = []
    print("Starting Window process")
    curr_time = 0
    curr_edge_window_time = 0
    curr_cloud_window_time = 0
    delay = 0
    i =0
    while True:
        try:
            frame = inp_q.get(timeout=None)
            if type(frame) == str and frame == 'END':
                out_q.put('END')
                break
            if slide_window == []:
                # curr_time = frame[1]
                curr_edge_window_time = frame[3]
                curr_cloud_window_time = get_time_milliseconds()
                delay = curr_cloud_window_time - curr_edge_window_time
            # print(f'In Slider - {frame.shape}')
            if (frame[3] - curr_edge_window_time) > query_predicates['RANGE'] * 1000 or (
                    get_time_milliseconds() - curr_cloud_window_time) >= ((query_predicates['RANGE'] * 1000) + delay):
                slide_time_end_idx = get_slide_time_window(slide_window,delay)
                #print('WINDOW Lenght*****', len(slide_window))
                out_q.put(slide_window)
                slide_window = slide_window[slide_time_end_idx:]
                # curr_time = slide_window[0][3]
                curr_edge_window_time = slide_window[0][3]
                curr_cloud_window_time = get_time_milliseconds()
                delay = curr_cloud_window_time-curr_edge_window_time
                i =0
            else:
                slide_window.append(frame)
                if i ==0 and (get_time_milliseconds()- curr_cloud_window_time)>= query_predicates['SLIDE']*1000+delay:
                    slide_window.append([0,1,'SLIDE',get_time_milliseconds()])
                    i = i+1

        except queue.Empty:
            if (get_time_milliseconds() - curr_cloud_window_time) >= ((query_predicates['RANGE'] * 1000) + delay):
                if len(slide_window)>0:
                    slide_time_end_idx = get_slide_time_window(slide_window, delay)
                    #print('WINDOW Lenght*****', len(slide_window))
                    out_q.put(slide_window)
                    slide_window = slide_window[slide_time_end_idx:]
                    # curr_time = slide_window[0][3]
                    curr_edge_window_time = slide_window[0][3]
                    curr_cloud_window_time = get_time_milliseconds()
                    delay = curr_cloud_window_time - curr_edge_window_time
                    i=0
            else:
                if i ==0 and (get_time_milliseconds()- curr_cloud_window_time)>= query_predicates['SLIDE']*1000+delay:
                    slide_window.append([0,1,'SLIDE', get_time_milliseconds()])
                    i = i+1


            #pass
