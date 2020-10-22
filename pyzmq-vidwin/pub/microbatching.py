import random
import queue
import cv2


#function to get distance between histograms of two frames.
def get_frame_distance(frame1, frame2):
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_frame1 = cv2.calcHist([hsv_frame1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_frame1, hist_frame1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_frame2 = cv2.calcHist([hsv_frame2], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_frame2, hist_frame2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # only one compare memthod
    dist_value = cv2.compareHist(hist_frame1, hist_frame2, 0)
    return dist_value
    # print('Method:', 0, ' Distance:',\
    #       base_base, '/')

    # for compare_method in range(4):
    #     base_base = cv2.compareHist(hist_frame1, hist_frame2, compare_method)
    #     print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',\
    #       base_base, '/')


# function to create micro batches
# if its I-Frame or distance between frame less than 0.99 or batch size > XXX
# or slide time end
MAX_BATCH_SIZE = 90
INTERFRAME_SIMILARITY_SCORE = 0.99
def batcher(inp_q, out_q, query_predicates):
    range_counter = 0
    slide_time =[]
    frames = []
    micro_batch_counter = 1
    prev_max_batch =1
    while True:
        try:
            #new_frame = inp_q.get(timeout = 0.1)
            new_frame = inp_q.get(timeout=None)

            if len(slide_time) == 0:
                slide_time.append(new_frame[3])
            # range time condition
            if range_counter ==0:
                if new_frame[3]-slide_time[0] >= query_predicates['RANGE']*1000:
                    #print('RANGE MICROBATCHING TIME~~~~~~~~~~~~~~~~~~~~~~~~~~',new_frame[3]-slide_time[0] )
                    out_q.put(frames)
                    micro_batch_counter+=1
                    print('RANGE MICRO-BATCH SIZE********************', len(frames))
                    frames = []
                    slide_time =[]
                    range_counter+=1

            # check slide time condition
            if range_counter != 0 and len(frames)>1:
                if new_frame[3] - slide_time[0] >= query_predicates['SLIDE'] * 1000:
                    #print('SLIDE MICROBATCHING TIME??????????????????????????? ', new_frame[3] - slide_time[0])
                    out_q.put(frames)
                    micro_batch_counter += 1
                    print('SLIDE MICRO-BATCH SIZE********************', len(frames))
                    frames = []
                    slide_time = []

            # other microbatch conditions...
            if len(frames)>1:
                if len(frames)==MAX_BATCH_SIZE:
                    ##########################################
                    # perform early filtering and drop the batch
                    ############################################
                    if (micro_batch_counter-prev_max_batch)==1:
                        print('Drop Frames*****************************')
                        frames = []
                        prev_max_batch = micro_batch_counter
                        micro_batch_counter += 1
                    else:
                        out_q.put(frames)
                        print('MAXXXXXXXXXXXXXXXXXXXXXXXXXXMICRO-BATCH SIZE********************', len(frames))
                        frames = []
                        prev_max_batch = micro_batch_counter
                        micro_batch_counter += 1


                else:
                    if new_frame[2] == 1 or get_frame_distance(frames[0][0],new_frame[0]) < INTERFRAME_SIMILARITY_SCORE:
                        out_q.put(frames)
                        micro_batch_counter += 1
                        #out_q.put(diff_batch)
                        print('MICRO-BATCH SIZE********************', len(frames))
                        frames = []
            frames.append(new_frame)
            #print('frame lengt************', len(frames))
        except queue.Empty:
            pass

# for EVALUATION
def fixed_batcher(inp_q, out_q,query_predicates):
    frames = []
    while True:
        try:
            new_frame = inp_q.get(timeout = 0.1)
            #print(new_frame)
            if len(frames)==40: #or new_frame[2] == 1:
                # put random batches
                idx = random.randint(5,15)
                #out_q.put(frames[:int(idx/2)] + frames[int(3*idx/2):])
                #diff_batch = create_diff_batch(frames) # send only diff of batch values
                #print('MEM*********************************************',asizeof(np.array(frames, dtype= object))/(1024*1024), asizeof(np.array(diff_batch, dtype= object))/(1024*1024),asizeof(zlib.compress(cPickle.dumps(np.array(frames, dtype= object))))/(1024*1024),asizeof(zlib.compress(cPickle.dumps(np.array(diff_batch, dtype= object))))/(1024*1024))               #print('MEMORY**************************************',asizeof(pk.dumps(frames))/(1024*1024), asizeof(pk.dumps(diff_batch))/(1024*1024))
                out_q.put(frames)
                #out_q.put(diff_batch)
                print('put')
                frames = []
            frames.append(new_frame)
            #print('frame lengt************', len(frames))
        except queue.Empty:
            pass
