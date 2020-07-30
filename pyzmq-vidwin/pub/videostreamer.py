# Author - Piyush Yadav
# Insight Centre for Data Analytics


# function to read video...
from __future__ import division
import numpy as np
import cv2
import queue

class VideoStreamer:

    #intialise video publisher
    def __init__(self,publisher_path):
        self.publisher_path =  publisher_path


    #load the video publisher
    def load_video(self):

        # read the video file
        cap = cv2.VideoCapture(self.publisher_path)

        # read the fps so that opencv read with same fps speed
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS for video: ",self.publisher_id, " : ", fps )
        # time to sleep the process
        sleep_time = 1/(fps+2) # 2 is added to make the reading frame time and video time equivalnet. this is an empirical value may change for others.
        print("Sleep Time : ", sleep_time )
        # to check the frame count
        frame_count_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('Frame COUNT: ',frame_count_num)
        duration = frame_count_num/fps
        print("Duration of video ", self.publisher_id, " ", duration%60)


        print("Video Publisher initiated==" + str(self.publisher_id))



        #add the windowing concept here...
        # get the time from query : presently only one query

        # window_time = read_query()
        # window = WindowAssigner(self.publisher_id,window_time)
        # window.assign_window()

        # process video
        i= 1
        while(True):
            # Capture frame-by-frame

            frame_info_list = []
            ret, frame = cap.read()
            if not ret:
                break
            print('frame: ', i)
            i = i+1

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()