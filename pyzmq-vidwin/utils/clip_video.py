from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import random

def clip_video(start, end):
    ffmpeg_extract_subclip("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/videoplayback.mp4",
                       start, end, targetname = "/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/videoplaybackclip.mp4")


clip_video(1,10)


# random_time = random.sample(range(10, 20000), 13)
#
# for time in random_time:
#     clip_video(time, time+10)

# import cv2
#
# cap = cv2.VideoCapture('/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test3.mp4')
# i =0
# while True:
#     ret, frame = cap.read()
#     if i ==1:
#         break
#     frame = cv2.resize(frame, (900, 900))
#     cv2.imwrite('resol900.jpg', frame)
#     i = i+1





