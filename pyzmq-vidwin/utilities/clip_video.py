from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import random

def clip_video(start, end):
    ffmpeg_extract_subclip("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/pizza_bratsilava_slovakia_20feb.mp4",
                       start, end, targetname = "/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/Videosonbasisofmotion/pizzashop"+str(start)+".mp4")


random_time = random.sample(range(10, 20000), 13)

for time in random_time:
    clip_video(time, time+10)