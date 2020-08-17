from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN/test1.mp4",
                       1, 5, targetname = "/home/dhaval/piyush/ViIDWIN/Datasets_VIDWIN//test3.mp4")
