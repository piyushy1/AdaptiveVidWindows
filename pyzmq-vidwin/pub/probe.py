import subprocess

# function to get i frames of the video
def get_i_frames(video_path):
	ffprobe = subprocess.Popen(["ffprobe", "-select_streams", "v", "-show_frames", "-show_entries", "frame=pict_type", "-of", "csv", f"{video_path}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	grep = subprocess.Popen(["grep", "-n", "I"], stdin=ffprobe.stdout, stdout=subprocess.PIPE)
	out, err = grep.communicate()
	if err is None:
		iframes = []
		for i in out.decode('utf-8').split('\n'):
			if i != "":
				iframes.append(int(i.split(":")[0]))
		return iframes
	raise ValueError("Error in ffprobe command")
