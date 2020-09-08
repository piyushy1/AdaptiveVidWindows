import time
import queue
from multiprocessing import Queue

# def get_len(slide_window):


fps = 15

def sliding(inp_q, out_q, time_segment, slide_time):

	def get_slide_time_window(slide_window):
		begin = slide_window[0][1]  #  begin time index
		for frame_idx in range(1,len(slide_window)):
			if (slide_window[frame_idx][1] - begin) >= slide_time*fps:
				return frame_idx


	slide_window = []
	print("Starting Window process")
	curr_time = 0
	while True:
		try:
			frame = inp_q.get(timeout=0.1)
			if type(frame) == str and frame == 'END':
				out_q.put('END')
				break
			if slide_window == []:
				curr_time = frame[1]
			# print(f'In Slider - {frame.shape}')
			if (frame[1] - curr_time)  > time_segment*fps:
				slide_time_end_idx = get_slide_time_window(slide_window)
				out_q.put(slide_window)
				slide_window = slide_window[slide_time_end_idx:]
				curr_time = slide_window[0][1]
			else:
				slide_window.append(frame)
		except queue.Empty:
			pass
