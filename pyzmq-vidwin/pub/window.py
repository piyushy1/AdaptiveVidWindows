import time
import queue
from multiprocessing import Queue

def sliding(inp_q, out_q, time_segment, slide_time):
	slide_window = []
	while True:
		try:
			frame = inp_q.get(timeout=0.1)
			if type(frame) == str and frame == 'END':
				out_q.put('END')
				break
			print(frame.shape)
			if len(slide_window) == time_segment:
				out_q.put(slide_window)
				slide_window = slide_window[slide_time:]
			else:
				slide_window.append(frame)
		except queue.Empty:
			pass
