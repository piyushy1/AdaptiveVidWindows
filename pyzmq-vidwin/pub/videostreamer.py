import cv2
import time

def stream(video_path):
    cap = cv2.VideoCapture(video_path)
    # get frame dimension
    width = cap.get(3)  # float
    height = cap.get(4)  # float

    print('width, height:', width, height)
    # read the fps so that opencv read with same fps speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for video: {video_path} : {fps}")
    # time to sleep the process
    #sleep_time = 1/(fps) # 2 is added to make the reading frame time and video time equivalnet. this is an empirical value may change for others.
    sleep_time = 1 / (30)
    print(f"Sleep Time : { sleep_time}")
    # to check the frame count
    frame_count_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('FRAME COUNT***', frame_count_num)
    #duration = frame_count_num/fps
    duration = frame_count_num / 30
    print(f"Duration of video {duration%60}")

    print("Video Publisher initiated== {video_path}")

    # attach the frame probe:
    # to be correct this is not a real streaming scenario. this is just a way around. need to create a pipe function from
    #ffmpeg to read frame by frame and attach each frame info

    # frame_types = get_frame_types(self.publisher_path,self.publisher_id)
    # i_frames = [x[0] for x in frame_types if x[1]=='I']
    # print(i_frames)
    frame_count =0
    dt1 = time.time()

    #add the windowing concept here...
    # get the time from query : presently only one query

    # window_time = read_query()
    # window = WindowAssigner(self.publisher_id,window_time)
    # window.assign_window()

    # process video
    while True:
        # Capture frame-by-frame

        # frame_info_list = []
        ret, frame = cap.read()
        if not ret:
            break
        # md = dict(
        #     dtype = str(frame.dtype),
        #     shape = frame.shape,
        # )
        # socket.send_json(md)
        # socket.send(frame)
        # frame resizing the frame...
        #frame = cv2.resize(frame, (224,224))frame[300:, 145:]
        #frame = frame[270:, 320:] # for auburn toomers corner
        #frame = frame[300:, 145:] # for sandylane post processing
        #frame = cv2.resize(frame, (480, 270),interpolation=cv2.INTER_LINEAR) # cloudseg 4x
        #frame = cv2.resize(frame, (240, 135), interpolation=cv2.INTER_LINEAR) #cloudseg 8x
        yield frame
        #print(f'Frame count = {frame_count}')

        # need to set the frame rate:
        # if frame_count in i_frames:
        #     frame_info_list.append("I")
        #     frame_info_list.append(frame)
        #     self.publisher_queue.put(frame_info_list)
        #     print("i-frame : "+ str(frame_count))

        # else:
        #     frame_info_list.append(frame)
        #     self.publisher_queue.put(frame_info_list)

        time.sleep(sleep_time)
        frame_count += 1
        # if frame_count >=50:
        #     time.sleep(4)

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        #cv2.imshow(str(self.publisher_id), gray)

        #print queue size
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # print("PublisherID "+ str(self.publisher_id) +"  Queue SIZE...."+ str(Config.publisher_queue_map.get(self.publisher_id).qsize()))
    dt2 = time.time()
    print(f"Time take to read video: {(dt2-dt1)}")



    # When everything done, release the capture
    cap.release()