import numpy as np
import threading
import cv2
import ffmpeg
from ndn.app import NDNApp
from ndn.encoding import Name, InterestParam, BinaryStr, FormalName, MetaInfo, Component
import logging
import time
from collections import deque
import argparse
from typing import Optional
import sys


try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

device_type = "cpu"  # "gpu" or "cpu"
height = 720
width = 1280
logging.basicConfig(level=logging.DEBUG)

try:
    import thread
except ImportError:
    import _thread as thread

# dth = threading.Thread(target=NDNstreaming1.run, args=('testecho', "examples_media_video.avi",width,height,"cpu"))
# dth.start()

producer_frame = np.zeros([height, width, 3])
display_image = producer_frame
app = NDNApp()

buffer_time = 100
current_I_frame = 1
I_frame_index = 0
frame_buffer_I = deque(maxlen=buffer_time)
frame_buffer_P = deque(maxlen=buffer_time * 30)

frame_buffer = deque(maxlen=buffer_time)
frame_buffer_dict = {}
interest_buffer = deque(maxlen=buffer_time)

file_l = []
pkt_length = 1024
for i in range(1051, 1061):
    with open("../models/samples/cl10/sample" + str(i) + ".drc", "rb") as file:
        file_data = file.read()
        file_l.append(file_data)

with open("../simple_sample/sample100.drc", "rb") as file:
    file_data = file.read()

@app.route('/edge')
def on_interest(name: FormalName, param: InterestParam, _app_param: Optional[BinaryStr]):
    logging.info(f'>> I: {Name.to_str(name)}, {param}')
    request = Name.to_str(name).split("/")
    print("handle Interest Name", Name.to_str(name))
    if request[3] == "metadata":
        print("handle Meta data")
        # content = json.dumps(list(pred_frame_buffer)).encode()
        # content = str(current_I_frame).encode()
        content = Name.to_str(name + [Component.from_number(current_I_frame, 0)]).encode()
        name = name
        app.put_data(name, content=content, freshness_period=300)
        logging.info("handle to name " + Name.to_str(name))
    elif request[3] == "frame":
        interest_frame_num = int(request[-1])
        app.put_data(name + [b'\x08\x02\x00\x00'], content=file_data, freshness_period=2000,
                             final_block_id=Component.from_segment(0))
        # if interest_frame_num in frame_buffer_dict:
        #     content = frame_buffer_dict[interest_frame_num]
        #     app.put_data(name + [b'\x08\x02\x00\x00'], content=content, freshness_period=2000, final_block_id=Component.from_segment(0))
        #     print(f'handle interest: publish pending interest' + Name.to_str(name) + "------------/" + str(interest_frame_num) + "length: ", len(content))
        # else:
        #     interest_buffer.append([interest_frame_num, name])
    else:
        print("handle Request missing ", Name.to_str(name))

    while len(interest_buffer) > 0 and len(frame_buffer) > 0 and frame_buffer[-1] >= interest_buffer[0][0]:
        pendingInterest = interest_buffer.popleft()
        pendingFN = pendingInterest[0]
        pendingName = pendingInterest[1]
        if pendingFN in frame_buffer_dict:
            content = frame_buffer_dict[pendingFN]
            app.put_data(pendingName + [b'\x08\x02\x00\x00'], content=content, freshness_period=2000, final_block_id=Component.from_segment(0))
            print(f'handle interest: publish pending interest' + Name.to_str(pendingName) + "------------/" + str(pendingFN) + "length: ", len(content))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDN AR demo")
    parser.add_argument('--task', type=str, default='raw', help='AR Task Type')  # Task
    args = parser.parse_args()
    # eth = threading.Thread(target=video_encoder)
    # eth.start()
    # fth = threading.Thread(target=get_frames)
    # fth.start()
    app.run_forever()
