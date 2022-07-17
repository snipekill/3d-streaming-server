# Original code here:
# https://github.com/opencv/opencv/blob/master/samples/python/video_threaded.py

#!/usr/bin/env python3

'''
Multithreaded video processing minimal sample.
Usage:
   python3 video_threaded.py
   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.
Keyboard shortcuts:
   ESC - exit
'''
# from collections import deque
# import multiprocessing
from multiprocessing import Process, Queue
from sys import getrefcount
# from multiprocessing.pool import ThreadPool
from datetime import datetime
from cv2 import VideoCapture,getNumberOfCPUs,CAP_PROP_FRAME_COUNT
from PIL import Image
from psutil import Process as Pro
from gc import get_count
import tracemalloc
import time
# import subprocess
# import sys
# import os

# import argparse
# import sys
from pydraco import encode_point_cloud
import DracoPy
# import os
# from PIL import Image
import numpy as np
# from scipy import misc
# import scipy.ndimage
# import os

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0
# def frame_encode(points, colors, q):
#     quantization_bits = 14
#     compression_level = 1
#     quantization_range = 1
#     quant_origin = 0
#     create_metadata = False
#     # print("cinuf")
#     encoding_data = encode_point_cloud(points,colors, [], 3, 0, quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
#     # print("dsds")
#     q.put(encoding_data.buffer)

def process_frame(VIDEO_SOURCE, VIDEO_DEPTH_SOURCE, frame_associated, pool_size, number_of_frames, queue):
    cap = VideoCapture(VIDEO_SOURCE)
    dep = VideoCapture(VIDEO_DEPTH_SOURCE)
    # print("dssfsdfsdfsdfsd", "kkkkkkkkkkkkkkkkkkk")
    tracemalloc.start()
    # tracker = SummaryTracker()
    index_ = 0
    # some intensive computation...
    # t = 1
    snapshot1 = None
    snapshot2 = None
    while True:
        # t-=1
        index_ += 1
        # if index_%20 == 0:
        #     snapshot2 = tracemalloc.take_snapshot()
        #     top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        #     print("[ Top 20 differences ]")
        #     for stat in top_stats[:20]:
        #         print(stat)
            # snapshot1 = tracemalloc.take_snapshot()
        cap.set(1, frame_associated)
        dep.set(1, frame_associated)
        frame_got_color, frame_color = cap.read()
        frame_got_depth, frame_depth = dep.read()
        if frame_got_color and frame_got_depth:
            start = datetime.now()
            # rgb = Image.fromarray(frame_color)
            # depth = Image.fromarray(frame_depth).convert("I")
            # points = []
            depth_np = np.array(Image.fromarray(frame_depth).convert("I"))
            color_np = np.array(Image.fromarray(frame_color))
            u = depth_np.shape[1]
            v = depth_np.shape[0]
            # rgb_ = []

            # print(rgb.size)
            # print(depth.size)
            # print(np.shape(rgb))
            # print(np.array(depth))
            # print(np.shape(depth))
            # print(rgb.getpixel((122,143)))
            # print(color_np[143][122])
            Z = depth_np * 1.0 / scalingFactor
            X = (np.arange(u) - centerX) * Z / focalLength
            Y = (np.arange(v).reshape(v,1) - centerY) * Z / focalLength
            final = np.dstack((X, Y, Z))
            final_arr = final[~np.all(final == 0, axis=2)]
            color_np = color_np[~np.all(final == 0, axis=2)]
            startfor = datetime.now()
            # for v in range(rgb.size[1]):
            #     for u in range(rgb.size[0]):
            #         color = rgb.getpixel((u, v))
            #         # print("rgb ",type(rgb), " ",rgb.size)
            #         # print("color at count u,v ",count, " ",u, " ",v," ",color)
            #         # Z = depth.getpixel((u,v)) / scalingFactor
            #         Z = depth.getpixel((u, v)) * 1.0 / scalingFactor
            #         if Z == 0: continue
            #         X = (u - centerX) * Z / focalLength
            #         Y = (v - centerY) * Z / focalLength
            #         # points2.append("%f %f %f\n" % (X, Y, Z))
            #         points.append(X)
            #         points.append(Y)
            #         points.append(Z)
            #         rgb_.append(color[0])
            #         rgb_.append(color[1])
            #         rgb_.append(color[2])
            #         # points2.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))

            points_vector = final_arr.reshape(-1).tolist()
            colors_vector = color_np.reshape(-1).tolist()
            quantization_bits = 14
            compression_level = 1
            quantization_range = 1
            quant_origin = 0
            create_metadata = False
            # time.sleep(0.25)
            # queue.put(bytes(data))
            # queue.put(b'0'*2200000)
            encoding_data = encode_point_cloud(final.reshape(-1).tolist(), color_np.reshape(-1).tolist(), [], 3, 0, quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
            # try:
            #     binary = DracoPy.encode(final.reshape(-1).tolist(), color_np.reshape(-1).tolist())
            #     queue.put(binary)
            # except Exception as e:
            #     print(e)
            # encoding_data = DracoPy.encode(final.reshape(-1).tolist(), )
            end = datetime.now()
            # os.nice(-19)
            # os.system("sudo renice -n -19 -p " + str(os.getpid()))
            # print(open("/proc/{pid}/stat".format(pid=os.getpid()), 'rb').read().split()[-14], frame_associated)
            # print("executiontime for count ", frame_associated , " ", (end-start) )
            # queue.put(bytes(encoding_data.buffer))
            # print(len(bytes(str(int(len(encoding_data))), "utf-8")))
            queue.put(bytes(encoding_data.buffer))
            # queue.put(encoding_data)
            # if index_ == 1:
            #     snapshot1 = tracemalloc.take_snapshot()
            # print(getrefcount(encoding_data), " ****************")
            # print(get_count(), '****************')
            # print(queue.qsize(), '****************')
            # with open(
            #     os.path.join(f"./batch_frames/frame{frame_associated}.drc"), "wb"
            # ) as test_file:
            #     test_file.write(bytes(encoding_data.buffer))

        frame_associated = (frame_associated + pool_size) % number_of_frames
    return frame_associated



class FramePipelineJob:
    def __init__(self, VIDEO_SOURCE, VIDEO_DEPTH_SOURCE):
        self.VIDEO_SOURCE = VIDEO_SOURCE
        self.VIDEO_DEPTH_SOURCE = VIDEO_DEPTH_SOURCE
        self.frameIndex = 0
        self.thread_num = getNumberOfCPUs() // 2
        self.process_queues = []
        self.processes = []
        self.number_of_frames = int(VideoCapture(self.VIDEO_SOURCE).get(CAP_PROP_FRAME_COUNT))
        for i in range(self.thread_num):
            queue = Queue()
            self.process_queues.append(queue)

    def getNextFrame(self):
        currentFrame = self.process_queues[self.frameIndex].get()
        self.frameIndex = (self.frameIndex + 1) % self.thread_num

        return currentFrame

    def startProcessing(self):
        for i in range(self.thread_num):
            processEntity = Process(target = process_frame, args = (self.VIDEO_SOURCE,self.VIDEO_DEPTH_SOURCE, i, self.thread_num, self.number_of_frames, self.process_queues[i],))
            processEntity.start()
            self.processes.append(processEntity)

    def stopProcessing(self):
        for i in range(self.thread_num):
            self.processes[i].terminate()

    def getMemoryFootPrint(self):
        total_mem = 0
        queue_size = 0
        for i in range(self.thread_num):
            total_mem += Pro(self.processes[i].pid).memory_info().rss // 1e6
            queue_size += self.process_queues[i].qsize()

        return (total_mem, queue_size)




# def process_frame(VIDEO_SOURCE, VIDEO_DEPTH_SOURCE, frame_associated, pool_size):
#     cap = cv.VideoCapture(VIDEO_SOURCE)
#     dep = cv.VideoCapture(VIDEO_DEPTH_SOURCE)
#     # some intensive computation...
#     master_dir = "./batch_frames/"
#     # cv.imwrite("./batch_frames/frame_color%d.jpg" % count, frame_color)
#     # cv.imwrite("./batch_frames/frame_depth%d.jpg" % count, frame_depth)
#     # t = 1
#     while True:
#         # t-=1
#         cap.set(1, frame_associated)
#         dep.set(1, frame_associated)
#         frame_got_color, frame_color = cap.read()
#         frame_got_depth, frame_depth = dep.read()
#         if frame_got_color and frame_got_depth:
#             start = datetime.now()
#             rgb = Image.fromarray(frame_color)
#             depth = Image.fromarray(frame_depth).convert("I")
#             points = []
#             # print(rgb.size)
#             # print(depth.size)
#             # print(np.shape(rgb))
#             # print(np.array(depth))
#             # print(np.shape(depth))
#             points2 = []
#             depth_np = np.array(depth)
#             Z = depth_np * 1.0 / scalingFactor
#             X = (depth_np - centerX) * Z / focalLength
#             Y = (depth_np - centerY) * Z / focalLength
#             # print(np.shape(X),np.shape(Y),np.shape(Z))
#             final = np.dstack((X, Y, Z))
#             final = final[~np.all(final == 0, axis=2)]
#             # print(np.shape(final))
#             startfor = datetime.now()

#             for v in range(rgb.size[1]):
#                 for u in range(rgb.size[0]):
#                     color = rgb.getpixel((u, v))
#                     # print("rgb ",type(rgb), " ",rgb.size)
#                     # print("color at count u,v ",count, " ",u, " ",v," ",color)
#                     # Z = depth.getpixel((u,v)) / scalingFactor
#                     Z = depth.getpixel((u, v)) * 1.0 / scalingFactor
#                     if Z == 0: continue
#                     X = (u - centerX) * Z / focalLength
#                     Y = (v - centerY) * Z / focalLength
#                     # points2.append("%f %f %f\n" % (X, Y, Z))
#                     # points.append(X)
#                     # points.append(Y)
#                     # points.append(Z)
#                     # rgb_.append(color[0])
#                     # rgb_.append(color[1])
#                     # rgb_.append(color[2])
#                     points2.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
#             # endfor = datetime.now()
#             # print("executiontime innerfor count ", frame_associated , " ", (endfor-startfor) )

#             file = open(master_dir + "frame%d.ply" % frame_associated, "w")
#             file.write('''ply
#                 format ascii 1.0
#                 element vertex %d
#                 property float x
#                 property float y
#                 property float z
#                 property uchar red
#                 property uchar green
#                 property uchar blue
#                 property uchar alpha
#                 end_header
#                 %s
#                 ''' % (len(points2), "".join(points2)))
#             file.close()
#             # subprocess.run(["sudo", "chmod", "777", f"./batch_frames/frame{frame_associated}.ply"])
#             # subprocess.run(["/app/pydraco/draco/build_dir/draco_encoder", "-point_cloud", "-i", f"./batch_frames/frame{frame_associated}.ply", "-o", f"./batch_frames/frame{frame_associated}-cmd.drc"])
#             quantization_bits = 14
#             compression_level = 1
#             quantization_range = 1
#             quant_origin = 0
#             create_metadata = False
#             encoding_data = pydraco.encode_point_cloud(final.reshape(-1).tolist(), quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
#             end = datetime.now()
#             # os.nice(-19)
#             # os.system("sudo renice -n -19 -p " + str(os.getpid()))
#             # print(open("/proc/{pid}/stat".format(pid=os.getpid()), 'rb').read().split()[-14], frame_associated)
#             print("executiontime for count ", frame_associated , " ", (end-start) )
#             # with open(
#             #     os.path.join(f"./batch_frames/frame{frame_associated}.drc"), "wb"
#             # ) as test_file:
#             #     test_file.write(bytes(encoding_data.buffer))
#         else:
#             break
#         frame_associated += pool_size
#     return frame_associated


# if __name__ == '__main__':
#     # Setup.
#     cap = cv.VideoCapture(VIDEO_SOURCE)
#     dep = cv.VideoCapture(VIDEO_DEPTH_SOURCE)

#     # thread_num = 1
#     thread_num = cv.getNumberOfCPUs()
#     # pool = ThreadPool(processes=thread_num)
#     # pool = multiprocessing.Pool(processes=thread_num)
#     # pending_task = deque()
#     count = 0
#     # sys.exit(0)
#     starttt = datetime.now()
    
#     cc_list=[]
#     for i in range(thread_num):
#         p = multiprocessing.Process(target = process_frame, args = (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
#         p.start()
#         cc_list.append(p)
#         # task = pool.apply_async(process_frame, (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
#         # pending_task.append(task)
#     for i in range(thread_num):
#         cc_list[i].join()
#     # for i in range(thread_num):
#     #     task = pool.apply_async(process_frame, (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
#     #     pending_task.append(task)
#     # print(len(pending_task))
#     # while len(pending_task) > 0:
#     #         if pending_task[0].ready():
#     #             cc = pending_task.popleft().get()
#     #             cc_list.append(cc)
#     #         if cv.waitKey(1) == 27:
#     #             break
    
#     enddd =datetime.now()
#     print("starttime , endtime , executiontime ", starttt, enddd,  enddd-starttt)
#     print(thread_num)
#     print("count ",count)
#     # print(cc_list)
# cv.destroyAllWindows()