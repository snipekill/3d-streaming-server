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
from collections import deque
import multiprocessing
from multiprocessing.pool import ThreadPool
from datetime import datetime
import cv2 as cv
from PIL import Image
import subprocess
import sys
import os

import argparse
import sys
import pydraco
import os
from PIL import Image
import numpy as np
from scipy import misc
import scipy.ndimage
import os

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0

VIDEO_SOURCE = "./RGBD/sample_source/Sample00101_color.mp4"
VIDEO_DEPTH_SOURCE = "./RGBD/sample_source/Sample00101_depth.mp4"


def process_frame(VIDEO_SOURCE, VIDEO_DEPTH_SOURCE, frame_associated, pool_size):
    cap = cv.VideoCapture(VIDEO_SOURCE)
    dep = cv.VideoCapture(VIDEO_DEPTH_SOURCE)
    # some intensive computation...
    master_dir = "./batch_frames/"
    # cv.imwrite("./batch_frames/frame_color%d.jpg" % count, frame_color)
    # cv.imwrite("./batch_frames/frame_depth%d.jpg" % count, frame_depth)
    # t = 1
    while True:
        # t-=1
        cap.set(1, frame_associated)
        dep.set(1, frame_associated)
        frame_got_color, frame_color = cap.read()
        frame_got_depth, frame_depth = dep.read()
        if frame_got_color and frame_got_depth:
            start = datetime.now()
            rgb = Image.fromarray(frame_color)
            depth = Image.fromarray(frame_depth).convert("I")
            points = []
            # print(rgb.size)
            # print(depth.size)
            # print(np.shape(rgb))
            # print(np.array(depth))
            # print(np.shape(depth))
            points2 = []
            depth_np = np.array(depth)
            Z = depth_np * 1.0 / scalingFactor
            X = (depth_np - centerX) * Z / focalLength
            Y = (depth_np - centerY) * Z / focalLength
            # print(np.shape(X),np.shape(Y),np.shape(Z))
            final = np.dstack((X, Y, Z))
            final = final[~np.all(final == 0, axis=2)]
            # print(np.shape(final))
            startfor = datetime.now()

            for v in range(rgb.size[1]):
                for u in range(rgb.size[0]):
                    color = rgb.getpixel((u, v))
                    # print("rgb ",type(rgb), " ",rgb.size)
                    # print("color at count u,v ",count, " ",u, " ",v," ",color)
                    # Z = depth.getpixel((u,v)) / scalingFactor
                    Z = depth.getpixel((u, v)) * 1.0 / scalingFactor
                    if Z == 0: continue
                    X = (u - centerX) * Z / focalLength
                    Y = (v - centerY) * Z / focalLength
                    # points2.append("%f %f %f\n" % (X, Y, Z))
                    # points.append(X)
                    # points.append(Y)
                    # points.append(Z)
                    # rgb_.append(color[0])
                    # rgb_.append(color[1])
                    # rgb_.append(color[2])
                    points2.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
            # endfor = datetime.now()
            # print("executiontime innerfor count ", frame_associated , " ", (endfor-startfor) )

            file = open(master_dir + "frame%d.ply" % frame_associated, "w")
            file.write('''ply
                format ascii 1.0
                element vertex %d
                property float x
                property float y
                property float z
                property uchar red
                property uchar green
                property uchar blue
                property uchar alpha
                end_header
                %s
                ''' % (len(points2), "".join(points2)))
            file.close()
            # subprocess.run(["sudo", "chmod", "777", f"./batch_frames/frame{frame_associated}.ply"])
            # subprocess.run(["/app/pydraco/draco/build_dir/draco_encoder", "-point_cloud", "-i", f"./batch_frames/frame{frame_associated}.ply", "-o", f"./batch_frames/frame{frame_associated}-cmd.drc"])
            quantization_bits = 14
            compression_level = 1
            quantization_range = 1
            quant_origin = 0
            create_metadata = False
            encoding_data = pydraco.encode_point_cloud(final.reshape(-1).tolist(), quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
            end = datetime.now()
            # os.nice(-19)
            # os.system("sudo renice -n -19 -p " + str(os.getpid()))
            # print(open("/proc/{pid}/stat".format(pid=os.getpid()), 'rb').read().split()[-14], frame_associated)
            print("executiontime for count ", frame_associated , " ", (end-start) )
            # with open(
            #     os.path.join(f"./batch_frames/frame{frame_associated}.drc"), "wb"
            # ) as test_file:
            #     test_file.write(bytes(encoding_data.buffer))
        else:
            break
        frame_associated += pool_size
    return frame_associated


if __name__ == '__main__':
    # Setup.
    cap = cv.VideoCapture(VIDEO_SOURCE)
    dep = cv.VideoCapture(VIDEO_DEPTH_SOURCE)

    # thread_num = 1
    thread_num = cv.getNumberOfCPUs()
    # pool = ThreadPool(processes=thread_num)
    # pool = multiprocessing.Pool(processes=thread_num)
    # pending_task = deque()
    count = 0
    # sys.exit(0)
    starttt = datetime.now()
    
    cc_list=[]
    for i in range(thread_num):
        p = multiprocessing.Process(target = process_frame, args = (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
        p.start()
        cc_list.append(p)
        # task = pool.apply_async(process_frame, (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
        # pending_task.append(task)
    for i in range(thread_num):
        cc_list[i].join()
    # for i in range(thread_num):
    #     task = pool.apply_async(process_frame, (VIDEO_SOURCE,VIDEO_DEPTH_SOURCE,i, thread_num,))
    #     pending_task.append(task)
    # print(len(pending_task))
    # while len(pending_task) > 0:
    #         if pending_task[0].ready():
    #             cc = pending_task.popleft().get()
    #             cc_list.append(cc)
    #         if cv.waitKey(1) == 27:
    #             break
    
    enddd =datetime.now()
    print("starttime , endtime , executiontime ", starttt, enddd,  enddd-starttt)
    print(thread_num)
    print("count ",count)
    # print(cc_list)
cv.destroyAllWindows()