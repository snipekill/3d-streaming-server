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
import multiprocessing as mp
from multiprocessing import Process
from threading import Thread
from multiprocessing.pool import ThreadPool
from datetime import datetime
import cv2 as cv
from PIL import Image
import subprocess

import argparse
import sys
import os
from PIL import Image
import numpy as np
from scipy import misc
import scipy.ndimage
import time

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0

VIDEO_SOURCE = "./RGBD/sample_source/Sample00101_color.mp4"
VIDEO_DEPTH_SOURCE = "./RGBD/sample_source/Sample00101_depth.mp4"

def process_for_loop(que, frame_color,frame_depth,count):
    rgb = Image.fromarray(frame_color)
    depth = Image.fromarray(frame_depth).convert("I")
    my_dict={}
    points = []
    # print((frame_color.shape))
    # print(rgb.size[1], rgb.size[0])
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            # color = rgb.getpixel((u, v))
            # print("rgb",type(rgb), " ",rgb.size)
            # print("color at count u,v ",count, " ",u, " ",v," ",color)
            # break
            # Z = depth.getpixel((u,v)) / scalingFactor
            Z = depth.getpixel((u, v)) * 1.0 / scalingFactor
            if Z == 0:
                continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            # points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
            points.append(X)
            points.append(Y)
            points.append(Z)
    
    my_dict[count]=points
    que.put(my_dict)    
    return 
    

def process_frame(frame_color,frame_depth, count):
    startFrame = time.time()
    # some intensive computation...
    master_dir = "./batch_frames/"
    # thread_num = cv.getNumberOfCPUs()
    # inner_pool = ThreadPool(processes=thread_num)
    # inner_pending_task = deque()
    # points_dict = {}
    # outer_points_list=[]
    startfor = time.time()
    
    cores=mp.cpu_count()
    # print("number of cores",cores)
    split_size = len(frame_color) // cores

    procs = [] 
    que = mp.Queue()
    for i in range(cores): 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                      
        # special case on the last chunk to account for uneven splits           
        end = None if i+1 == cores else (i+1) * split_size 
        proc = Thread(target=process_for_loop, args=(que, frame_color[start:end], frame_depth[start:end], start))
        procs.append(proc)
        proc.start()

    # collecting data from all process from queue
    l = [que.get() for p in procs]  
    print("l..... ",len(l), type(l[0]), l[0].keys(), l[1].keys(),l[2].keys(), l[3].keys())
    outer_points_list =[]
    sorted_dict = {}
    for bla in l:
        sorted_dict.update(bla)
    
    for k,v in sorted(sorted_dict.items()):
        print(k)
        outer_points_list=outer_points_list+v
    
    # print("outer_points_list ",outer_points_list)
    for proc in procs:
        proc.join()
    #   print("joined proc")
    print("all processes returned")

    endfor = datetime.now()
    # print("executiontime innerfor count ", count , " ", endfor-startfor)
    print("--- %s seconds --- executiontime innerfor count" % (time.time() - startfor))

    file = open(master_dir + "frame%d_.ply" % count, "w")
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
        ''' % (len(outer_points_list), "".join(outer_points_list)))
    file.close()
    subprocess.run(["sudo", "chmod", "777", f"./batch_frames/frame{count}_.ply"])
    subprocess.run(["/app/pydraco/draco/build_dir/draco_encoder", "-point_cloud", "-i", f"./batch_frames/frame{count}_.ply", "-o", f"./batch_frames/frame{count}_.drc"])

    # ./app/pydraco/draco/build_dir/draco_encoder -point_cloud -i ./batch_frames/frame${count}.ply -o out.drc

    endFrame = datetime.now()
    print("--- %s seconds --- for total process" % (time.time() - startFrame))
    # print("executiontime for count ", count , " ", endFrame-startFrame)
    return count


if __name__ == '__main__':
    # Setup.
    cap = cv.VideoCapture(VIDEO_SOURCE)
    dep = cv.VideoCapture(VIDEO_DEPTH_SOURCE)

    thread_num = cv.getNumberOfCPUs()
    # pool = ThreadPool(processes=thread_num)
    pool = mp.Pool(processes=thread_num)
    pending_task = deque()

    print(thread_num)

    count = 0
    start = datetime.now()
    
    cc_list=[]
    while True:
        # Consume the queue when the threads returns.
        while len(pending_task) > 0 and pending_task[0].ready():
            cc = pending_task.popleft().get()
            cc_list.append(cc)
            # print(" cc returned ",cc)
            # print("inside using the queue conteny ", len(res))
            # cv.imwrite("./batch_frames/frame%d.jpg" % count, res)
            # count += 1
            # cv.imshow('threaded video', res)

        # Populate the queue.
        if len(pending_task) < thread_num:
            frame_got_color, frame_color = cap.read()
            frame_got_depth, frame_depth = dep.read()
            if frame_got_color and frame_got_depth:
                task = pool.apply_async(process_frame, (frame_color.copy(),frame_depth.copy(),count,))
                pending_task.append(task)
                count += 1
                # if(count==10):
                #     break
                # print(" len of pending_task after appending",len(pending_task))


        # Show preview.
        if cv.waitKey(1) == 27 or not frame_got_color:
            "aaya idhar ? "
            break
    
    end =datetime.now()
    print("starttime , endtime , executiontime ", start, end,  end-start)
    print(thread_num)
    print("count ",count)
    print(cc_list)
cv.destroyAllWindows()