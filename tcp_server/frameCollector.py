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
# from multiprocessing.pool import ThreadPool
from datetime import datetime
import cv2 as cv
from PIL import Image
# import subprocess
# import sys
# import os

# import argparse
# import sys
from pydraco import decode_buffer_to_point_cloud
import DracoPy
# import os
from PIL import Image
import numpy as np
import time
# from scipy import misc
# import scipy.ndimage
# import os

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0


def process_frame(input_queue, output_queue):
    while True:
        # print("coming *********************************")
        data = input_queue.get()
        # print("stuck ***********************************")
        # point_cloud_object = DracoPy.decode(data)
        point_cloud_object = decode_buffer_to_point_cloud(data, len(data))
        # print(len(point_cloud_object.points),len(point_cloud_object.rgba), type(point_cloud_object.rgba))
        # time.sleep(0.042)
        output_queue.put([point_cloud_object.points, point_cloud_object.rgba])
        # output_queue.put([[], []])
        # print(len(point_cloud_object.points), len(point_cloud_object.faces))
        # output_queue.put([point_cloud_object.points, point_cloud_object.faces])



class FramePipelineJob:
    def __init__(self, process_input_queues, process_output_queues):
        self.frameIndex = 0
        self.process_input_index = 0
        self.process_output_index = 0
        self.thread_num = cv.getNumberOfCPUs() // 4
        self.process_input_queues = process_input_queues
        self.process_output_queues = process_output_queues
        self.processes = []

    def startProcessing(self):
        for i in range(self.thread_num):
            processEntity = Process(target = process_frame, args = (self.process_input_queues[i], self.process_output_queues[i], ))
            processEntity.start()
            self.processes.append(processEntity)

    def stopProcessing(self):
        for i in range(self.thread_num):
            self.processes[i].terminate()