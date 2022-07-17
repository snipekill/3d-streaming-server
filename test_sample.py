from multiprocessing import Process, Queue
from sys import getrefcount
# from multiprocessing.pool import ThreadPool
from datetime import datetime
from cv2 import VideoCapture,getNumberOfCPUs,CAP_PROP_FRAME_COUNT
from PIL import Image
from psutil import Process as Pro
from gc import get_count
import tracemalloc
from pydraco import encode_point_cloud, decode_buffer_to_point_cloud
import numpy as np 
import os
import DracoPy

VIDEO_SOURCE = "./RGBD/sample_source/Sample00101_color.mp4"
VIDEO_DEPTH_SOURCE = "./RGBD/sample_source/Sample00101_depth.mp4"

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0

def frame_encode(points, colors, q):
    quantization_bits = 14
    compression_level = 1
    quantization_range = 1
    quant_origin = 0
    create_metadata = False
    # print("cinuf")
    encoding_data = encode_point_cloud(points,colors, [], 3, 0, quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
    # print("dsds")
    q.put(encoding_data.buffer)

def process_frame(frame_associated):
    cap = VideoCapture(VIDEO_SOURCE)
    dep = VideoCapture(VIDEO_DEPTH_SOURCE)
    tracemalloc.start()
    encoding_average = 0
    decoding_average = 0
    size_after_compression_average = 0

    index_ = 0
    inventory_mem = [0]
    time_arr = [0]
    tt = datetime.now()

    while True:
        index_ += 1
        if index_ % 20 == 0:
            delta = datetime.now() - tt
            # tt = datetime.now()
            time_arr.append(round(delta.total_seconds(),0))
            inventory_mem.append(Pro(os.getpid()).memory_info().rss // 1e6)
            # print("Mem: ", Pro(os.getpid()).memory_info().rss // 1e6)
            print(time_arr)
            print(inventory_mem)


        cap.set(1, frame_associated)
        dep.set(1, frame_associated)
        frame_got_color, frame_color = cap.read()
        frame_got_depth, frame_depth = dep.read()

        if frame_got_color and frame_got_depth:
            start = datetime.now()
            depth_np = np.array(Image.fromarray(frame_depth).convert("I"))
            color_np = np.array(Image.fromarray(frame_color))
            u = depth_np.shape[1]
            v = depth_np.shape[0]

            Z = depth_np * 1.0 / scalingFactor
            X = (np.arange(u) - centerX) * Z / focalLength
            Y = (np.arange(v).reshape(v,1) - centerY) * Z / focalLength
            final = np.dstack((X, Y, Z))
            final_arr = final[~np.all(final == 0, axis=2)]
            color_np = color_np[~np.all(final == 0, axis=2)]

            points_vector = final_arr.reshape(-1).tolist()
            colors_vector = color_np.reshape(-1).tolist()




            
            quantization_bits = 14
            compression_level = 0
            quantization_range = 1
            quant_origin = 0
            create_metadata = False
            encoding_data = encode_point_cloud(points_vector,colors_vector, [], 3, 0, quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
            data = bytes(encoding_data.buffer)
            # q = Queue()
            # processEntity = Process(target = frame_encode, args = (points_vector, colors_vector, q,))
            # processEntity.start()
            # data = q.get()
            # processEntity.terminate()
            time_diff = datetime.now() - start
            encoding_average += int(time_diff.total_seconds() * 1000)
            size_after_compression_average += len(data)
            # print(round(encoding_average/index_,0), "encoding time",round(size_after_compression_average/(1e3*index_),0), "KB")
            start = datetime.now()
            # point_cloud_object = decode_buffer_to_point_cloud(data, len(data))
            # time_diff = datetime.now() - start
            # decoding_average += int(time_diff.total_seconds() * 1000)
            # print(round(decoding_average/index_,0), "ms decoding time", len(point_cloud_object.points), len(point_cloud_object.rgba))
            # binary = DracoPy.encode(points_vector, colors_vector)
            # encoding_data = encode_point_cloud(points_vector,colors_vector, [], 3, 0, quantization_bits, compression_level, quantization_range, quant_origin, create_metadata)
            # print(len(points_vector), len(points))
            # print(len(colors_vector), len(rgb_))
            # for i in range(len(points)):
            #     if points[i] != points_vector[i]:
            #         print("suar", points[i], points_vector[i])
            #         break

            # for i in range(len(colors_vector)):
            #     if rgb_[i] != colors_vector[i]:
            #         print("suar", rgb_[i], colors_vector[i])
            #         break

        frame_associated += 1
    return frame_associated







if __name__ == "__main__":
    process_frame(0)