"""
This script reads a registered pair of color and depth images and generates a
colored 3D point cloud in the PLY format.
"""

import argparse
import sys
import os
from PIL import Image
import numpy as np
from scipy import misc
import scipy.ndimage
from datetime import datetime

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 2.0

import time

def current_milli_time():
    return time.time() * 1000


# def generate_pointcloud(rgb_file,depth_file,ply_file):
def generate_pointcloud():
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """

    master_dir = "./RGBD/"
    rgb = Image.open(master_dir + "modelimage.jpg")
    depth = Image.open(master_dir + "modeldepth.png").convert("I")
    # dep = np.fromfile(master_dir + "00000.png", dtype=np.uint16).reshape(480, 640)[::-1, :]
    # dep = scipy.ndimage.rotate(dep, 90)
    # tempmin = 0  # np.amin(np.array(dep.flatten())[dep.flatten() != 0])
    # dep = (dep - tempmin) * 1.0 / (np.max(dep.flatten()) - tempmin)
    # dep[dep == dep[0][0]] = 0


    # rgb = rgb.crop((363,1,574,266))
    # depth = depth.crop((363,1,574,266))
    '''if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")'''

    points = []
    print(rgb.size)
    print(np.shape(rgb))
    print(np.array(depth))
    print(np.shape(depth))
    start_time = current_milli_time()

    # Z = depth.getpixel((rgb[0], v)) * 1.0 / scalingFactor
    # Z = np.array(depth)*1.0/ scalingFactor
    # # print((current_milli_time() - start_time))
    # # start_time = current_milli_time()
    # # break
    # # if Z == 0: continue
    # X = (np - centerX) * Z / focalLength
    # Y = (v - centerY) * Z / focalLength

    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            # start_time = current_milli_time()
            # color = rgb.getpixel((u, v))
            # print(u, v)
            break
            # Z = depth.getpixel((u,v)) / scalingFactor
            Z = depth.getpixel((u, v)) * 1.0 / scalingFactor
            # print((current_milli_time() - start_time))
            start_time = current_milli_time()
            break
            if Z == 0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
            # print((current_milli_time() - start_time))
    # print((current_milli_time() - start_time)/1000)
    file = open(master_dir + "model.ply", "w")
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
        ''' % (len(points), "".join(points)))
    file.close()


if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description='''
    This script reads a registered pair of color and depth images and generates a colored 3D point cloud in the
    PLY format. 
    ''')
    parser.add_argument('rgb_file', help='input color image (format: png)')
    parser.add_argument('depth_file', help='input depth image (format: png)')
    parser.add_argument('ply_file', help='output PLY file (format: ply)')
    args = parser.parse_args()"""

    # generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)
    generate_pointcloud()


