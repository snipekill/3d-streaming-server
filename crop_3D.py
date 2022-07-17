import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
from facenet_pytorch import MTCNN


pcd = o3d.io.read_point_cloud("./models/longdress_vox10_1060.ply")
diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
# o3d.visualization.draw_geometries([pcd])
print("diameter: ",diameter)

mtcnnModel = MTCNN(keep_all=True, device="gpu")

print("Define parameters used for hidden_point_removal")
camera = [500, 500, 1000]
radius = diameter * 1000

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

# print("Visualize result")
# pcd = pcd.select_by_index(pt_map)
# o3d.visualization.draw([pcd], show_ui=True)

aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
bdbox = np.asarray(aabb.get_box_points())
print(np.asarray(aabb.get_box_points()))
print("source: ", bdbox)

cropBox = np.array([[0, 1000, 0],
                    [1000,1000,0],
                    [1000,800,0],
                    [0,800,0]])

vol = o3d.visualization.read_selection_polygon_volume("./cropped.json")
print("vol1: ", np.asarray(vol.bounding_polygon))

vol.update_bounding(o3d.cpu.pybind.utility.Vector3dVector(cropBox))
print("vol2: ", np.asarray(vol.bounding_polygon))

pcd = vol.crop_point_cloud_test(pcd)
# pcd = cpbox.crop_point_cloud(pcd)

# obb = pcd.get_oriented_bounding_box()
obb = pcd.get_axis_aligned_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw([pcd, aabb, obb], show_ui=True)
    #                               zoom=0.7,
    #                               front=[0.5439, -0.2333, -0.8060],
    #                               lookat=[2.4615, 2.1331, 1.338],
    #                               up=[-0.1781, -0.9708, 0.1608])

