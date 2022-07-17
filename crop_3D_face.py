from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import torch
import torch.backends.cudnn as cudnn
import argparse

import cv2
import numpy as np
import open3d as o3d
import time, sys
from facenet_pytorch import MTCNN
from utils.torch_utils import select_device
import mediapipe as mp

# try:
#     sys.path.append('/usr/local/python')
#     from openpose import pyopenpose as op
# except ImportError as e:
#     print(
#         'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
#     raise e
def convert_recog_to_positions(face, yolo, bdbox, frame_num):
    print("face:----------", face)
    print("yolo-----------", yolo)
    print("bdbox----------", bdbox)

    x1 = bdbox[0] + (face[0] - yolo[0]) / (yolo[2] - yolo[0]) * (bdbox[1] - bdbox[0])
    x2 = bdbox[0] + (face[2] - yolo[0]) / (yolo[2] - yolo[0]) * (bdbox[1] - bdbox[0])
    y1 = bdbox[2] + (yolo[3]-face[1]) / (yolo[3] - yolo[1]) * (bdbox[3] - bdbox[2])
    y2 = bdbox[2] + (yolo[3]-face[3]) / (yolo[3] - yolo[1]) * (bdbox[3] - bdbox[2])
    print("face positions: ", x1, y1, x2, y2)

    # Control the ratio to tolerant the selection
    ratio = 0.5
    face_xmin = min(x1, x2) - (max(x1, x2) - min(x1, x2)) * ratio
    face_xmax = max(x1, x2) + (max(x1, x2) - min(x1, x2)) * ratio
    face_ymin = min(y1, y2) - (max(y1, y2) - min(y1, y2)) * ratio
    face_ymax = max(y1, y2) + (max(y1, y2) - min(y1, y2)) * ratio

    return [face_xmin, face_xmax, face_ymin, face_ymax]

def main():
    pcd = o3d.io.read_point_cloud("./models/longdress_vox10_1051.ply")
    o3d.io.write_point_cloud("copy_of_fragment.xyzrgb", pcd)
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    # o3d.visualization.draw_geometries([pcd])
    # print("diameter: ",diameter)
    device = select_device(args.device)
    mtcnnModel = MTCNN(keep_all=True, device=device)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()

    vol = o3d.visualization.read_selection_polygon_volume("./cropped.json")
    print("vol1: ", np.asarray(vol.bounding_polygon))

    # ======================================Load Openpose model=========================================================
    # try:
    #     params = dict()
    #     params["model_folder"] = "./models/"
    #     # params["face"] = True
    #     # params["hand"] = True
    #     # Starting OpenPose
    #     opWrapper = op.WrapperPython()
    #     opWrapper.configure(params)
    #     opWrapper.start()
    # except Exception as e:
    #     print(e)

    # =============================== Mediapipe ==========================================================
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    """============================Hidden Point Removal ============================================="""

    source, weights, view_img, save_txt, imgsz = args.source, args.weights, args.view_img, args.save_txt, args.img_size

    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # ======================================Load Yolo model=========================================================
    yoloModel = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(yoloModel.stride.max())
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        yoloModel.half()  # to FP16

    view_img = True
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Get names and colors
    yoloNames = yoloModel.module.names if hasattr(yoloModel, 'module') else yoloModel.names
    yoloColors = [[random.randint(0, 255) for _ in range(3)] for _ in yoloNames]

    if device.type != 'cpu':
        yoloModel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yoloModel.parameters())))  # run once

    """============================Hidden Point Removal ============================================="""
    print("Define parameters used for hidden_point_removal")
    camera = [500, 500, 1000]
    radius = diameter * 1000

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    # print("Visualize result")
    # pcd = pcd.select_by_index(pt_map)
    # o3d.visualization.draw([pcd], show_ui=True)

    frame_num = 0
    # cv2.namedWindow("rendered_source", cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow("face_detect", cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow("Yolo", cv2.WINDOW_GUI_NORMAL)

    while True:
        t0 = time.time()

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(True)
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = norm_image.astype(np.uint8)
        cv2.imshow("rendered_source", image)
        height, width, _ = image.shape
        face_position = []

        """============================Draw Face Bonding Box ============================================="""
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            face_results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Draw face detections of each face.
            if not face_results.detections:
                print("No Face Detected")
            else:
                annotated_image = image.copy()
                for detection in face_results.detections:
                    # print('Nose tip:')
                    # print(mp_face_detection.get_key_point(
                    #     detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    mp_drawing.draw_detection(annotated_image, detection)
                    print("detection: ",detection.location_data.relative_bounding_box)
                    face_reletive_box = detection.location_data.relative_bounding_box
                    face_position = [face_reletive_box.xmin * width,
                                     face_reletive_box.ymin * height,
                                     (face_reletive_box.xmin + face_reletive_box.width) * width,
                                     (face_reletive_box.ymin + face_reletive_box.height) * height]
                    print("face_box: ", face_position)
                cv2.imshow("face_detect", annotated_image)

        """============================Draw Yolo Bonding Box ============================================="""
        producer_frame = image.copy()
        # check for common shapes
        s = np.stack([letterbox(producer_frame, new_shape=imgsz)[0].shape], 0)  # inference shapes
        yolo_rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not yolo_rect:
            print('WARNING: Different stream shapes detected. Please supply similarly-shaped streams.')
        # Letterbox
        img = [letterbox(producer_frame, new_shape=imgsz, auto=yolo_rect)[0]]
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = yoloModel(img, augment=args.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes,
                                   agnostic=args.agnostic_nms)

        # Process detections
        yolo_results = []
        for i, det in enumerate(pred):  # detections per image
            p, s, display_image = "yolo", '', producer_frame.copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(display_image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to display_image size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], display_image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, yoloNames[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (yoloNames[int(cls)], conf)
                    plot_one_box(xyxy, display_image, label=label, color=yoloColors[int(cls)], line_thickness=3)
                    yolo_results.append(
                        [yoloNames[int(cls)], str(int(xyxy[0])), str(int(xyxy[1])), str(int(xyxy[2])),
                         str(int(xyxy[3]))])

            print("Results: ", yolo_results)
            cv2.imshow("Yolo", display_image)


        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        bdbox = np.asarray(aabb.get_box_points())
        # print(np.asarray(aabb.get_box_points()))
        print("source box: ", bdbox)

        xmax, ymax, zmax = bdbox.max(axis=0)
        xmin, ymin, zmin = bdbox.min(axis=0)
        print("bdbox: xmax, ymax, zmax: ", xmax, ymax, zmax)
        print("bdbox: xmin, ymin, zmin: ", xmin, ymin, zmin)

        face_loc = convert_recog_to_positions(face_position,
                                   [float(yolo_results[0][1]),float(yolo_results[0][2]),float(yolo_results[0][3]),float(yolo_results[0][4])],
                                   [xmin, xmax, ymin, ymax], frame_num)

        # cropBox = np.array([[0, 1000, 0],
        #                     [1000, 1000, 0],
        #                     [1000, 800, 0],
        #                     [0, 800, 0]])

        # cropBox = np.array([[face_loc[0], face_loc[1], 0],
        #                     [face_loc[2], face_loc[1], 0],
        #                     [face_loc[2], face_loc[3], 0],
        #                     [face_loc[0], face_loc[3], 0]])
        #
        # vol.update_bounding(o3d.cpu.pybind.utility.Vector3dVector(cropBox))
        # print("vol2: ", np.asarray(vol.bounding_polygon))


        model_minb = pcd.get_min_bound()
        model_maxb = pcd.get_max_bound()

        print("model_minb: ", model_minb)
        print("model_maxb: ", model_maxb)


        crop_pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
        crop_box = o3d.geometry.AxisAlignedBoundingBox(np.array([face_loc[0], face_loc[2], model_minb[2]], np.float64), np.array([face_loc[1], face_loc[3], model_maxb[2]], np.float64))

        # crop_pcd = vol.crop_point_cloud_test(crop_pcd)
        crop_pcd = crop_pcd.crop(crop_box)
        crop_pcd.paint_uniform_color((0, 1, 0))

        crop_reverse_pcd = o3d.cpu.pybind.geometry.PointCloud(pcd)
        crop_reverse_pcd = crop_reverse_pcd.crop_reverse(crop_box)
        crop_reverse_pcd.paint_uniform_color((0, 1, 1))

        # obb = pcd.get_oriented_bounding_box()
        obb = crop_pcd.get_axis_aligned_bounding_box()
        obb.color = (0, 1, 0)

        print("--------------------------------------FPS: ", 1 / (time.time() - t0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # o3d.visualization.draw([crop_pcd, crop_reverse_pcd, aabb, obb], show_ui=True)


        R = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
        # pcd.rotate(R, center=(0, 0, 0))
        pcd.rotate(R, center=pcd.get_center())
        frame_num += 1
        # ctr.rotate()
        # o3d.visualization.draw_geometries([pcd])

    vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NDN AR demo")
    parser.add_argument('--task', type=str, default='yolo', help='AR Task Type')  # Task
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--weights', nargs='+', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = parser.parse_args()
    main()


