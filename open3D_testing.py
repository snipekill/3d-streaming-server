import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time

def main():
    print("Load a ply point cloud, print it, and render it")
    # pcd = o3d.io.read_point_cloud("./models/longdress_vox10_1051.ply")
    pcd = o3d.io.read_point_cloud("./models/longdress_vox10_1051.ply")
    # pcd = o3d.io.read_point_cloud("./RGBD/model.ply")
    # pcd2 = o3d.io.read_point_cloud("./models/longdress_vox10_1060.ply")
    # pcd2 = o3d.io.read_point_cloud("./simple_sample/sample5.ply")
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    o3d.visualization.draw([pcd], show_ui=True)
    # convert Open3D.o3d.geometry.PointCloud to numpy array
    # xyz_load = np.asarray(pcd.points)
    # xyz_load = np.asarray(pcd.points.colors)
    #
    # print('xyz_load')
    # print(xyz_load)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()

    # render = vis.rendering.OffscreenRenderer(1920, 1080)
    # render.scene.add_geometry("box", pcd)
    # render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    # img = render.render_to_image()

    """============================Hidden Point Removal ============================================="""
    diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    print("Define parameters used for hidden_point_removal")
    camera = [500, 500, 1000]
    radius = diameter * 1000

    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)

    print("Visualize result")
    rem_pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw([rem_pcd], show_ui=True)

    while True:
        t0 = time.time()
        # depth = vis.capture_depth_float_buffer(do_render=False)
        # plt.imsave("depth.png", np.asarray(depth), dpi=1)
        # print(depth)
        image = vis.capture_screen_float_buffer(True)
        # plt.imsave("image.png", np.asarray(image), dpi = 1)
        # plt.imshow(image)
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = norm_image.astype(np.uint8)
        cv2.imshow("results", image)


        print("--------------------------------------FPS: ", 1/(time.time() - t0))
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # ctr.change_field_of_view(0.1)
        # parameters = ctr.convert_to_pinhole_camera_parameters()
        # print(parameters)
        # fv = ctr.get_field_of_view()
        ctr.rotate(10, 0)
        # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_xxxx.json")
        # ctr.convert_from_pinhole_camera_parameters(parameters)

    vis.destroy_window()

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    # img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    # o3d.io.write_image("../../TestData/sync.png", img)
    # o3d.visualization.draw_geometries([img])


if __name__ == "__main__":
    main()