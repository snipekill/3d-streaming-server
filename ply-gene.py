import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import matplotlib.pyplot as plt


class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])


    def start_processing_stream(self):
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            are_truedepth_camera_data_being_streamed = depth.shape[0] == 640
            if are_truedepth_camera_data_being_streamed:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Show the RGBD Stream
            cv2.imshow('RGB', rgb)
            cv2.imshow('Depth', depth)

            cv2.imwrite("camcolor.jpg", np.asarray(rgb))
            plt.imsave("camdepth.png", np.asarray(depth*255, np.uint16))
            depth_channel = cv2.convertScaleAbs(depth)
            depth_channel = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            # depth_channel = cv2.convertScaleAbs(depth)
            cv2.imwrite("depth.png", depth_channel)
            # cv2.imwrite("depth.png", np.asarray(depth,np.float32))
            # cv2.imwrite("depth.png", np.asarray(depth))

            focalLength = 525.0
            centerX = 319.5
            centerY = 239.5
            scalingFactor = 2.0

            points = []
            print(depth.shape)
            for v in range(rgb.shape[1]):
                for u in range(rgb.shape[0]):
                    color = rgb[u][v]
                    # Z = depth.getpixel((u,v)) / scalingFactor
                    Z = depth[u][v] * 255.0 / scalingFactor
                    if Z == 0: continue
                    X = (u - centerX) * Z / focalLength
                    Y = (v - centerY) * Z / focalLength
                    points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[2], color[1], color[1]))
            file = open("pccam.ply", "w")
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




            print(depth * 255)

            print("rgb shape: ", rgb.shape)
            print("depth shape: ", depth.dtype)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.event.clear()


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
