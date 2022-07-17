# import logging
# import sys
import socketserver
import socket
import threading
from os import getpid
from psutil import Process as Pro
# import binascii
# import time
from datetime import datetime

from frameDispenser import FramePipelineJob

# logging.basicConfig(format="%(asctime)s %(thread)d %(threadName)s %(message)s", stream=sys.stdout, level=logging.INFO)
# log = logging.getLogger()


class Handler(socketserver.BaseRequestHandler):
    # file_l = []
    # print("cming here")
    pkt_length = 65535
    # for i in range(1051, 1061):
    #     with open("../models/samples/sample" + str(i) + ".ply", "rb") as file:
    #         file_data = file.read()
    #         print(len(bytes(str(int(len(file_data))), "utf-8")))
    #         file_l.append(file_data)
    # print(type(file_data))
    # print(len(file_l))
    # print(binascii.hexlify(len(file_data).to_bytes(4, byteorder='big')))
    VIDEO_SOURCE = "../RGBD/sample_source/Sample00101_color.mp4"
    VIDEO_DEPTH_SOURCE = "../RGBD/sample_source/Sample00101_depth.mp4"

    def setup(self):
        # print("cming in setip")
        super().setup()
        self.event = threading.Event()
        self.data_transfered = 0
        self.dispenser = FramePipelineJob(self.VIDEO_SOURCE, self.VIDEO_DEPTH_SOURCE)
        # logging.info("New connection from {}".format(self.client_address))

    def handle(self):
        super().handle()
        sk: socket.socket = self.request
        try:
            self.dispenser.startProcessing()
            no_of_frames = 0
            startTime = datetime.now()
            startTime_ = datetime.now()
            data = 0
            timer = 0
            while True:
                timer += 1
                if timer % 160 == 0:
                    delta = datetime.now() - startTime_
                    speed = (self.data_transfered*8/(delta.total_seconds()))//1e6
                    print(speed, " Mb/s")
                    startTime_ = datetime.now()
                    self.data_transfered = 0
                current_frame = self.dispenser.getNextFrame()
                dif_len = 7 - len(bytes(str(len(current_frame)), "utf-8"))
                sk.send(b''.join([b'0'*dif_len,bytes(str(len(current_frame)), "utf-8")]))
                data += len(current_frame)
                for i in range(len(current_frame) // self.pkt_length + 1):
                    transferBytes = current_frame[self.pkt_length * i:self.pkt_length * (i + 1)]
                    self.data_transfered += len(transferBytes)
                    sk.send(transferBytes)
                no_of_frames += 1
                if no_of_frames % 25 == 0:
                    print(data // 1e3, ' KB ', datetime.now()-startTime)
                    mem, q = self.dispenser.getMemoryFootPrint()
                    print(Pro(getpid()).memory_info().rss // 1e6, " | ", mem, "MB q size : ", q)
                    startTime = datetime.now()
                    data = 0
                # logging.info("Transmission successful!")
        except Exception as e:
            # logging.info(e)
            print(e, "******")
            pass
        

    def finish(self):
        super().finish()
        self.dispenser.stopProcessing()
        self.event.set()
        self.request.close()


if __name__ == "__main__":
    server = socketserver.ThreadingTCPServer(("localhost", 12356), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="server", daemon=True).start()
    while True:
        cmd = input(">>>")
        if cmd.strip() == "quit":
            server.shutdown()
            server.server_close()
            break
        # logging.info(threading.enumerate())
