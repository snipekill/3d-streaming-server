import logging
import sys
import socketserver
import socket
import threading
import binascii
import time

logging.basicConfig(format="%(asctime)s %(thread)d %(threadName)s %(message)s", stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()


class Handler(socketserver.BaseRequestHandler):
    file_l = []
    print("cming here")
    pkt_length = 1024
    for i in range(1051, 1061):
        with open("../models/samples/sample" + str(i) + ".ply", "rb") as file:
            file_data = file.read()
            print(len(bytes(str(int(len(file_data))), "utf-8")))
            file_l.append(file_data)
    print(type(file_data))
    print(len(file_l))
    print(binascii.hexlify(len(file_data).to_bytes(4, byteorder='big')))

    def setup(self):
        print("cming in setip")
        super().setup()
        self.event = threading.Event()
        logging.info("New connection from {}".format(self.client_address))

    def handle(self):
        super().handle()
        sk: socket.socket = self.request
        # while not self.event.is_set():
            # frame_num = 0
        try:
            # while True:
            for frame_num in range(len(self.file_l)):
                # data = sk.recv(1024).decode()
                # file_data = self.file_l[frame_num % len(self.file_l)]
                file_data = self.file_l[frame_num]
                print(len(file_data))
                sk.send(bytes(str(len(file_data)), "utf-8"))
                # sk.send(len(file_data).to_bytes(4, byteorder='little'))
                # print(len(str(len(file_data)).encode('UTF-8')))
                # sk.send(str(len(file_data)).encode('UTF-8'))
                # sk.send(len(file_data).to_bytes(4, byteorder='big').encode('utf-8'))
                for i in range(len(file_data) // self.pkt_length + 1):
                    sk.send(file_data[self.pkt_length * i:self.pkt_length * (i + 1)])
                logging.info("Transmission successful!")
                # frame_num += 1
                # time.sleep(0.1)
        except Exception as e:
            logging.info(e)
        

    def finish(self):
        super().finish()
        self.event.set()
        self.request.close()


if __name__ == "__main__":
    server = socketserver.ThreadingTCPServer(("localhost", 12338), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="server", daemon=True).start()
    while True:
        cmd = input(">>>")
        if cmd.strip() == "quit":
            server.shutdown()
            server.server_close()
            break
        # logging.info(threading.enumerate())
