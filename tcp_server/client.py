import socket
from frameCollector import FramePipelineJob
import multiprocessing
from multiprocessing import Process, Queue
from cv2 import getNumberOfCPUs
from datetime import datetime
import time

def startPipelineProcess(process_input_queues):
    # print("coming here")
    tcpClient = TCPClient('localhost', 12356, False, 7, 65535, "./tmp/test", process_input_queues)
    # print("coming here 2")
    tcpClient.processStream()
    # print("coming here 3")


class TCPClient:
    def __init__(self, IP_ADDRESS = 'localhost', PORT = 12344, SAVE_FILE = False, HEADER_SIZE = 10, PACKET_LEN = 8192, SAVE_LOCATION=None, process_input_queues = []):
        self.tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_client_socket.connect((IP_ADDRESS, PORT))
        print("setting up connection")
        self.SAVE_FILE = SAVE_FILE
        self.processed_frames = 0
        self.HEADER_SIZE = HEADER_SIZE
        self.remaining_processed_bytes = 0
        self.PACKET_LEN = PACKET_LEN
        self.HEADER_BUFFER = []
        self.DATA_BUFFER = []
        self.IS_HEADER_SET = False
        self.SAVE_LOCATION = SAVE_LOCATION
        self.FILE_DESCRIPTOR = None
        self.process_input_queues = process_input_queues
        self.thread_num = len(process_input_queues)
        self.data_transfered = 0

    def closeStream(self):
        self.tcp_client_socket.close()

    def processStream(self):
        startTime = datetime.now()
        timer = 0
        try:
            file = 0
            while True:
                timer += 1
                if timer % 10000 == 0:
                    delta = datetime.now() - startTime
                    speed = (self.data_transfered*8/(delta.total_seconds()))//1e6
                    print(speed, " Mb/s")
                    startTime = datetime.now()
                    self.data_transfered = 0
                # print("checking data")
                if not self.IS_HEADER_SET:
                    headerBufferRemainingLen = self.HEADER_SIZE - len(b''.join(self.HEADER_BUFFER))
                    currentBuffer = self.tcp_client_socket.recv(headerBufferRemainingLen)
                    # print(len(currentBuffer))
                    if len(currentBuffer) == 0:
                        continue
                    self.HEADER_BUFFER.append(currentBuffer)
                    headerInBytes = b''.join(self.HEADER_BUFFER)
                    if len(headerInBytes) == self.HEADER_SIZE:
                        self.IS_HEADER_SET = True
                        self.remaining_processed_bytes = int(str(headerInBytes, "utf-8"))
                        self.HEADER_BUFFER = []
                        # file = open("./tmp/test" +str(file_index)+".drc", "wb")
                        # file_index+=1
                else:
                    currentBuffer = self.tcp_client_socket.recv(min(self.remaining_processed_bytes, self.PACKET_LEN))
                    if len(currentBuffer) == 0:
                        continue
                    if self.SAVE_FILE:
                        if self.FILE_DESCRIPTOR == None:
                            self.FILE_DESCRIPTOR = open(self.SAVE_LOCATION + str(self.processed_frames) + ".drc", "wb")
                        self.FILE_DESCRIPTOR.write(currentBuffer)
                    self.DATA_BUFFER.append(currentBuffer)
                    self.remaining_processed_bytes -= len(currentBuffer)
                    self.data_transfered += len(currentBuffer)
                    if self.remaining_processed_bytes == 0:
                        self.IS_HEADER_SET = False
                        if self.SAVE_FILE:
                            self.FILE_DESCRIPTOR.close()
                            self.FILE_DESCRIPTOR = None
                        # self.process_input_queues[self.processed_frames%thread_num].put(b''.join(self.DATA_BUFFER))
                        self.processed_frames += 1
                        # print(self.processed_frames)
                        self.DATA_BUFFER = []                
        except Exception as e:
            print("Download Exception", e)
        else:
            print("Download Successful")






if __name__ == '__main__':
    thread_num = getNumberOfCPUs() // 4
    process_input_queues = []
    process_output_queues = []
    startTime = datetime.now()
    for i in range(thread_num):
        input_queue = Queue()
        output_queue = Queue()
        process_input_queues.append(input_queue)
        process_output_queues.append(output_queue)
    FrameProcessor = FramePipelineJob(process_input_queues, process_output_queues)
    FrameProcessor.startProcessing()
    socketProcess = multiprocessing.Process(target = startPipelineProcess, args=(process_input_queues, ))
    # print("is it not returning ddauihdiihih")
    socketProcess.start()
    current_frame_to_display = 0
    # while True:
    #     current_frame_info = process_output_queues[current_frame_to_display%thread_num].get()
    #     # print(current_frame_to_display, len(current_frame_info[0]), len(current_frame_info[1]))
    #     current_frame_to_display += 1
    #     if current_frame_to_display%100 == 0:
    #         # print("***************************")
    #         # print("--- %s seconds ---" % (time.time() - startTime))
    #         startTime = time.time()



