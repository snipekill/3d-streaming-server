import socket
import sys

tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# tcp_client_socket.bind(("", 12345))
tcp_client_socket.connect(("localhost", 12338))
index = 0
file_index = 0
remaininglength = 0
len_set = -1
header_size = 7
pkt_length = 1024
file_length_arr = []
try:
    file = 0
    while True:
        if len_set == -1:
            # print("waiting for server to send file size")
            len_data_required = header_size - len(b''.join(file_length_arr))
            file_data = tcp_client_socket.recv(len_data_required)
            # print("hey", len(file_data), type(file_data), type(file_length_arr))
            if len(file_data) == 0:
                continue
            file_length_arr.append(file_data)
            bytes_data = b''.join(file_length_arr)
            # print(bytes_data, type(bytes_data))
            # data = bytes.join(b''.join(file_length_arr))
            if len(bytes_data) == header_size:
                len_set = 1
                remaininglength = int(str(bytes_data, "utf-8"))
                # print(remaininglength)
                # sys.exit(0)
                file_length_arr = []
                file = open("./tmp/test" +str(file_index)+".ply", "wb")
                file_index+=1
        else:
            file_data = tcp_client_socket.recv(min(remaininglength, pkt_length))
            if len(file_data):
                file.write(file_data)
            remaininglength -= len(file_data)
            if remaininglength == 0:
                file.close()
                len_set = -1
    # with open("./tmp/test" +str(file_index)+".drc", "wb") as file:
    #     while True:
    #         index += 1
    #         # if index == 20:
    #             # break
    #         file_data = tcp_client_socket.recv(1024)
    #         # file_data_to_display = str(file_data, "utf-8")
    #         # print(type(file_data), len(file_data), file_data_to_display)
    #         if file_data:
    #             file.write(file_data)
    #             print(file_data)
    #         else:
    #             break
except Exception as e:
    print("Download Exception", e)
else:
    print("Download Successful")
tcp_client_socket.close()