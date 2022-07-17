import socket
import time
import binascii

tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_server_socket.bind(("", 12333))
tcp_server_socket.listen(128)

file_l = []

for i in range(1051,1061):
    with open("../models/samples/sample"+ str(i)+ ".drc", "rb") as file:
        file_data = file.read()
        file_l.append(file_data)

print(len(file_data))
print(len(file_l))
print(binascii.hexlify(len(file_data).to_bytes(4, byteorder='big')))

pkt_length = 1024

while True:
    client_socket, client_ip = tcp_server_socket.accept()
    print("Client：", client_ip, "Connection")
    # file_name_data = client_socket.recv(1024)
    # file_name = file_name_data.decode()
    frame_num = 0
    try:
        while True:
            file_data = file_l[frame_num % len(file_l)]
            client_socket.send(len(file_data).to_bytes(4, byteorder='big'))
            for i in range(len(file_data) // pkt_length+1):
                client_socket.send(file_data[pkt_length * i:pkt_length * (i + 1)])
            print("Transmission successful!")
            frame_num += 1
            time.sleep(0.1)
    except Exception as e:
        print("Exception：", e)
    client_socket.close()
