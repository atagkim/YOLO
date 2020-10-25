import socket

BUFF_SIZE = 1024
TEACHER = "0"

OUR_IP_ADDR = "3.34.49.51"
# OUR_IP_ADDR = "127.0.0.1"
# ip address and port of the server
HOST, PORT = OUR_IP_ADDR, 9876
client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_sock.connect((HOST, PORT))
print("Connected with server")

data = TEACHER
client_sock.send(data.encode())

while True:
    data = client_sock.recv(BUFF_SIZE)

    print("[SERVER]: {}".format(data))