import socket
from threading import Thread

import time

HOST, PORT = "", 9876
ADDR = (HOST, PORT)
BUFF_SIZE = 1024
TEACHER = b"0"
STUDENT = b"1"
tthreads = []

class StudentThread(Thread):
    def __init__(self,host,port,sock,id):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_sock = sock
        self.id = id
        print('[Student({}, {})]: connected'.format(self.host, self.port))

    def run(self):
        while True:
            global tthreads
            data = self.client_sock.recv(BUFF_SIZE)
            tthreads[-1].send(data)


class TeacherThread(Thread):
    def __init__(self, host, port, sock, id):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_sock = sock
        self.id = id
        print('[Teacher({}, {})]: connected'.format(self.host, self.port))


    def run(self):
        global tthreads
        tthreads.append(self.client_sock)

        while True:
            data = self.client_sock.recv(BUFF_SIZE)


def main():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(ADDR)

    server_sock.listen(5)

    while True:
        print("Waiting for connections...")
        (client_sock, (host, port)) = server_sock.accept()
        print('Connection from ', (host, port))

        id = client_sock.recv(BUFF_SIZE)

        if id==TEACHER:
            newthread = TeacherThread(host, port, client_sock, id)
            newthread.start()
            print("Teacher is connected")
        elif id==STUDENT:
            newthread = StudentThread(host, port, client_sock, id)
            newthread.start()
            print("Student is connected")



if __name__ == "__main__":
    main()

