import socket
from threading import Thread

HOST, PORT = "", 9876
ADDR = (HOST, PORT)
BUFF_SIZE = 1024
TEACHER = b"0"
STUDENT = b"1"
teacher_sock = "x"
sv = 1

class StudentThread(Thread):
    def __init__(self,host,port,sock,id):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_sock = sock
        self.id = id
        print('[Student({}, {})]: connected'.format(self.host, self.port))

    def run(self):
        global teacher_sock
        global sv

        while True:
            data = self.client_sock.recv(BUFF_SIZE)
            if not data:
                print('[Student({}, {})]: closed'.format(self.host, self.port))
                break
            print('[Student({}, {})]: {}'.format(self.host, self.port, data.decode()))

            teacher_sock.send(data.encode())
            try:
                sv += 1
            except:
                print("Teacher is not connected")


class TeacherThread(Thread):
    def __init__(self, host, port, sock, id):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_sock = sock
        self.id = id
        print('[Teacher({}, {})]: connected'.format(self.host, self.port))

    def run(self):
        import time
        global sv
        while True:
            time.sleep(1)
            print("sv:", sv)
            # data = self.client_sock.recv(BUFF_SIZE)
            # if not data:
            #     break


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
        print("id: {}".format(id))
        if id==TEACHER:
            newthread = TeacherThread(host, port, client_sock, id)
            print("Teacher is connected")
        elif id==STUDENT:
            newthread = StudentThread(host, port, client_sock, id)
            print("Student is connected")
        newthread.start()


if __name__ == "__main__":
    main()

