import socket
from threading import Thread

HOST, PORT = "127.0.0.1", 9876
ADDR = (HOST, PORT)
BUFF_SIZE = 1024


class ClientThread(Thread):
    def __init__(self,host,port,sock):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.client_sock = sock
        print('[Client({}, {})]: connected'.format(self.host, self.port))

    def run(self):
        while True:
            data = self.client_sock.recv(BUFF_SIZE)
            if not data:
                print('[Client({}, {})]: closed'.format(self.host, self.port))
                break

            print('[Client({}, {})]: {}'.format(self.host, self.port, data))

def main():
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serverSocket.bind(ADDR)
    threads = []

    serverSocket.listen(5)

    while True:
        print("Waiting for connections...")
        (clientSocket, (host, port)) = serverSocket.accept()
        print('Connection from ', (host, port))

        newthread = ClientThread(host, port, clientSocket)
        newthread.start()
        threads.append(newthread)

        for t in threads:
            t.join()


if __name__ == "__main__":
    main()

