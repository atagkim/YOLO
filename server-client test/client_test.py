import socket

# ip address and port of the server
HOST, PORT= 'localhost', 9876
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.connect((HOST, PORT))

while True:
    data = input('enter: ')
    # exit command
    if(data=='q'):
        print('exit')
        break

    client_socket.send(data.encode('ascii'))

client_socket.close()