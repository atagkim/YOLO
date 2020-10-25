import socket
import cv2
import time
import sys
from PyQt5.QtWidgets import *

OUR_IP_ADDR = "3.34.49.51"
# OUR_IP_ADDR = "127.0.0.1"
STUDENT = "1"


def check_student(name, cs):
    import cv2
    import time

    canvas = None

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

    beforeTime = 0
    currentTime = 0
    result = 0
    flag = 0

    last_switch = 0;

    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if (len(faces) == 0):

            ttm = time.time()
            rtm = time.gmtime(ttm)

            if (flag == 0):
                flag = 1
                beforeTime = rtm.tm_sec
                currentTime = beforeTime

            else:
                currentTime = rtm.tm_sec

            if (currentTime - beforeTime < 0):
                result = currentTime - beforeTime + 60
            else:
                result = currentTime - beforeTime

            if (result > 2 and time.time() - last_switch > 1):
                last_switch = time.time();

                data = '{} no attention'.format(name)
                print("data: ", data)

                cs.send(data.encode())

                print(name, " ?? ??? ??")
        else:
            currentTime = 0
            beforeTime = 0
            flag = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('frame', frame)

        esckey = cv2.waitKey(5) & 0xFF
        if esckey == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


class CLineEditWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Input Name")
        self.setGeometry(100, 100, 300, 100)

        self.label = QLabel("name : ", self)
        self.label.move(20, 20)
        self.label.resize(150, 20)

        self.lineEdit = QLineEdit("", self)
        self.lineEdit.move(60, 20)
        self.lineEdit.resize(200, 20)
        self.lineEdit.textChanged.connect(self.lineEdit_textChanged)

        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)

        btnSave = QPushButton("Save", self)
        btnSave.move(10, 50)
        btnSave.clicked.connect(self.btnSave_clicked)

        btnClear = QPushButton("Reset", self)
        btnClear.move(100, 50)
        btnClear.clicked.connect(self.btnClear_clicked)

        btnQuit = QPushButton("Exit", self)
        btnQuit.move(190, 50)
        btnQuit.clicked.connect(self.btnQuit_clicked)
        # btnQuit.clicked.connect(QCoreApplication.instance().quit)

    def btnSave_clicked(self):
        print("Save?")
        msg = "Do you want to save?"
        msg += "\nname : " + self.lineEdit.text()
        buttonReply = QMessageBox.question(self, 'save', msg,
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Cancel)

        if buttonReply == QMessageBox.Yes:
            print('Yes')
            # save
            QMessageBox.about(self, "Save", "Saved")
            self.statusBar.showMessage("Saved completely")

            name = self.lineEdit.text()
            print("name: ", name)
            self.close()

            # ip address and port of the server
            HOST, PORT = OUR_IP_ADDR, 9876
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            client_sock.connect((HOST, PORT))
            print("Connected with server")

            data = STUDENT
            client_sock.send(data.encode())

            check_student(name, client_sock)

        if buttonReply == QMessageBox.No:
            print('No')
        if buttonReply == QMessageBox.Cancel:
            print('Cancel')

    def btnClear_clicked(self):
        # clear
        self.lineEdit.clear()

    def btnQuit_clicked(self):
        # exit
        sys.exit()

    def lineEdit_textChanged(self):
        pass
        # self.statusBar.showMessage(self.lineEdit.text())


def main():
    app = QApplication(sys.argv)
    window = CLineEditWindow()
    window.show()
    app.exec_()


if __name__ == "__main__":
    # ip address and port of the server
    HOST, PORT = OUR_IP_ADDR, 9876
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_sock.connect((HOST, PORT))
    print("Connected with server")

    data = STUDENT
    client_sock.send(data.encode())

    while (True):
        time.sleep(1)
        data = '{} no attention'.format("test")
        print("data: ", data)

        client_sock.send(data.encode())
    # main()
