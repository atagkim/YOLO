import cv2
import time
from tkinter import messagebox

import numpy as np


def alert():
    messagebox.showinfo("51번학생이", "수업에 집중을 하지 않고 있습니다.")

canvas = None

#웹캠에서 영상을 읽어온다
cap = cv2.VideoCapture(0)
cap.set(3, 1280) #WIDTH
cap.set(4, 720) #HEIGHT

#얼굴 인식 캐스케이드 파일 읽는다
face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')

beforeTime = 0
currentTime = 0
result = 0
flag = 0


while(True):
    # frame 별로 capture 한다
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #현재 시간 체크

    #인식된 얼굴 갯수를 출력
    if(len(faces) == 0):

        ttm = time.time()
        rtm = time.gmtime(ttm)

        if(flag == 0):
            flag = 1
            beforeTime = rtm.tm_sec
            currentTime = beforeTime

        else:
            currentTime = rtm.tm_sec

        if(currentTime - beforeTime < 0):
            result = currentTime - beforeTime + 60
        else:
            result = currentTime - beforeTime

        print("beforeTime",beforeTime)
        print("currentTime", currentTime)

        if(result > 2):
            cv2.putText(frame, '25 student no attention', (900, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 0)
            print("집중 안하고 있닭")


    else:
        currentTime = 0
        beforeTime = 0
        flag = 0


    # 인식된 얼굴에 사각형을 출력한다
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #화면에 출력한다
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()