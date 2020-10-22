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

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


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


        if(result > 2 and time.time() - last_switch > 1):
            last_switch = time.time();

            cv2.putText(frame, '25th student no attention', (840, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 0)

            print("beforeTime", beforeTime)
            print("currentTime", currentTime)
            print("집중 안하고 있닭")


    else:
        currentTime = 0
        beforeTime = 0
        flag = 0


    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


    cv2.imshow('frame',frame)

    esckey = cv2.waitKey(5) & 0xFF
    if esckey == 27:
        break



cap.release()
cv2.destroyAllWindows()