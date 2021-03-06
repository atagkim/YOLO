import cv2
import numpy as np
import time
import keyboard
import modules.DrawOpenGL as DrawOpenGL
import modules.helper as helper
from PIL import ImageGrab

import socket
BUFF_SIZE = 1024
TEACHER = "0"

studentflag2 = 0
studenttime = 0
studentflag = 0
studentname = None
screentime = 0
screenflag = False

OUR_IP_ADDR = "3.34.49.51"
# OUR_IP_ADDR = "127.0.0.1"

# 콜백용으로 만들어놓은 아무것도 아닌 함수
def nothing(x):
    pass

# 가상 칠판에서 쓸 펜 초기값을 설정하는 함수
def initialize_HSV_values():

    # 캠 초기화
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # 윈도우 생성
    windowName = "pen init"
    cv2.namedWindow("Trackbars")

    # 색 설정을 위한 바 생성
    # Now create 6 tracbars that will control the lower and upper range of H,S & V channels.
    # The Arguments are like this: Name of trackbar, window name, range, callback function.
    # For Hue the range is 0-179 and for S,V its 0-255.
    # H는 색상, S는 채도, V는 채도
    # 색상은 원래 가시광선 스펙트럼 고리모양 (0도 == 360도(빨강))
    # 채도는 진하기로 생각하면 됨
    # 명도는 밝은 정도, 정확한 색을 100%, 검은색을 0%로 생각하면 됨
    # OpenCV에선 H는 0~180, S,V는 0~255로 표현(8bit) -1노-
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    # 돌아 돌아
    while True:

        ## [프레임 전처리]
        # 캠 리딩
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전 할거면 해
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # Get the new values of the trackbar in real time as the user changes them
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        # Set the lower and upper HSV range according to the value selected by the trackbar
        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])


        ## [색 설정에 따른 결과값 화면 생성]
        # 설정된 값에 인식 되는 파트만 바이너리로 마스크 따오는거
        # Filter the image and get the binary mask, where white represents your target color
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # You can also visualize the real part of the target color (Optional)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # 합체
        stacked = np.hstack((mask_3,frame,res))

        # Show this stacked frame at 40% of the size.
        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))


        key = cv2.waitKey(1)
        if key == 27:
            break

        # Press 's' button to save HSV values and exit
        if key == ord('s'):
            # 설정값 저장
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)

            np.save('penval',thearray)
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame


def start_blackboard():
    # 펜 값 로딩
    load_from_disk = True
    if load_from_disk:
        penval = np.load('penval.npy')

    # 캠 설정
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    #툴바
    tool_eraser_img=cv2.resize(cv2.imread('images/tool_eraser.png', 1), (650, 50))
    tool_pen_img=cv2.resize(cv2.imread('images/tool_pen.png', 1), (650, 50))
    help_img=cv2.resize(cv2.imread('images/help.png', 1), (50, 50))

    #색깔
    chacol_img=cv2.resize(cv2.imread('images/chacol.png', 1), (50, 50))
    green_img=cv2.resize(cv2.imread('images/green.png', 1), (50, 50))
    pink_img=cv2.resize(cv2.imread('images/pink.png', 1), (50, 50))
    red_img=cv2.resize(cv2.imread('images/red.png', 1), (50, 50))
    white_img=cv2.resize(cv2.imread('images/white.png', 1), (50, 50))

    #펜 크기
    pen5_img=cv2.resize(cv2.imread('images/5px.png', 1), (50, 50))
    pen10_img=cv2.resize(cv2.imread('images/10px.png', 1), (50, 50))
    pen15_img=cv2.resize(cv2.imread('images/15px.png', 1), (50, 50))
    pen20_img=cv2.resize(cv2.imread('images/20px.png', 1), (50, 50))
    pen30_img=cv2.resize(cv2.imread('images/30px.png', 1), (50, 50))

    kernel = np.ones((5, 5), np.uint8)

    # Making window size adjustable
    cv2.namedWindow('Untacked Virtual Blackboard', cv2.WINDOW_NORMAL)

    # 정적인 부분을 걸러내는거. 이유는 뭐 적당히 그런거 때문일거임
    # Create a background subtractor Object
    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)


    ## [이거저거 변수들 초기값 설정]
    # 가장 먼저 캔버스부터
    canvas = None

    # 좌표 초기화
    x1, y1 = 0, 0

    switch = 'Pen'
    last_switch = time.time() # With this variable we will monitor the time between previous switch.

    # 잘라내기 좌표
    cutchk = False
    cutcanvas = None
    tmpcanvas = None
    stx = 0
    sty = 0
    edx = 0
    edy = 0
    estx = 0
    esty = 0
    eedx = 0
    eedy = 0

    # 확대 축소 좌표 & 플래그

    expchk = False
    redchk = False
    v_chk = False
    z_chk = False

    # 기능들 초기 상태
    clear = False
    paint_cap = False
    change_color = False
    change_font_size = False
    add_3d=False
    help_chk = False

    # 딜레이용 변수들
    draw_delay = False
    draw_chk = False
    additinalDelay = 0.5 # 초단위임 time.time 정수파트는

    screenshotCnt = 0

    # 펜 속성
    font_color = [255, 0, 0]
    font_size = 5
    font_size_erase = 20

    # threshold 설정
    noiseth = 800
    wiper_thresh = 40000
    background_threshold = 200 # This threshold determines the amount of disruption in background.


    # start capturing
    while (1):

        # 프레임 초기화
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Initilize the canvas as a black image
        if canvas is None:
            canvas = np.zeros_like(frame)

        if cutcanvas is None:
            cutcanvas = np.zeros_like(frame)


        # 펜 설정값 로딩
        if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]

        else:
            lower_range = np.array([30, 80, 110])
            upper_range = np.array([50, 200, 200])


        ## 펜 인식 과정
        # hsv로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 펜 설정값에 맞게 촤좌작
        mask = cv2.inRange(hsv, lower_range, upper_range)

        # 이거저거 하면 좋은 것들
        # Perform morphological operations to get rid of the noise
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 펜 윤곽선 따기
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 기능 버튼들 설정
        pen_or_eraser_frame = mask[0:50, 0:50]
        paint_cap_frame = mask[0:50, 150:200]
        change_color_frame = mask[0:50, 300:350]
        change_font_size_frame = mask[0:50, 450:500]
        add_3d_frame = mask[0:50, 600:650]
        help_frame = mask[0:50, 1230:1280]

        # Note the number of pixels that are white,this is the level of disruption.
        switch_thresh = np.sum(pen_or_eraser_frame == 255)
        paint_cap_thresh = np.sum(paint_cap_frame == 255)
        change_color_thresh = np.sum(change_color_frame == 255)
        font_size_thresh = np.sum(change_font_size_frame == 255)
        add_3d_thresh = np.sum(add_3d_frame == 255)
        help_thresh = np.sum(help_frame == 255)

        # 첫번째
        add_chacol_frame = mask[110:160, 0:50]
        add_chacol_thresh = np.sum(add_chacol_frame == 255)
        # 두번째
        add_green_frame = mask[210:260, 0:50]
        add_green_thresh = np.sum(add_green_frame == 255)
        # 세번째
        add_pink_frame = mask[310:360, 0:50]
        add_pink_thresh = np.sum(add_pink_frame == 255)
        # 네번째
        add_red_frame = mask[410:460, 0:50]
        add_red_thresh = np.sum(add_red_frame == 255)
        # 다섯번째
        add_white_frame = mask[510:560, 0:50]
        add_white_thresh = np.sum(add_white_frame == 255)


        # If the disruption is greater than background threshold and there has been some time after the previous switch
        # then you can change the object type.
        if switch_thresh > background_threshold and (
                time.time() - last_switch - additinalDelay) > 1:  # 단순 -1은 분명 역전있을거같긴한데 일단 패스

            last_switch = time.time()

            print("펜 지우개 토글")

            if switch == 'Pen':
                switch = 'Eraser'
            else:
                switch = 'Pen'

        if paint_cap_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
            last_switch = time.time()

            paint_cap = True

        if change_color_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
            last_switch = time.time()

            if change_color == False:
                change_color = True
            else:
                change_color = False


        if font_size_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
            last_switch = time.time()

            if change_font_size == False:
                change_font_size = True
            else:
                change_font_size = False

        if add_3d_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
            last_switch = time.time()

            add_3d = True

        if help_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
            last_switch = time.time()
            help_chk = True


        # Make sure there is a contour present and also its size is bigger than noise threshold.
        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:

            c = max(contours, key=cv2.contourArea)
            x2, y2, w, h = cv2.boundingRect(c)

            # Get the area of the contour
            area = cv2.contourArea(c)

            # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2

            else:
                if switch == 'Pen':
                    # 맨 처음 그려질때 튀는거 막기 위해
                    if draw_delay == True:
                        draw_delay = False

                        time.sleep(0.25)
                        x1, y1 = x2, y2

                    if keyboard.is_pressed(' '):
                        canvas = cv2.line(canvas, (x1, y1), (x2, y2), font_color, font_size)

                else:
                    cv2.circle(canvas, (x2, y2), font_size_erase, (0, 0, 0), -1)

            x1, y1 = x2, y2

            if area > wiper_thresh:
                cv2.putText(canvas, 'Clearing Canvas', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1,
                            cv2.LINE_AA)
                clear = True

        # 윤곽선 탐지 안 될경우 포인터 위치 초기화
        else:
            x1, y1 = 0, 0

        # 하면 아마도 좋은 것들
        # Now this piece of code is just for smooth drawing. (Optional)
        _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(foreground, background)

        global studentflag2
        global studentflag
        global studentname

        global screentime
        global screenflag
        if screenflag == True:
            if time.time() - screentime < 1:
                cv2.putText(frame, "ScreenShot Completed", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                screenflag = False

        if(studentflag==1):
            if(studentflag2 == 0):
                studenttime = time.time()
                studentflag2 = 1

            if(time.time() - studenttime < 1):

                message = studentname
                message = message + " student no attention"
                cv2.putText(frame, message, (500, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 0)

            else:
                studentflag = 0
                studentflag2 = 0

        # 잘라내기 기능
        if cutchk == True:
            cutx = edx - stx
            cuty = edy - sty
            if tmpcanvas is None:
                tmpcanvas = np.zeros_like(frame)
            for i in range(0, cuty):
                for j in range(0, cutx):
                    if y2 + i >= 720 or x2 + j >= 1280:
                        continue
                    tmpcanvas[y2 + i][x2 + j] = cutcanvas[sty + i][stx + j]

            frame = cv2.add(frame, tmpcanvas)
            if z_chk == True:
                canvas = cv2.add(canvas,tmpcanvas)
                z_chk = False
            if v_chk == False:
                for i in range(0, cuty):
                    for j in range(0, cutx):
                        if y2 + i >= 720 or x2 + j >= 1280:
                            continue
                        tmpcanvas[y2 + i][x2 + j] = 0
            else:
                canvas = cv2.add(canvas, tmpcanvas)
                v_chk = False
                cutcanvas = None
                tmpcanvas = None
                cutchk = False

        # 확대축소 기능
        elif expchk == True or redchk==True:
            if tmpcanvas is None:
                tmpcanvas = np.zeros_like(frame)
            tmpimg = canvas[esty:eedy,estx:eedx]
            cx = int((estx+eedx)/2)
            cy = int((esty+eedy)/2)

            if(expchk==True):
                big = cv2.resize(tmpimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                height, width, _ = big.shape
                height = int(height)
                width = int(width)
                ypoint = max(cy-int(height/2),0)
                xpoint = max(cx-int(width/2),0)
                suby = max(0,ypoint+height-720)
                subx = max(0,xpoint+width-1280)

                canvas[esty:eedy, estx:eedx] = 0
                tmpcanvas[ypoint:ypoint+height-suby,xpoint:xpoint+width-subx] = big[0:height-suby,0:width-subx]

            else:
                small = cv2.resize(tmpimg, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                height, width, _ = small.shape
                height = int(height)
                widht = int(width)
                ypoint = cy-int(height/2)
                xpoint = cx-int(widht/2)
                canvas[esty:eedy, estx:eedx] = 0
                tmpcanvas[ypoint:ypoint+height,xpoint:xpoint+width] = small


            canvas = cv2.add(canvas,tmpcanvas)
            tmpcanvas = None
            expchk=False
            redchk=False


        frame[0:50, 1230:1280] = help_img
        # 기능버튼 아이콘 변경
        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), font_size_erase, (255, 255, 255), -1)
            frame[0:50, 0:650] = tool_eraser_img
            #frame[0:50, 0:50] = eraser_img
        else:
            frame[0:50, 0:650] = tool_pen_img
            #frame[0:50, 0:50] = pen_img

            #frame[0:50, 150:200] = paint_cap_img
            #frame[0:50, 300:350] = change_color_img
            #frame[0:50, 450:500] = change_font_size_img
            #frame[0:50, 600:650] = add_3d_img

        if change_color == True :
            if change_font_size==True:
                change_color=False
            else:
                # 640,360
                frame[110:160, 0:50] = chacol_img
                frame[210:260, 0:50] = green_img
                frame[310:360, 0:50] = pink_img
                frame[410:460, 0:50] = red_img
                frame[510:560, 0:50] = white_img

        if change_font_size == True:
            if change_color == True:
                change_font_size = False
            else:
                frame[110:160, 0:50] = pen5_img
                frame[210:260, 0:50] = pen10_img
                frame[310:360, 0:50] = pen15_img
                frame[410:460, 0:50] = pen20_img
                frame[510:560, 0:50] = pen30_img

        # 프레임 쇼
        cv2.imshow('Untacked Virtual Blackboard', frame)


        ## [키 설정]
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        # 잘라내기 x -> c
        elif k == ord('x'):
            stx, sty = x2, y2

        elif k == ord('c'):
            edx, edy = x2, y2
            cutcanvas[sty:edy, stx:edx] = canvas[sty:edy, stx:edx]
            canvas[sty:edy, stx:edx] = 0
            cutchk = True

        # 붙여넣기 v
        elif k == ord('v'):
            v_chk = True

        # z
        elif k == ord('z'):
            z_chk = True

        # 확대축소 기능
        elif k== ord('a'):
            estx, esty = x2, y2

        elif k== ord('s'):
            eedx, eedy = x2,y2

        elif k==ord('d'):
            expchk=True

        elif k == ord('f'):
            redchk = True

        #클립보드 이미지 획득
        elif k == ord('o'):

            if tmpcanvas is None:
                tmpcanvas = np.zeros_like(frame)

            im = ImageGrab.grabclipboard()
            try:
                im.save('images/clipboard.png', 'PNG')  # PNG 포맷으로 저장
                clipboard_img = cv2.resize(cv2.imread('images/clipboard.png', 1), (250, 250))
                tmpcanvas[225:475,525:775]=clipboard_img
                canvas=cv2.add(canvas,tmpcanvas)
            except AttributeError:
                print("클립보드에 이미지가 없습니다")

            tmpcanvas = None

        if add_3d==True:
            if tmpcanvas is None:
                tmpcanvas = np.zeros_like(frame)
            if DrawOpenGL.myOpenGL():
                cube_img = cv2.resize(cv2.imread('images/3D.png', 1), (500, 500))
                tmpcanvas[100:600, 400:900] = cube_img
                canvas = cv2.add(canvas, tmpcanvas)
            tmpcanvas=None
            add_3d=False

        ## [기능들 동작 과정]
        # clear canvas
        if clear == True:

            time.sleep(0.25)

            canvas = None

            clear = False

            draw_delay = True
        # 캡쳐 기능 동작
        if paint_cap == True:

            print("그림판 캡쳐")

            cv2.imwrite("save/ScreenShot{}.png".format(screenshotCnt), canvas)
            paint_cap = False
            screenshotCnt += 1
            draw_delay = True
            screentime = time.time()
            screenflag = True




        # 펜 속성 변경
        if change_color == True:

            if add_chacol_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                print('펜 컬러 변경')
                #b
                font_color[0]=255
                #g
                font_color[1]=255
                #r
                font_color[2]=0
                change_color = False
                change_color = False

            if add_green_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                print('펜 컬러 변경')
                #b
                font_color[0]=204
                #g
                font_color[1]=255
                #r
                font_color[2]=102
                change_color = False

            if add_pink_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                print('펜 컬러 변경')
                #b
                font_color[0]=204
                #g
                font_color[1]=51
                #r
                font_color[2]=255
                change_color = False

            if add_red_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                print('펜 컬러 변경')
                #b
                font_color[0]=51
                #g
                font_color[1]=0
                #r
                font_color[2]=255
                change_color = False

            if add_white_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                print('펜 컬러 변경')
                #b
                font_color[0]=255
                #g
                font_color[1]=255
                #r
                font_color[2]=255
                change_color = False

        if change_font_size == True:

            if add_chacol_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                font_size = 5
                font_size_erase = 20
                change_font_size = False

            if add_green_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                font_size = 10
                font_size_erase = 50
                change_font_size = False

            if add_pink_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                font_size = 15
                font_size_erase = 70
                change_font_size = False

            if add_red_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                last_switch = time.time()
                font_size = 20
                font_size_erase = 100
                change_font_size = False

            if add_white_thresh > background_threshold and (time.time() - last_switch - additinalDelay) > 1:
                font_size = 30
                font_size_erase = 150
                change_font_size = False

        if help_chk == True:
            helper.helper_func()
            help_chk = False

    cv2.destroyAllWindows()
    cap.release()


def check_student():

    # ip address and port of the server
    HOST, PORT = OUR_IP_ADDR, 9876
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_sock.connect((HOST, PORT))
    print("Connected with server")

    data = TEACHER
    client_sock.send(data.encode())

    while True:

        global studentname
        global studentflag
        # 졸고 있는 학생 이름
        data = client_sock.recv(BUFF_SIZE)

        studentflag = 1
        studentname = data.decode('utf-8')

        print("from server: {}".format(data))


def main():
    from threading import Thread
    t = Thread(target=check_student)
    t.start()

    initialize_HSV_values()
    start_blackboard()

    t.join()


if __name__ == "__main__":
    main()