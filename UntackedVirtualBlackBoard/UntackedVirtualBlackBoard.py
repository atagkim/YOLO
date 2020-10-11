import cv2
import numpy as np
import time
import keyboard


def nothing(x):
    pass


def initialize_HSV_values():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")

        lower_range = np.array([l_h, l_s, l_v])
        upper_range = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower_range, upper_range)

        res = cv2.bitwise_and(frame, frame, mask=mask)

        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        stacked = np.hstack((mask_3,frame,res))

        cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))

        key = cv2.waitKey(1)
        if key == 27:
            break

        # Press 's' button to save HSV values and exit
        if key == ord('s'):
            thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
            print(thearray)

            np.save('penval',thearray)
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame


def start_blackboard():
    load_from_disk = True
    if load_from_disk:
        penval = np.load('penval.npy')

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # initialize
    pen_img = cv2.resize(cv2.imread('images/pen.png', 1), (50, 50))
    eraser_img = cv2.resize(cv2.imread('images/eraser.jpg', 1), (50, 50))
    paint_cap_img = cv2.resize(cv2.imread('images/camera1.png', 1), (50, 50))
    change_color_img = cv2.resize(cv2.imread('images/change_color_img.png', 1), (50, 50))
    change_font_size_img = cv2.resize(cv2.imread('images/change_font_size.png', 1), (50, 50))

    kernel = np.ones((5, 5), np.uint8)

    cv2.namedWindow('Untacked Virtual Blackboard', cv2.WINDOW_NORMAL)

    canvas = None

    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    background_threshold = 400

    switch = 'Pen'

    last_switch = time.time()

    x1, y1 = 0, 0

    cutchk = False
    cutcanvas = None
    tmpcanvas = None
    stx = 0
    sty = 0
    edx = 0
    edy = 0
    cutw = 0
    cuth = 0
    v_chk = False

    noiseth = 800

    wiper_thresh = 40000

    clear = False
    paint_cap = False
    change_color = False
    change_font_size = False


    font_color = [255, 0, 0]
    font_size = 5
    font_size_erase = 20

    draw_delay = False
    draw_chk = False

    cnt = 0

    # start capturing
    while (1):
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        if cutcanvas is None:
            cutcanvas = np.zeros_like(frame)

        pen_or_eraser_frame = frame[0:50, 0:50]
        paint_cap_frame = frame[0:50, 150:200]
        change_color_frame = frame[0:50, 300:350]
        change_font_size_frame = frame[0:50, 450:500]

        fgmask = backgroundobject.apply(pen_or_eraser_frame)
        fgmask_paint_cap = backgroundobject.apply(paint_cap_frame)
        fgmask_change_color = backgroundobject.apply(change_color_frame)
        fgmask_change_font_size = backgroundobject.apply(change_font_size_frame)

        switch_thresh = np.sum(fgmask == 255)
        paint_cap_thresh = np.sum(fgmask_paint_cap == 255)
        change_color_thresh = np.sum(fgmask_change_color == 255)
        font_size_thresh = np.sum(fgmask_change_font_size == 255)

        if switch_thresh > background_threshold and (time.time() - last_switch) > 1:

            last_switch = time.time()

            if switch == 'Pen':
                switch = 'Eraser'
            else:
                switch = 'Pen'

        if paint_cap_thresh > background_threshold and (time.time() - last_switch) > 1:
            paint_cap = True

        if change_color_thresh > background_threshold and (time.time() - last_switch) > 1:
            change_color = True

        if font_size_thresh > background_threshold and (time.time() - last_switch) > 1:
            change_font_size = True

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if load_from_disk:
            lower_range = penval[0]
            upper_range = penval[1]

        else:
            lower_range = np.array([30, 80, 110])
            upper_range = np.array([50, 200, 200])

        mask = cv2.inRange(hsv, lower_range, upper_range)

        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:

            c = max(contours, key=cv2.contourArea)
            x2, y2, w, h = cv2.boundingRect(c)

            area = cv2.contourArea(c)

            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2

            else:
                if switch == 'Pen':
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
        else:
            x1, y1 = 0, 0

        _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(foreground, background)

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

        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), font_size_erase, (255, 255, 255), -1)
            frame[0:50, 0:50] = eraser_img
        else:
            frame[0:50, 0:50] = pen_img

        frame[0:50, 150:200] = paint_cap_img
        frame[0:50, 300:350] = change_color_img
        frame[0:50, 450:500] = change_font_size_img

        cv2.imshow('Untacked Virtual Blackboard', frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

        elif k == ord('x'):
            stx, sty = x2, y2

        elif k == ord('c'):
            edx, edy = x2, y2
            cutcanvas[sty:edy, stx:edx] = canvas[sty:edy, stx:edx]
            canvas[sty:edy, stx:edx] = 0
            cutchk = True

        elif k == ord('v'):
            v_chk = True

        if clear == True:
            time.sleep(0.5)
            canvas = None

            clear = False

            draw_delay = True

        if paint_cap == True:
            time.sleep(0.25)
            print("그림판 캡쳐")
            cv2.imwrite("save/ScreenShot{}.png".format(cnt), canvas)
            paint_cap = False
            cnt += 1
            draw_delay = True

        if change_color == True:
            change_color = False

            time.sleep(0.25)
            print('펜 컬러 변경')

            if font_color[0] and 255:
                font_color[0] = 0
                font_color[2] = 255
            else:
                font_color[0] = 255
                font_color[2] = 0

            draw_delay = True

        if change_font_size == True:
            change_font_size = False

            time.sleep(0.25)
            print('폰트 사이즈 변경')

            if font_size == 5:
                font_size = 20
                font_size_erase = 100
            else:
                font_size = 5
                font_size_erase = 20

            draw_delay = True

    cv2.destroyAllWindows()
    cap.release()


def main():
    initialize_HSV_values()
    start_blackboard()


if __name__ == "__main__":
    main()


