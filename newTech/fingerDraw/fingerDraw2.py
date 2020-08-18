import cv2
import numpy as np
import time
import keyboard

'''

# step1
# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# Intializing the webcam feed.
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Create a window named trackbars.
cv2.namedWindow("Trackbars")

# Now create 6 tracbars that will control the lower and upper range of H,S & V channels.
# The Arguments are like this: Name of trackbar, window name, range, callback function.
# For Hue the range is 0-179 and for S,V its 0-255.

##
# H는 색상, S는 채도, V는 채도
# 색상은 원래 가시광선 스펙트럼 고리모양 (0도 == 360도(빨강))
# 채도는 진하기로 생각하면 됨
# 명도는 밝은 정도, 정확한 색을 100%, 검은색을 0%로 생각하면 됨
# OpenCV에선 H는 0~180, S,V는 0~255로 표현(8bit)
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)


while True:

    # Start reading the webcam feed frame by frame.
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally (Not required)
    frame = cv2.flip( frame, 1 )

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

    # Filter the image and get the binary mask, where white represents your target color
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    ## 어퍼바운드, 로어바운드 정해서 마스크 씌움(해당되는 색만 뽑기)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so we can stack it with the others
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,frame,res))

    # Show this stacked frame at 40% of the size.
    cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))

    # If the user presses ESC then exit the program
    key = cv2.waitKey(1)
    if key == 27:
        break

    # If the user presses `s` then print this array.
    if key == ord('s'):

        thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
        print(thearray)

        # Also save this array as penval.npy
        np.save('penval',thearray)
        break

# Release the camera & destroy the windows.
cap.release()
cv2.destroyAllWindows()





'''




#
# # This variable determines if we want to load color range from memory or use the ones defined here.
# load_from_disk = True
# #step2
# # This variable determines if we want to load color range from memory or use the ones defined here.
# #load_from_disk = False
#
# # If true then load color range from memory
# if load_from_disk:
#     penval = np.load('penval.npy')
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
#
# # Creating A 5x5 kernel for morphological operations
# kernel = np.ones((5,5),np.uint8)
#
# while(1):
#
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame = cv2.flip( frame, 1 )
#
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # If you're reading from memory then load the upper and lower ranges from there
#     if load_from_disk:
#             lower_range = penval[0]
#             upper_range = penval[1]
#
#     # Otherwise define your own custom values for upper and lower range.
#     else:
#        #lower_range  = np.array([26,80,147])
#        #lower_range  = np.array([26,80,147])
#        lower_range = np.array([32, 80, 140])
#        upper_range = np.array([81,255,255])
#
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#
#     # Perform the morphological operations to get rid of the noise.
#     # Erosion Eats away the white part while dilation expands it.
#     mask = cv2.erode(mask,kernel,iterations = 1)
#     mask = cv2.dilate(mask,kernel,iterations = 2)
#
#     res = cv2.bitwise_and(frame,frame, mask= mask)
#
#     mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#
#     # stack all frames and show it
#     stacked = np.hstack((mask_3,frame,res))
#     cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()
#
# # This variable determines if we want to load color range from memory or use the ones defined in notebook.
# load_from_disk = True
# #step3
# # This variable determines if we want to load color range from memory or use the ones defined in notebook.
# #load_from_disk = False
#
# # If true then load color range from memory
# if load_from_disk:
#     penval = np.load('penval.npy')
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
#
# # kernel for morphological operations
# kernel = np.ones((5,5),np.uint8)
#
# # set the window to autosize so we can view this full screen.
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
# # This threshold is used to filter noise, the contour area must be bigger than this to qualify as an actual contour.
# noiseth = 500
#
# while(1):
#
#     _, frame = cap.read()
#     frame = cv2.flip( frame, 1 )
#
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # If you're reading from memory then load the upper and lower ranges from there
#     if load_from_disk:
#             lower_range = penval[0]
#             upper_range = penval[1]
#
#     # Otherwise define your own custom values for upper and lower range.
#     else:
#         # lower_range  = np.array([26,80,147])
#         lower_range = np.array([32, 80, 140])
#         upper_range = np.array([81,255,255])
#
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#
#     # Perform the morphological operations to get rid of the noise
#     mask = cv2.erode(mask,kernel,iterations = 1)
#     mask = cv2.dilate(mask,kernel,iterations = 2)
#
#     # Find Contours in the frame.
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_SIMPLE)
#
#     # Make sure there is a contour present and also make sure its size is bigger than noise threshold.
#     if contours and cv2.contourArea(max(contours,
#                                         key = cv2.contourArea)) > noiseth:
#
#         # Grab the biggest contour with respect to area
#         c = max(contours, key = cv2.contourArea)
#
#         # Get bounding box coordinates around that contour
#         x,y,w,h = cv2.boundingRect(c)
#
#         # Draw that bounding box
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,25,255),2)
#
#     cv2.imshow('image',frame)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()
#
#
#
# #step4
# load_from_disk = True
# if load_from_disk:
#     penval = np.load('penval.npy')
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
#
# kernel = np.ones((5,5),np.uint8)
#
# # Initializing the canvas on which we will draw upon
# canvas = None
#
# # Initilize x1,y1 points
# x1,y1=0,0
#
# # Threshold for noise
# noiseth = 800
#
# while(1):
#     _, frame = cap.read()
#     frame = cv2.flip( frame, 1 )
#
#     # Initilize the canvas as a black image of same size as the frame.
#     if canvas is None:
#         canvas = np.zeros_like(frame)
#
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # If you're reading from memory then load the upper and lower ranges from there
#     if load_from_disk:
#             lower_range = penval[0]
#             upper_range = penval[1]
#
#     # Otherwise define your own custom values for upper and lower range.
#     else:
#         # lower_range  = np.array([26,80,147])
#         lower_range = np.array([32, 80, 140])
#         upper_range = np.array([81,255,255])
#
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#
#     # Perform morphological operations to get rid of the noise
#     mask = cv2.erode(mask,kernel,iterations = 1)
#     mask = cv2.dilate(mask,kernel,iterations = 2)
#
#     # Find Contours
#     contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Make sure there is a contour present and also its size is bigger than the noise threshold.
#     if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
#
#         c = max(contours, key = cv2.contourArea)
#         x2,y2,w,h = cv2.boundingRect(c)
#
#         # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
#         # This is true when we writing for the first time or when writing again when the pen had disapeared from view.
#         if x1 == 0 and y1 == 0:
#             x1,y1= x2,y2
#
#         else:
#             # Draw the line on the canvas
#             canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 4)
#
#         # After the line is drawn the new points become the previous points.
#         x1,y1= x2,y2
#
#     else:
#         # If there were no contours detected then make x1,y1 = 0
#         x1,y1 =0,0
#
#     # Merge the canvas and the frame.
#     frame = cv2.add(frame,canvas)
#
#     # Optionally stack both frames and show it.
#     stacked = np.hstack((canvas,frame))
#     cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.6,fy=0.6))
#
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#
#     # When c is pressed clear the canvas
#     if k == ord('c'):
#         canvas = None
#
# cv2.destroyAllWindows()
# cap.release()
#
#
#
# #step5
# load_from_disk = True
# if load_from_disk:
#     penval = np.load('penval.npy')
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
#
# kernel = np.ones((5,5),np.uint8)
#
# # Making window size adjustable
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#
# # This is the canvas on which we will draw upon
# canvas=None
#
# # Initilize x1,y1 points
# x1,y1=0,0
#
# # Threshold for noise
# noiseth = 800
#
# # Threshold for wiper, the size of the contour must be bigger than for us to clear the canvas
# wiper_thresh = 40000
#
# # A varaible which tells when to clear canvas, if its True then we clear the canvas
# clear = False
#
# while(1):
#     _ , frame = cap.read()
#     frame = cv2.flip( frame, 1 )
#
#     # Initilize the canvas as a black image
#     if canvas is None:
#         canvas = np.zeros_like(frame)
#
#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # If you're reading from memory then load the upper and lower ranges from there
#     if load_from_disk:
#             lower_range = penval[0]
#             upper_range = penval[1]
#
#     # Otherwise define your own custom values for upper and lower range.
#     else:
#         # lower_range  = np.array([26,80,147])
#         lower_range = np.array([25, 70, 120])
#         upper_range = np.array([81,255,255])
#
#     mask = cv2.inRange(hsv, lower_range, upper_range)
#
#     # perform the morphological operations to get rid of the noise
#     mask = cv2.erode(mask,kernel,iterations = 1)
#     mask = cv2.dilate(mask,kernel,iterations = 2)
#
#     # Find Contours.
#     contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#
#     # Make sure there is a contour present and also its size is bigger than the noise threshold.
#     if contours and cv2.contourArea(max(contours, key = cv2.contourArea)) > noiseth:
#
#         c = max(contours, key = cv2.contourArea)
#         x2,y2,w,h = cv2.boundingRect(c)
#
#         # Get the area of the contour
#         area = cv2.contourArea(c)
#
#         # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
#         if x1 == 0 and y1 == 0:
#             x1,y1= x2,y2
#
#         else:
#             # Draw the line on the canvas
#             canvas = cv2.line(canvas, (x1,y1),(x2,y2), [255,0,0], 5)
#
#         # After the line is drawn the new points become the previous points.
#         x1,y1= x2,y2
#
#         # Now if the area is greater than the wiper threshold then set the clear variable to True and warn User.
#         if area > wiper_thresh:
#            cv2.putText(canvas,'Clearing Canvas',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5, cv2.LINE_AA)
#            clear = True
#
#     else:
#         # If there were no contours detected then make x1,y1 = 0
#         x1,y1 =0,0
#
#
#     # Now this piece of code is just for smooth drawing. (Optional)
#     _ ,mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
#     foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
#     background = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(mask))
#     frame = cv2.add(foreground,background)
#
#     cv2.imshow('image',frame)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
#     # Clear the canvas after 1 second if the clear variable is true
#     if clear == True:
#
#         time.sleep(1)
#         canvas = None
#
#         # And then set clear to false
#         clear = False
#
# cv2.destroyAllWindows()
# cap.release()

# step6
load_from_disk = True
if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load these 2 images and resize them to the same size.
pen_img = cv2.resize(cv2.imread('pen.png', 1), (50, 50))
eraser_img = cv2.resize(cv2.imread('eraser.jpg', 1), (50, 50))
cam1_img = cv2.resize(cv2.imread('camera1.png', 1), (50, 50))
cam2_img = cv2.resize(cv2.imread('camera2.png', 1), (50, 50))
change_color_img = cv2.resize(cv2.imread('change_color_img.png', 1), (50, 50))
font_size_img = cv2.resize(cv2.imread('font_size.png', 1), (50, 50))


kernel = np.ones((5, 5), np.uint8)

# Making window size adjustable
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# This is the canvas on which we will draw upon
canvas = None

# Create a background subtractor Object
backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

# This threshold determines the amount of disruption in background.
background_threshold = 400

# A variable which tells you if you're using a pen or an eraser.
switch = 'Pen'

# With this variable we will monitor the time between previous switch.
last_switch = time.time()

# Initilize x1,y1 points
x1, y1 = 0, 0

# Threshold for noise
noiseth = 800

# Threshold for wiper, the size of the contour must be bigger than this for us to clear the canvas
wiper_thresh = 40000

# A varaible which tells when to clear canvas
clear = False
cap1 = False
cap2 = False
change_color = False
change_font_size = False

font_color = [255, 0, 0]
font_size = 5
font_size_erase = 20

draw_delay = False
draw_chk = False

while (1):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Initilize the canvas as a black image
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Take the top left of the frame and apply the background subtractor there
    top_left = frame[0: 50, 0: 50]
    top_left_cap1 = frame[0: 50, 150: 200]
    top_left_cap2 = frame[0: 50, 300: 350]
    top_left_change_color = frame[0: 50, 450: 500]
    top_left_change_font_size = frame[0: 50, 600: 650]

    fgmask = backgroundobject.apply(top_left)
    fgmask_cap1 = backgroundobject.apply(top_left_cap1)
    fgmask_cap2 = backgroundobject.apply(top_left_cap2)
    fgmask_change_color = backgroundobject.apply(top_left_change_color)
    fgmask_change_font_size = backgroundobject.apply(top_left_change_font_size)

    # Note the number of pixels that  are white,this is the level of disruption.
    switch_thresh = np.sum(fgmask==255)
    cap1_thresh = np.sum(fgmask_cap1==255)
    cap2_thresh = np.sum(fgmask_cap2==255)
    change_color_thresh = np.sum(fgmask_change_color==255)
    font_size_thresh = np.sum(fgmask_change_font_size==255)

    # If the disruption is greater than background threshold and there has been some time after the previous switch then you
    # can change the object type.
    if switch_thresh > background_threshold and (time.time() - last_switch) > 1:

        # Save the time of the switch.
        last_switch = time.time()

        if switch == 'Pen':
            switch = 'Eraser'
        else:
            switch = 'Pen'

    if cap1_thresh > background_threshold and (time.time() - last_switch) > 1:
        cap1 = True

    if cap2_thresh > background_threshold and (time.time() - last_switch) > 1:
        cap2 = True

    if change_color_thresh > background_threshold and (time.time() - last_switch) > 1:
        change_color = True

    if font_size_thresh > background_threshold and (time.time() - last_switch) > 1:
        change_font_size = True


    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # If you're reading from memory then load the upper and lower ranges from there
    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]

    # Otherwise define your own custom values for upper and lower range.
    else:
        lower_range = np.array([30, 80, 110])
        #lower_range = np.array([55, 40, 0])
        upper_range = np.array([50, 200, 200])

    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Perform morphological operations to get rid of the noise
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find Contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make sure there is a contour present and also its size is bigger than noise threshold.
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noiseth:

        c = max(contours, key=cv2.contourArea)
        x2, y2, w, h = cv2.boundingRect(c)

        # Get the area of the contour
        area = cv2.contourArea(c)

        # If there were no previous points then save the detected x2,y2 coordinates as x1,y1.
        if x1 == 0 and y1 == 0:
            x1,y1= x2,y2

        else:

            if switch == 'Pen':
                # Draw the line on the canvas
                if draw_delay == True:
                    draw_delay = False

                    time.sleep(1)
                    x1, y1 = x2, y2
                if keyboard.is_pressed(' '):
                    canvas = cv2.line(canvas, (x1, y1), (x2, y2), font_color, font_size)

            else:
                cv2.circle(canvas, (x2, y2), font_size_erase, (0, 0, 0), -1)

        # After the line is drawn the new points become the previous points.
        x1, y1 = x2, y2

        # Now if the area is greater than the wiper threshold then set the clear variable to True
        if area > wiper_thresh:
            cv2.putText(canvas, 'Clearing Canvas', (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1, cv2.LINE_AA)
            clear = True


    else:
        # If there were no contours detected then make x1,y1 = 0
        x1, y1 = 0, 0

    # Now this piece of code is just for smooth drawing. (Optional)
    _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
    foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    frame = cv2.add(foreground, background)

    # Switch the images depending upon what we're using, pen or eraser.
    if switch != 'Pen':
        cv2.circle(frame, (x1, y1), font_size_erase, (255, 255, 255), -1)
        frame[0: 50, 0: 50] = eraser_img
    else:
        frame[0: 50, 0: 50] = pen_img

    frame[0: 50, 150: 200] = cam1_img
    frame[0: 50, 300: 350] = cam2_img
    frame[0: 50, 450: 500] = change_color_img
    frame[0: 50, 600: 650] = font_size_img

    cv2.imshow('image', frame)

    ## 디버깅 용도
    # Optionally stack both frames and show it.
    # stacked = np.hstack((canvas, frame))
    # cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    # #키보드 캡쳐부분
    # elif k == ord('s'):
    #     print("화면 캡쳐")
    #     # cv2.imwrite("F:\YOLO\newTech\fingerDraw" + "test" + ".png", frame)
    #     cv2.imwrite("Cam Test.png", frame)
    #
    # elif k == ord('d'):
    #     print("그림판 캡쳐")
    #     # cv2.imwrite("F:\YOLO\newTech\fingerDraw" + "test" + ".png", frame)
    #     cv2.imwrite("Paint Test.png", canvas)

    # Clear the canvas after 1 second, if the clear variable is true
    if clear == True:

        time.sleep(0.5)
        canvas = None

        # And then set clear to false
        clear = False

        draw_delay = True

    if cap1 == True:
        time.sleep(0.5)
        print("화면 캡쳐")
        cv2.imwrite("Cam Test.png", frame)
        cap1 = False

        draw_delay = True

    if cap2 == True:
        time.sleep(0.5)
        print("그림판 캡쳐")
        cv2.imwrite("Paint Test.png", canvas)
        cap2 = False

        draw_delay = True

    if change_color == True:
        change_color = False

        time.sleep(1)
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

        time.sleep(0.5)
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