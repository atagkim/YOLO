import cv2 as cv
import numpy as np
import os


def detect(img, cascade):

    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def removeFaceAra(img, cascade):

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    rects = detect(gray, cascade)

    height, width = img.shape[:2]

    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    return img


def make_mask_image(img_bgr):
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # img_h,img_s,img_v = cv.split(img_hsv)

    low = (0, 30, 0)
    high = (15, 255, 255)

    img_mask = cv.inRange(img_hsv, low, high)
    return img_mask


def distanceBetweenTwoPoints(start, end):
    x1, y1 = start
    x2, y2 = end

    return int(np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)))


def calculateAngle(A, B):
    A_norm = np.linalg.norm(A)
    B_norm = np.linalg.norm(B)
    C = np.dot(A, B)

    angle = np.arccos(C / (A_norm * B_norm)) * 180 / np.pi
    return angle


def findMaxArea(contours):
    max_contour = None
    max_area = -1

    # 가장 큰 애를 찾기위해 포문 돌면서 계속 갱신
    for contour in contours:
        area = cv.contourArea(contour)

        # boundingRect는 주어진 contour의 외접하는 사각형 얻는것
        x, y, w, h = cv.boundingRect(contour)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = contour

    if max_area < 10000:
        max_area = -1

    return max_area, max_contour


def getFingerPosition(max_contour, img_result, debug):
    points1 = []

    # STEP 6-1
    M = cv.moments(max_contour)

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    max_contour = cv.approxPolyDP(max_contour, 0.02 * cv.arcLength(max_contour, True), True)
    hull = cv.convexHull(max_contour)

    for point in hull:
        if cy > point[0][1]:
            points1.append(tuple(point[0]))

    if debug:
        cv.drawContours(img_result, [hull], 0, (0, 255, 0), 2)
        for point in points1:
            cv.circle(img_result, tuple(point), 15, [0, 0, 0], -1)

    # STEP 6-2
    hull = cv.convexHull(max_contour, returnPoints=False)
    defects = cv.convexityDefects(max_contour, hull)

    if defects is None:
        return -1, None

    points2 = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])

        angle = calculateAngle(np.array(start) - np.array(far), np.array(end) - np.array(far))

        if angle < 90:
            if start[1] < cy:
                points2.append(start)

            if end[1] < cy:
                points2.append(end)

    if debug:
        cv.drawContours(img_result, [max_contour], 0, (255, 0, 255), 2)
        for point in points2:
            cv.circle(img_result, tuple(point), 20, [0, 255, 0], 5)

    # STEP 6-3
    points = points1 + points2
    points = list(set(points))

    # STEP 6-4
    new_points = []
    for p0 in points:

        i = -1
        for index, c0 in enumerate(max_contour):
            c0 = tuple(c0[0])

            if p0 == c0 or distanceBetweenTwoPoints(p0, c0) < 20:
                i = index
                break

        if i >= 0:
            pre = i - 1
            if pre < 0:
                pre = max_contour[len(max_contour) - 1][0]
            else:
                pre = max_contour[i - 1][0]

            next = i + 1
            if next > len(max_contour) - 1:
                next = max_contour[0][0]
            else:
                next = max_contour[i + 1][0]

            if isinstance(pre, np.ndarray):
                pre = tuple(pre.tolist())
            if isinstance(next, np.ndarray):
                next = tuple(next.tolist())

            angle = calculateAngle(np.array(pre) - np.array(p0), np.array(next) - np.array(p0))

            if angle < 90:
                new_points.append(p0)

    return 1, new_points


def process(img_bgr, debug):

    # 프로세스 과정을 통해 만들 이미지 결과물 선언
    img_result = img_bgr.copy()


    # STEP 1: 얼굴인식을 통해 얼굴파트 지우기 => 이를 통해 손가락 인식할때 얼굴색으로 인한 탐지실패를 막음
    img_bgr = removeFaceAra(img_bgr, cascade)


    # STEP 2: ??
    img_binary = make_mask_image(img_bgr)


    # STEP 3: 흑백화면으로 현재 인식되는 이미지 보여주기
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_binary = cv.morphologyEx(img_binary, cv.MORPH_CLOSE, kernel, 1)
    cv.imshow("Binary", img_binary)


    # STEP 4: contour 찾는 과정
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


    # 걍 디버그용
    if debug:
        for cnt in contours:
            cv.drawContours(img_result, [cnt], 0, (255, 0, 0), 3)


    # STEP  5: 찾은 contour 기반으로 맥스 에어리어 설정
    max_area, max_contour =findMaxArea(contours)

    # 맥스에어리어 안잡힐시 아웃
    if max_area == -1:
        return img_result

    # 걍 디버그용
    if debug:
        cv.drawContours(img_result, [max_contour], 0, (0, 0, 255), 3)


    # STEP 6: 손가락 위치잡기
    ret, points = getFingerPosition(max_contour, img_result, debug)


    # STEP 7: 찾은 손꾸락 동구라미 치기
    if ret > 0 and len(points) > 0:
        for point in points:
            cv.circle(img_result, point, 20, [255, 0, 255], 5)

    return img_result



# main

# 걍 디렉토리 경로 뽑는거임 아래에있는 얼굴인식 모델 접근할때 쓰기위한거
current_file_path = os.path.dirname(os.path.realpath(__file__))
# 얼굴인식 모델 로딩
cascade = cv.CascadeClassifier(cv.samples.findFile(current_file_path + "\haarcascade_frontalface_alt3.xml"))


# cap은 이미지 buf라고 보면됨. videocapture안의 변수가 0이면 캠으로, 경로면 영상이.
cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('test.avi')


while True:

    # cap을 통해 현재 이미지를 읽음
    ret, img_bgr = cap.read()

    # 이미지 못 읽을시 아웃
    if ret == False:
        break

    # 이미지를 가지고 정의된 process 함수 실행
    img_result = process(img_bgr, debug=False)

    # esc누르면 반복문 밖으로 나와요!
    key = cv.waitKey(1)
    if key == 27:
        break

    # 결과 보여주기
    cv.imshow("Result", img_result)


# free 과정
cap.release()
cv.destroyAllWindows()