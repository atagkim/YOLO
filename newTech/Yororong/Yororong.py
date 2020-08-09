import cv2
import numpy as np
import pyautogui

hand_hist = None
traverse_point = []
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

chkfst = True


def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def contours(hist_mask_image):
    gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #윤곽선 찾기(이미지, 윤곽선찾는방법(윤곽선 찾고 상하구조 구성Tree), 윤곽선 찾을때 사용하는 근사화 방법(윤곽선 그릴 수 있는 포인트만 반환))
    return cont

def draw_rect(frame):
    rows, cols, _ = frame.shape
    #프레임 크기 뽑음
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    #9개 짜리 네모, 프레임 비율에 맞춰서 위치조정
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    #9개 짜리 네모, 크기 조정(네모를 그릴 때 종료좌표를 10만큼 크게줘서 크기가 10으로 그려짐
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)
    #네모를 그려줌, (그릴 이미지, 시작좌표, 종료좌표, 색, 두께)

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #색상 변환
    #BGR -> HSV로 변환
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    #0으로 찬 배열 생성

    for i in range(total_rectangle):
    #total_rectangle은 처음 뜨는 9개 사각형
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
        #hand_rect_one는 위에

    #roi는 입력 이미지의 9개 점의 배열
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #히스토그램을 뽑는데 roi배열에서, 0번 1번 채널에서 입력받겠다는 의미, 마스크이미지, X축 요소의 개수, Y축 요소값의 범위
    #여기서 히스토그램의 X축은 픽셀값, Y축은 픽셀개수를 나타냄
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    #정규화, (입력 배열, 결과 배열, 알파, 베타, 기준)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # BGR -> HSV로 변환

    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    #backProjection : 사진이나 영화를 배경막이나 백드롭에 투사하는 것
    #(오브젝트를 찾을 이미지, backprojection계산에 사용할 채널 인덱스, 히스토그램, 채널별 히스토그램 범위, 사용하는 scale factor)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    #타원모양 배열 생성(직선형이라 십자모양이랑 비슷)
    cv2.filter2D(dst, -1, disc, dst)
    #convolution과 동일
    #(이미지, 이미지 깊이(입력과 동일하면 -1), 적용할 커널)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    #(인풋 이미지, 임계값, 임계값 이상이면 적용할 값, 타입)
    #임계값보다 크면 백(1), 작으면 흑(0)

    # thresh = cv2.dilate(thresh, None, iterations=5)

    thresh = cv2.merge((thresh, thresh, thresh))
    #이미지 채널 합치기

    return cv2.bitwise_and(frame, thresh)
    #이거 자체가 hand_hist에서 뽑힌 색 기준으로 원본이미지도 나눔
    #손 색이랑 똑같은 색 이미지에서 뽑아서 반환

def centroid(max_contour):
    moment = cv2.moments(max_contour)
    # 윤곽선 기준으로 무게중심 구하는 공식
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))
        #거리 계산

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def draw_circles(frame, traverse_point):
    if traverse_point is not None:
        for i in range(len(traverse_point)):
            cv2.circle(frame, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)


def manage_image_opr(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)
    # hist_masking함수로 ㄱ

    hist_mask_image = cv2.erode(hist_mask_image, None, iterations=2)
    hist_mask_image = cv2.dilate(hist_mask_image, None, iterations=2)
    #말그대로 그림 강화, 약화 -> 이유는 잘..ㅋㅋ

    contour_list = contours(hist_mask_image)
    #contours함수로 ㄱ
    max_cont = max(contour_list, key=cv2.contourArea)
    #면적이 젤 큰 애 뽑음

    cnt_centroid = centroid(max_cont)
    #centroid함수로 ㄱ
    cv2.circle(frame, cnt_centroid, 5, [255, 0, 255], -1)
    #중간 지점 원그려줌

    global chkfst

    if max_cont is not None:
        #윤곽선 기준으로 convexity defect찾는다.
        hull = cv2.convexHull(max_cont, returnPoints=False)
        defects = cv2.convexityDefects(max_cont, hull)
        far_point = farthest_point(defects, max_cont, cnt_centroid)
        #farthest_point 함수 ㄱ
        #무게중심이랑 가장 먼 지점 계산
        print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
        cv2.circle(frame, far_point, 5, [0, 0, 255], -1)
        #수정하면 그림 그리기 가능

        ##마우스 제어
        if chkfst == True :
            chkfst = False
            pyautogui.moveTo(far_point)
            pyautogui.mouseDown()
        pyautogui.moveTo(far_point)

        ##프레임 같이쓰면 이런식으로 정지가능인데 따로써서 정지안되는중
        # chkstop = cv2.waitKey(1)
        # if chkstop & 0xFF == ord('x'):
        #     pyautogui.mouseUp()

        if len(traverse_point) < 20:
            traverse_point.append(far_point)
        else:
            traverse_point.pop(0)
            traverse_point.append(far_point)

        draw_circles(frame, traverse_point)

def main():
    #hand_hist는 처음 네모칸에 들어간 손바닥 의미
    global hand_hist
    is_hand_hist_created = False
    capture = cv2.VideoCapture(0)
    #비디오 캡쳐 객체 생성, 캠이 여러개면 뒤에 숫자 바뀜

    while capture.isOpened():
    #정상적으로 열렸는지 확인 용도
        pressed_key = cv2.waitKey(1)
        # x ms동안 대기
        _, frame = capture.read()
        #_에는 카메라 상태 저장(정상이면 True), frame에는 현재 frame 저장
        frame = cv2.flip(frame, 1)
        #flip(1은 좌우반전, 0은 상하 반전)

        if pressed_key & 0xFF == ord('z'):
        #pressed_key의 마지막 8비트 뽑음 -> 그냥 z가 눌렸는지 확인
            is_hand_hist_created = True
            hand_hist = hand_histogram(frame)
            #hand_histogram함수로 ㄱ

        if is_hand_hist_created:
            manage_image_opr(frame, hand_hist)
            #manage_image_opr함수로 ㄱ

        else:
            frame = draw_rect(frame)

        cv2.imshow("Live Feed", rescale_frame(frame))
        #화면 표시해줌, 화면 크기 조정 함수


        if pressed_key == 27:
            break


    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()