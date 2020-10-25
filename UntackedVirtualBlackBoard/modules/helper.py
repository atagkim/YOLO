import cv2

def helper_func():
    image = cv2.imread("images/INFO.jpg", cv2.IMREAD_ANYCOLOR)
    cv2.imshow("Helper", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()