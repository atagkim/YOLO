from tkinter import *
import pyautogui

# 변수 선언
window = None
canvas = None
x1, x2, y1, y2 = None, None, None, None  # 선이 그려지는 코드 추가

# 함수
# def mouseClick(event):
#     global x1, y1
#     x1 = event.x
#     y1 = event.y


# def mouseDrop(event):
#     global x2, y2
#     x2 = event.x
#     y2 = event.y
#     # 그림을 그리는 코드
#     canvas.create_line(x1, y1, x2, y2, width=5, fill="red")


# 드래그 함수
def mouseMove(event):
    global x1, y1
    x1 = event.x
    y1 = event.y
    canvas.create_oval(x1, y1, x1 + 1, y1 + 1, width=5, fill="red")
    print("{0},{1}".format(x1,y1))

## 메인 코드
window = Tk()
window.title("그림판 비슷한 프로그램")
canvas = Canvas(window, height=1000, width=1000)
# 버튼1왼쪽마우스, 버튼3오른쪽마우스, B1은드래그
#tx, ty = pyautogui.position()

canvas.bind("<B1-Motion>", mouseMove)
canvas.pack()
window.mainloop()