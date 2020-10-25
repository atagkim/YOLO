# Untacked Virtual BlackBoard
비대면 온라인 교육의 효과적인 교육을 위한 가상 칠판 서비스
<br>

# How it works
## 가상 칠판  
  - 색 설정  
    1. 프로그램 시작 시 보이는 HSV Trackbar를 조정하면서 터치펜으로 사용할 펜의 색을 뽑는다.  
    2. L-H는 HSV색의 색상 최소값, U-H는 HSV색의 색상 최대값, L-S는 HSV색의 채도 최소값, U-S는 HSV색의 채도 최대값, L-V는 HSV색의 명도 최소값, U-V는 HSV색의 명도 최대값이다.  
    3. HSV Trackbar를 조정하면서 컴퓨터가 인식하는 부분을 확인할 수 있는데, 좌측 캠화면의 흰색부분과 우측 캠화면의 본 색이 컴퓨터가 인식하는 부분임을 확인할 수 있다.  
    4. 인식이 제대로 된다고 생각되면 s키를 눌러 penval.py파일을 저장해준다.  
    
  - 가상 칠판 사용
    - 펜 기능
    1. 첫번째 - 펜/지우개 버튼에 터치펜을 가져다 대면 모드를 변경할 수 있다. 펜 모드일때는 스페이스바를 눌러 그림을 그릴 수 있다. 지우개 모드일때는 펜을 가져다 대면 그림을 지울 수 있다.
    2. 세번째 - 펜 색상 변경 버튼에 터치펜을 가져다 대면 색상을 변경할 수 있다. 펜 색상 선택은 좌측 가장자리에 나타나며, 총 5가지 색깔로 변경할 수 있다.
    3. 네번째 - 펜 굵기 설정 버튼에 터치펜을 가져다 대면 펜/지우기의 굵기를 변경할 수 있다. 펜 굵기 선택은 좌측 가장자리에 나타나며, 총 5가지 크기로 변경할 수 있다.
    4. 지정한 터치펜을 웹캠에 가까이 가져다 대면 현재까지 그린 그림들이 모두 지워진다. 
    
    - 화면 기능
    1. 잘라내려고 하는 캔버스의 좌측 상단 좌표를 인식해주고 x버튼, 우측 하단 좌표를 인식해주고 c버튼을 누르면 그림이 잘라내지고, 잘라내진 그림이 표시된다. 이것을 옮기고 싶은 위치로 인식시킨 후 v버튼을 누르면 그림이 다시 고정된다.
    2. 확대하고자 하는 캔버스의 좌측 상단 좌표를 인식하고 a버튼, 우측 하단 좌표를 인식해주고 s버튼을 누른 후, 확대를 원하면 d버튼, 축소를 원하면 f 버튼을 누르면 확대/축소된 화면이 캔버스에 옮겨준다.
    3. 두번째 - 캔버스 캡처 버튼에 손을 가져다 대면 프로그램 하위 폴더에 있는 images 폴더에 현재 캔버스의 화면이 png 파일로 저장된다.
    4. Windows 10 환경에서 현재 클립보드에 이미지가 존재할 시, o버튼을 눌러서 현재 클립보드에 존재하는 이미지를 캔버스로 불러온다.

## 3D 기능
   1. 5번째 버튼 - 3D 도형 버튼에 터치펜을 가져다 대면 도형을 회전시킬 수 있는 새로운 창이 등장한다.
   2. 1번 혹은 2번 버튼을 입력해서 각각 정육면체, 사각뿔 도형을 로드할 수 있고, 키보드 방향키와 q, w 버튼을 이용해 원하는 만큼 도형을 회전시킬 수 있다.
   3. s버튼을 누르면 그 도형의 모양을 캔버스로 로드한다.

## 집중도 확인  
  - 선생 프로그램
  1. 학생 프로그램에서 집중을 하지 않은 학생의 이름을 확인한다.
  - 학생 프로그램
  1. 프로그램 실핼 후 1분 이상 화면에서 학생의 얼굴이 인식되지 않을 경우 해당 학생의 프로그램에 알람이 팝업된다.
  - 서버 프로그램
  1. 처음에 학생 프로그램과 선생 프로그램을 연결한다. 이후에 학생이 집중하지 않고 있다고 판단될 경우, 서버 프로그램은 학생 프로그램에게서 해당 학생의 이름을 수신한다. 선생 프로그램은 해당 학생의 이름을 서버로부터 받아서 선생 프로그램의 화면에 팝업한다.
