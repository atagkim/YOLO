#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;

int main()
{
    Mat img = imread("image.jpg", IMREAD_COLOR);    // 이미지 파일 본래의 색으로 출력

    imshow("OpenCV_Test", img);        // 프로그램 실행창 이름 설정

    waitKey(0);        // 키 입력 대기

    return 0;
}

