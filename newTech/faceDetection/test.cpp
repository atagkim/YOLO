#include <opencv2\core\core.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;

int main()
{
    Mat img = imread("image.jpg", IMREAD_COLOR);    // �̹��� ���� ������ ������ ���

    imshow("OpenCV_Test", img);        // ���α׷� ����â �̸� ����

    waitKey(0);        // Ű �Է� ���

    return 0;
}

