#include <opencv2/imgproc/imgproc.hpp> // Gaussian Blur 
#include <opencv2/core/core.hpp>  // Basic OpenCV structures (cv::Mat, Scalar) 
#include <opencv2/highgui/highgui.hpp> // OpenCV window I/O 
#include <opencv2/features2d/features2d.hpp> 
#include <opencv2/objdetect/objdetect.hpp> 

#include <stdio.h> 
#include <string> 
#include <vector> 
#include <iostream> 

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

    //load image, in this case it's allready gray 
    Mat img = imread("input1.jpg");

    Mat grayImg;
    cvtColor(img, grayImg, CV_BGR2GRAY);

    //create vector of rectangles that will represent the faces 
    vector<Rect> faces;

    CascadeClassifier* faseCascade = new CascadeClassifier("C:\\YOLO\\newTech\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml");

    faseCascade->detectMultiScale(grayImg, faces);

    //draw rectangle on img; param: image, rectangle, color 
    cv::rectangle(img, faces[0], Scalar(255, 0, 0), 2);

    //display image 
    imshow("image", img);

    waitKey(0);

    return 0;
}