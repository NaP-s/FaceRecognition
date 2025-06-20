#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class Camera
{
public:
	Camera();
	~Camera();

	VideoCapture GetCapture();
	void SetCapture(VideoCapture);
private:
	VideoCapture _capture;

};

