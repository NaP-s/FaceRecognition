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

	CvCapture* GetCapture();
	void SetCapture(CvCapture*);
private:
	CvCapture* _capture;

};

