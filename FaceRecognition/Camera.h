#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

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

