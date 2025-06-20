#include "Camera.h"
#include <windows.h>


Camera::Camera()
{
void Camera::SetCapture(CvCapture* capture){_capture = capture;}

	_capture = cvCaptureFromCAM(1);
	if (!_capture)	// Si NoK alors on se connecte ŕ la webcam
	{
		_capture = cvCaptureFromCAM(0);
	}
}


Camera::~Camera(){}

CvCapture* Camera::GetCapture(){return (_capture);}
void Camera::SetCapture(CvCapture* capture){_capture = capture;}