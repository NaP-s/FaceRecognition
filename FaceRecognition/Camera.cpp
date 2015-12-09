#include "Camera.h"
#include <windows.h>


Camera::Camera()
{
	//On se connecte par d�faut � la cam�ra
	_capture = cvCaptureFromCAM(1);
	if (!_capture)	// Si NoK alors on se connecte � la webcam
	{
		_capture = cvCaptureFromCAM(0);
	}
}


Camera::~Camera(){}

CvCapture* Camera::GetCapture(){return (_capture);}
void Camera::SetCapture(CvCapture* capture){_capture = capture;}