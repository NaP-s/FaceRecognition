#include "Camera.h"
#include <windows.h>


Camera::Camera()
{
        _capture.open(1);
        if (!_capture.isOpened())  // Si NoK alors on se connecte a la webcam
        {
                _capture.open(0);
        }
VideoCapture Camera::GetCapture(){return (_capture);}
void Camera::SetCapture(VideoCapture capture){_capture = capture;}



Camera::~Camera(){}

CvCapture* Camera::GetCapture(){return (_capture);}
void Camera::SetCapture(CvCapture* capture){_capture = capture;}