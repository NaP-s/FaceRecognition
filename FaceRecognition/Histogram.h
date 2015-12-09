#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Camera.h"
using namespace cv;
using namespace std;


class Histogram
{
public:
	Histogram();
	~Histogram();
	//Fonctions
	static void CalculAndDisplayHistogramme(Mat, String);
	static void CalculAndDisplayHistogrammeNdg(Mat frame, String windowsNames);
};

