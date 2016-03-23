#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class Histogram
{
public:
	Histogram();
	Histogram(Mat frame);
	~Histogram();

	void CreateHistogrammeCouleur(Mat frame);
	void CreateHistogrammeNDG(Mat frame);

	Mat get_graphHistogram() const
	{
		return (_graphHistogram);
	}
private:
	Mat _graphHistogram;
	int* _matriceHistogram;

};

