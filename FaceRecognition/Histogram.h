#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

class Histogram
{
public:
	Histogram();
	Histogram(cv::Mat frame);
	~Histogram();

	void CreateHistogrammeCouleur(cv::Mat frame);
	void CreateHistogrammeNDG(cv::Mat frame);

	cv::Mat get_graphHistogram() const
	{
		return (_graphHistogram);
	}
private:
	cv::Mat _graphHistogram;
	int* _matriceHistogram;

};

