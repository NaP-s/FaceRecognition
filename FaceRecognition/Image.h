#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Histogram.h"
using namespace cv;
using namespace std;


class Image
{
public:
	Image();
	Image(Image&);
	Image(Mat frame, bool convertToNdg, bool convertToNdgAndEqualizeHistogram, bool convertToLbp, bool createHistogramColor, bool createHistogramNdg, bool createHistogramLbp);
	~Image();

	static Mat ConvertToNdg(Mat frameColor, bool equalizeHistogram);
	Mat ConvertToLbp(Mat frameNdg) const;
	Mat CreateLbpImage(Mat frame) const;
	Mat Normalize(const Mat src) const;
	Mat PreprocessingWithTanTrigs(InputArray src, float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1, int sigma1 = 2) const;

	Mat get_frameCouleur() const
	{
		return (_frameCouleur);
	}
	Mat get_frameNdg() const
	{
		return (_frameNdg);
	}
	Mat get_frameLbp() const
	{
		return (_frameLbp);
	}
	Histogram get_frameHistogramColor() const
	{
		return (_histogramColor);
	}
	Histogram get_frameHistogramNdg() const
	{
		return (_histogramNdg);
	}
	Histogram get_frameHistogramLbp() const
	{
		return (_histogramLbp);
	}

private:
	Mat _frameCouleur;
	Mat _frameNdg;
	Mat _frameLbp;

	Histogram _histogramNdg;
	Histogram _histogramColor;
	Histogram _histogramLbp;
};


