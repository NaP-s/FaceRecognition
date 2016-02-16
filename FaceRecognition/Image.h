#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Histogram.h"
using namespace cv;
using namespace std;


class Image
{
	Mat m;
public:
	Image();
	Image(Image&);
	Image(Mat frame, bool convertToNdg, bool convertToNdgAndEqualizeHistogram, bool convertToLbp, bool createHistogramColor, bool createHistogramNdg, bool createHistogramLbp);
	~Image();

	static Mat ConvertToNdg(Mat frameColor, bool equalizeHistogram);
	static Mat ConvertToNdgFromNotColorImage(Mat frame, bool equalizeHistogram);
	Mat ConvertToLbp(Mat frameNdg) const;
	Mat CreateLbpImage(Mat frame) const;
	Mat Normalize(const Mat src) const;
	Mat PreprocessingWithTanTrigs(InputArray src, float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1, int sigma1 = 2) const;

	static Mat resize(Mat frame, Size size);
	Mat get_frameCouleur() 
	{
		return (_frameCouleur.empty() ? m : _frameCouleur);
	}
	Mat get_frameNdg() 
	{
		return (_frameNdg.empty() ? m : _frameNdg);
	}
	void set_frameNdg(Mat frameNdg)
	{
		this->_frameNdg = frameNdg;
	}
	Mat get_frameLbp() 
	{
		return (_frameLbp.empty() ? m : _frameLbp);
	}
	void set_frameLbp(Mat frameLbp)
	{
		this->_frameLbp = frameLbp;
	}
	Histogram get_frameHistogramColor() const
	{
		return (_histogramColor);
	}
	void set_histogramColor(Histogram histoColor)
	{
		this->_histogramColor = histoColor;
	}
	Histogram get_frameHistogramNdg() const
	{
		return (_histogramNdg);
	}
	void set_histogramNdg(Histogram histoNdg)
	{
		this->_histogramNdg = histoNdg;
	}
	Histogram get_frameHistogramLbp() const
	{
		return (_histogramLbp);
	}
	void set_histogramLbp(Histogram histoLbp)
	{
		this->_histogramLbp = histoLbp;
	}

private:
	Mat _frameCouleur;
	Mat _frameNdg;
	Mat _frameLbp;

	Histogram _histogramNdg;
	Histogram _histogramColor;
	Histogram _histogramLbp;
};


