#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "Histogram.h"


class Image
{
	cv::Mat m;
public:
	Image();
	Image(Image&);
	Image(cv::Mat frame, bool convertToNdg, bool convertToNdgAndEqualizeHistogram, bool convertToLbp, bool createHistogramColor, bool createHistogramNdg, bool createHistogramLbp);
	Image(cv::Mat);
	~Image();

	static cv::Mat ConvertToNdg(cv::Mat frameColor, bool equalizeHistogram);
	static cv::Mat ConvertToNdgFromNotColorImage(cv::Mat frame, bool equalizeHistogram);
	cv::Mat ConvertToLbp(cv::Mat frameNdg);
	cv::Mat CreateLbpImage(cv::Mat frame) const;
	template <class _Tp>
	cv::Mat CreateLbpImageExtended(const cv::Mat& src, int radius, int neighbors);
	cv::Mat Normalize(const cv::Mat src) const;
	

	static cv::Mat resize(cv::Mat frame, cv::Size size);
	cv::Mat get_frameCouleur() 
	{
		return (_frameCouleur.empty() ? m : _frameCouleur);
	}
	cv::Mat get_frameNdg() 
	{
		return (_frameNdg.empty() ? m : _frameNdg);
	}
	void set_frameNdg(cv::Mat frameNdg)
	{
		this->_frameNdg = frameNdg;
	}
	cv::Mat get_frameLbp() 
	{
		return (_frameLbp.empty() ? m : _frameLbp);
	}
	void set_frameLbp(cv::Mat frameLbp)
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
	cv::Mat _frameCouleur;
	cv::Mat _frameNdg;
	cv::Mat _frameLbp;

	Histogram _histogramNdg;
	Histogram _histogramColor;
	Histogram _histogramLbp;
};


