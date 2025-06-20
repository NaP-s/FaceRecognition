#include "Histogram.h"


Histogram::Histogram()
{
}

Histogram::Histogram(cv::Mat frame)
{
	if (frame.empty())
		return;
	switch (frame.channels()) {
	case 1:
		this->CreateHistogrammeNDG(frame);
		break;
	case 3:
		this->CreateHistogrammeCouleur(frame);
		break;
	default:
		break;
	}
}

Histogram::~Histogram()
{
}

void Histogram::CreateHistogrammeCouleur(cv::Mat frame)
{
	//Histogramme

	/// Separate the image in 3 places ( B, G and R )
	std::vector<cv::Mat> bgr_planes;
	cv::split(frame, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = true;

	cv::Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound(double(hist_w) / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	cv::normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());
	cv::normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());
	cv::normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			cv::Scalar(0, 255, 0), 2, 8, 0);
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			cv::Scalar(0, 0, 255), 2, 8, 0);
	}
	this->_graphHistogram = histImage;
}
void Histogram::CreateHistogrammeNDG(cv::Mat frame)
{
	//Histogramme
	/// Taile de l'histogramme
	int histSize = 256;

	/// Set the ranges)
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = true;

	cv::Mat ndg_hist;

	/// Compute the histograms:
	cv::calcHist(&frame, 1, 0, cv::Mat(), ndg_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	cv::normalize(ndg_hist, ndg_hist, 0, histImage.rows, NORM_MINMAX, -1, cv::Mat());


	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		cv::line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(ndg_hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(ndg_hist.at<float>(i))),
			cv::Scalar(255, 255, 255), 2, 8, 0);
	}
	this->_graphHistogram = histImage;
	int const size = histSize;
	//this->_matriceHistogram = new int[256];
	//for (int i = 0; i <= histSize-1; i++)
	//	this->_matriceHistogram[i] = cvRound(ndg_hist.at<float>(i));
	int i =0;
}

