#include "Image.h"
#include "Traitements.h"


Image::Image()
{
}

Image::Image(Image& image)
{
	this->_frameCouleur = image._frameCouleur;
	this->_frameLbp = image._frameLbp;
	this->_frameNdg = image._frameNdg;
	this->_histogramNdg = image._histogramNdg;
	this->_histogramColor = image._histogramColor;
	this->_histogramLbp = image._histogramLbp;
}

Image::Image(Mat frame, bool convertToNdg = 0, bool convertToNdgAndEqualizeHistogram = 0, bool convertToLbp = 0, bool createHistogramColor = 0, bool createHistogramNdg = 0, bool createHistogramLbp = 0)
{
	this->_frameCouleur = frame;
	if (convertToNdg)
		this->_frameNdg = ConvertToNdg(this->_frameCouleur, convertToNdgAndEqualizeHistogram);
	if (convertToLbp && !this->_frameNdg.empty())
		this->_frameLbp = ConvertToLbp(this->_frameNdg);
	if (createHistogramNdg && !this->_frameNdg.empty())
		this->_histogramNdg = *(new Histogram(this->_frameNdg));
	if (createHistogramLbp)
		this->_histogramLbp = *(new Histogram(this->_frameLbp));
	if (createHistogramColor)
		this->_histogramColor = *(new Histogram(this->_frameCouleur));
}
Image::Image(Mat frameLbp)
{
	this->_frameLbp = frameLbp;
}

Image::~Image()
{
}

Mat Image::ConvertToNdg(Mat frameColor, bool equalizeHistogram)
{
	Mat frameNdg;
	cvtColor(frameColor, frameNdg, COLOR_BGR2GRAY);
	if (equalizeHistogram)
		equalizeHist(frameNdg, frameNdg);
	return (frameNdg);
}

Mat Image::ConvertToNdgFromNotColorImage(Mat frame, bool equalizeHistogram)
{
	Mat frameNdg;
	frame.convertTo(frameNdg, CV_8UC1);
	return (frameNdg);
}

Mat Image::ConvertToLbp(Mat frameNdg)
{
	Mat frameLbp;
	//frameLbp =  Traitements::ELBP(frameNdg,1,8);
	frameLbp = Traitements::LBP(frameNdg);

	return(frameLbp);
}



Mat Image::Normalize(Mat src) const
{
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}


Mat Image::resize(Mat frame, Size size)
{
	Mat rezized;
	cv::resize(frame, rezized, size, 0, 0, INTER_LINEAR);
	return(rezized);
}
