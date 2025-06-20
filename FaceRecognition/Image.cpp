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

Image::Image(cv::Mat frame, bool convertToNdg = 0, bool convertToNdgAndEqualizeHistogram = 0, bool convertToLbp = 0, bool createHistogramColor = 0, bool createHistogramNdg = 0, bool createHistogramLbp = 0)
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
Image::Image(cv::Mat frameLbp)
{
	this->_frameLbp = frameLbp;
}

Image::~Image()
{
}

cv::Mat Image::ConvertToNdg(cv::Mat frameColor, bool equalizeHistogram)
{
	cv::Mat frameNdg;
	cv::cvtColor(frameColor, frameNdg, COLOR_BGR2GRAY);
	if (equalizeHistogram)
	{
		cv::GaussianBlur(frameNdg, frameNdg, cv::Size(1, 1), 0, 0);
		//equalizeHist(frameNdg, frameNdg);
		//normalize(frameNdg, frameNdg, 0, 255, NORM_MINMAX, CV_8UC1);
	}
	return (frameNdg);
}

cv::Mat Image::ConvertToNdgFromNotColorImage(cv::Mat frame, bool equalizeHistogram)
{
	cv::Mat frameNdg;
	frame.convertTo(frameNdg, CV_8UC1);
	return (frameNdg);
}

cv::Mat Image::ConvertToLbp(cv::Mat frameNdg)
{
	cv::Mat frameLbp;
	//frameLbp =  Traitements::ELBP(frameNdg,1,4);
	frameLbp = Traitements::LBP(frameNdg);

	return(frameLbp);
}



cv::Mat Image::Normalize(cv::Mat src) const
{
        // Crée et renvoie une image normalisée :
	cv::Mat dst;
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


cv::Mat Image::resize(cv::Mat frame, cv::Size size)
{
	cv::Mat rezized;
	cv::resize(frame, rezized, size, 0, 0, INTER_LINEAR);
	return(rezized);
}

