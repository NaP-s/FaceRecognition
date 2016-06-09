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
	//frameNdg.convertTo(frameNdg, CV_8U, 0.1625);
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
	//frameLbp =  Traitements::PreprocessingWithTanTrigs(frameNdg);
	frameLbp = Traitements::LBP(frameNdg);

	return(frameLbp);
}


// TODO : Check Why dupliacet function
Mat Image::CreateLbpImage(Mat frame) const
{
	Mat dst = Mat::zeros(frame.rows - 2, frame.cols - 2, CV_8UC1);
	for (int i = 1; i < frame.rows - 1; i++) {
		for (int j = 1; j < frame.cols - 1; j++) {
			uchar center = frame.at<uchar>(i, j);
			unsigned char code = 0;
			code |= ((frame.at<uchar>(i - 1, j - 1)) > center) << 7;
			code |= ((frame.at<uchar>(i - 1, j)) > center) << 6;
			code |= ((frame.at<uchar>(i - 1, j + 1)) > center) << 5;
			code |= ((frame.at<uchar>(i, j + 1)) > center) << 4;
			code |= ((frame.at<uchar>(i + 1, j + 1)) > center) << 3;
			code |= ((frame.at<uchar>(i + 1, j)) > center) << 2;
			code |= ((frame.at<uchar>(i + 1, j - 1)) > center) << 1;
			code |= ((frame.at<uchar>(i, j - 1)) > center) << 0;
			dst.at<uchar>(i - 1, j - 1) = code;
		}
	}
	return dst;
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

Mat Image::PreprocessingWithTanTrigs(InputArray src, float alpha, float tau, float gamma, int sigma0, int sigma1) const
{

	// Convert to floating point:
	Mat X = src.getMat();
	X.convertTo(X, CV_32FC1);
	// Start preprocessing:
	Mat I;
	pow(X, gamma, I);
	// Calculate the DOG Image:
	{
		Mat gaussian0, gaussian1;
		// Kernel Size:
		int kernel_sz0 = (3 * sigma0);
		int kernel_sz1 = (3 * sigma1);
		// Make them odd for OpenCV:
		kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
		kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
		GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
		GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
		subtract(gaussian0, gaussian1, I);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(abs(I), alpha, tmp);
			meanI = mean(tmp).val[0];

		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp;
			pow(min(abs(I), tau), alpha, tmp);
			meanI = mean(tmp).val[0];
		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	// Squash into the tanh:
	{
		Mat exp_x, exp_negx;
		exp(I / tau, exp_x);
		exp(-I / tau, exp_negx);
		divide(exp_x - exp_negx, exp_x + exp_negx, I);
		I = tau * I;
	}
	return I;
}

Mat Image::resize(Mat frame, Size size)
{
	Mat rezized;
	cv::resize(frame, rezized, size, 0, 0, INTER_LINEAR);
	return(rezized);
}
