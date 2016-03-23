#include "Image.h"


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
	frameLbp = CreateLbpImage(frameNdg);
	
	return(frameLbp);
}

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

long double M_PI = 3.14159265359;
template <typename _Tp>
Mat Image::CreateLbpImageExtended(const Mat& src, int radius, int neighbors) {
	neighbors = max(min(neighbors, 31), 1); // set bounds...
	// Note: alternatively you can switch to the new OpenCV Mat_
	// type system to define an unsigned int matrix... I am probably
	// mistaken here, but I didn't see an unsigned int representation
	// in OpenCV's classic typesystem...
	Mat dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius)* cos(2.0*M_PI*n / static_cast<float>(neighbors));
		float y = static_cast<float>(radius)* -sin(2.0*M_PI*n / static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				float t = w1*src.at<_Tp>(i + fy, j + fx) + w2*src.at<_Tp>(i + fy, j + cx) + w3*src.at<_Tp>(i + cy, j + fx) + w4*src.at<_Tp>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) && (abs(t - src.at<_Tp>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
			}
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
	cv::resize(frame,rezized, size, 0, 0, INTER_LINEAR);
	return(rezized);
}
