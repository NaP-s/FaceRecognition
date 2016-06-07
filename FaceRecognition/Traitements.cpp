#include "Traitements.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14
#endif 
// Gestion Histogramme

Mat Traitements::HistogrammeCouleur(Mat frame)
{
	//Histogramme

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(frame, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = true;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound(double(hist_w) / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	return(histImage);
}
Mat Traitements::HistogrammeNDG(Mat frame)
{
	//Histogramme
	/// Taile de l'histogramme
	int histSize = 256;

	/// Set the ranges)
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = true;

	Mat ndg_hist;

	/// Compute the histograms:
	calcHist(&frame, 1, 0, Mat(), ndg_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(ndg_hist, ndg_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());


	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(ndg_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(ndg_hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}
	//Mat mat =
	/// Return
	return histImage;
}
Mat Traitements::LBP(Mat img){
	Mat dst = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			uchar center = img.at<uchar>(i, j);
			unsigned char code = 0;
			code |= ((img.at<uchar>(i - 1, j - 1)) >= center) << 7;
			code |= ((img.at<uchar>(i - 1, j)) >= center) << 6;
			code |= ((img.at<uchar>(i - 1, j + 1)) >= center) << 5;
			code |= ((img.at<uchar>(i, j + 1)) >= center) << 4;
			code |= ((img.at<uchar>(i + 1, j + 1)) >= center) << 3;
			code |= ((img.at<uchar>(i + 1, j)) >= center) << 2;
			code |= ((img.at<uchar>(i + 1, j - 1)) >= center) << 1;
			code |= ((img.at<uchar>(i, j - 1)) >= center) << 0;
			dst.at<uchar>(i - 1, j - 1) = code;
		}
	}
	return dst;
}
Mat Traitements::ELBP(const Mat& src, int radius, int neighbors) {
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
				float t = w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) && (abs(t - src.at<uchar>(i, j)) > std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	return dst;
}


vector<int> Traitements::CreateHistograme(Mat image)
{
	std::vector<int> vector1(256, 0);
	int with = image.size().width;
	int height = image.size().height;
	int temp;
	for (int i = 0; i < with ; i++)
		for (int j = 0; j < height; j++)
		{
			temp = image.at<uchar>(i, j);
			vector1.at(image.at<uchar>(i, j)) += 1;
		}
	return vector1;
}


Mat Traitements::PreprocessingWithTanTrigs(InputArray src, float alpha, float tau, float gamma, int sigma0, int sigma1) 
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
