#include "Traitements.h"

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
	/// Return
	return histImage;
}
Mat Traitements::LBP(Mat img){
	Mat dst = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			uchar center = img.at<uchar>(i, j);
			unsigned char code = 0;
			bool b;
			uchar val;
			val = img.at<uchar>(i - 1, j - 1);
			val = img.at<uchar>(i - 1, j);
			val = img.at<uchar>(i - 1, j + 1);
			val = img.at<uchar>(i, j + 1);
			val = img.at<uchar>(i + 1, j + 1);
			val = img.at<uchar>(i + 1, j);
			val = img.at<uchar>(i + 1, j - 1);
			val = img.at<uchar>(i, j - 1);

			b = ((img.at<uchar>(i - 1, j - 1)) >= center) << 7;
			b = ((img.at<uchar>(i - 1, j)) >= center) << 6;
			b = ((img.at<uchar>(i - 1, j + 1)) >= center) << 5;
			b = ((img.at<uchar>(i, j + 1)) >= center) << 4;
			b =  ((img.at<uchar>(i + 1, j + 1)) > center) << 3;
			b = ((img.at<uchar>(i + 1, j)) >= center) << 2;
			b = ((img.at<uchar>(i + 1, j - 1)) >= center) << 1;
			b = ((img.at<uchar>(i, j - 1)) >= center) << 0;



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


// Comparaison d'histo
int comp(Mat a, Mat b, Mat c)
{
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1;
	Mat src_test2, hsv_test2;
	Mat hsv_half_down;

	src_base = a;
	src_test1 = b;
	src_test2 = c;

	/// Convert to HSV
	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(src_test2, hsv_test2, COLOR_BGR2HSV);

	hsv_half_down = hsv_base(Range(hsv_base.rows / 2, hsv_base.rows - 1), Range(0, hsv_base.cols - 1));

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };


	/// Histograms
	MatND hist_base;
	MatND hist_half_down;
	MatND hist_test1;
	MatND hist_test2;

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false);
	normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());

	/// Apply the histogram comparison methods
	for (int i = 0; i < 1; i++)
	{
		int compare_method = i;
		double base_base = compareHist(hist_base, hist_base, compare_method);
		double base_half = compareHist(hist_base, hist_half_down, compare_method);
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
		double base_test2 = compareHist(hist_base, hist_test2, compare_method);



		printf(" Method [%d] Perfect, Base-Half, Base-Test(1), Base-Test(2) : %f, %f, %f, %f \n", i, base_base, base_half, base_test1, base_test2);
	}

	printf("Done \n");

	return 0;
}
