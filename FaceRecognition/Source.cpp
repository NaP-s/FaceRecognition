// Libraries
//#include "opencv2/core/core.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
#include "Histogram.h"
#include "ImageIntegral.h"
#define _CRT_SECURE_NO_DEPRECATE

// Namespace declaration
using namespace std;
using namespace cv;


// Function Headers
void detectAndDisplay(Mat frame);
int comp(Mat a, Mat b, Mat c);
Mat CalculHistogrammeNdg(Mat);
Mat tan_triggs_preprocessing(InputArray src,float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1,int sigma1 = 2);
Mat norm_0_255(const Mat& src);


string name = "toto";
stringstream ss;
int i = 0;
Mat LBP(Mat img);
Mat imTestLucas;
Mat imTestJulien;

Mat dst;
Mat dst2;
Mat histo;
Mat histo2;

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;


//Fonction de test HISTO_ALL
//int test_histogram(int argc, char** argv)
//{
//	Mat i1, i2, i3, i4;
//	char* imageName1 = argv[1];
//	char* imageName2 = argv[2];
//	///// Load three images with different environment settings
//	//if (argc < 3)
//	//{
//	//	printf("** Error. Usage: ./compareHist_Demo <image_settings0> <image_setting1> <image_settings2>\n");
//	//	return -1;
//	//}
//
//	//cerr<< argv[1] <<":" << argv[2] << endl;
//	i1 = imread(imageName1, 1);
//	i2 = imread(imageName2, 1);
//	//cerr << i2.channels() << endl;
//	//i3 = imread( argv[3], 1 );
//	//i4 = imread( argv[4], 1 );
//
//	cvtColor(i1, i3, CV_BGR2HSV);
//	cvtColor(i2, i4, CV_BGR2HSV);
//	//cvtColor( i3,i3, CV_BGR2HSV );
//	//cvtColor( i4,i4, CV_BGR2HSV );
//
//	Histogram h1, h2, h3, h4;
//	vector<int> channel;
//	channel.push_back(0);
//	//channel.push_back(1);
//	//channel.push_back(2);
//	h1.setChannel(channel);
//	h2.setChannel(channel);
//	//h3.setChannel(channel);
//	//h4.setChannel(channel);
//	vector<int> size;
//	size.push_back(50);
//	size.push_back(60);
//	size.push_back(40);
//	h1.setHistSize(size);
//	h2.setHistSize(size);
//
//	h1.BuildHistogram(i3);
//	h2.BuildHistogram(i4);
//	//h3.BuildHistogram(i3);
//	//h4.BuildHistogram(i4);
//	Mat h11 = h1.drawHist();
//	imshow("Hist 1", h11);
//	h11 = h2.drawHist();
//	imshow("Hist 2", h11);
//
//	cv::waitKey(0);
//
//
//	double r1 = h1.compareHist(h2, Histogram::comparison_method::BHATTACHRYA);
//	double r2 = h1.compareHist(h2, Histogram::comparison_method::CHI_SQUARED);
//	double r3 = h1.compareHist(h2, Histogram::comparison_method::INTERSECTION);
//	double r4 = h1.compareHist(h2, Histogram::comparison_method::CORRELATION);
//
//	cerr << r1 << ":" << r2 << ":" << r3 << ":" << r4 << endl;
//
//	r1 = cv::compareHist(h1.getHist(), h2.getHist(), CV_COMP_BHATTACHARYYA);
//	r2 = cv::compareHist(h1.getHist(), h2.getHist(), CV_COMP_CHISQR);
//	r3 = cv::compareHist(h1.getHist(), h2.getHist(), CV_COMP_INTERSECT);
//	r4 = cv::compareHist(h1.getHist(), h2.getHist(), CV_COMP_CORREL);
//
//	cerr << r1 << ":" << r2 << ":" << r3 << ":" << r4 << endl;
//	cv::waitKey(0);
//}
//// Fin Fonction de test HISTO_ALL

// Function main
int main()
{
	// Start cvStartWindowThread to create a thread process. VERY IMPORTANT
	cvStartWindowThread();

	// Initializing local variables
	int k = 1;
	CvCapture* capture;
	Mat frame;

	//On charge les deux images à comparer
	
	// TODO ZARN 
	/*imTestLucas = imread("D:/Users/julien.zarniak/Documents/visual studio 2013/Projects/OpenCV_FaceDetection/Images/Lucas_crop.jpg", 1);
	imTestJulien = imread("D:/Users/julien.zarniak/Documents/visual studio 2013/Projects/OpenCV_FaceDetection/Images/Julien_crop.jpg", 1);
*/
	// Load the cascade, use ifs (if more than one xml files are used) to prevent segmentation fault
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading\n");
		return (-1);
	}
	if (!eyes_cascade.load(eyes_cascade_name)){
		printf("--(!)Error loading\n");
		return -1;
	};

	// Start the program, capture from CAM with CAMID =0    
	capture = cvCaptureFromCAM(0);
	if (capture != 0){
		while (k == 1){
			frame = cvQueryFrame(capture);
			cv::flip(frame, frame, 1);
			//-- 3. Apply the classifier to the frame
			if (!frame.empty()){
				detectAndDisplay(frame);
			}
			else{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			int c = waitKey(1);
			if ((char)c == 'c') {
				k = 0;
				destroyWindow("FYP Live Camera");
				break;
			}
			if ((char)c == 's') {
				if (!dst.empty())
				{
					i++;
					ss << "../cImages/" << i << ".jpg";
					imwrite(ss.str(), dst);
					ss.str("");
				}
			}
		}
	}
	else{
		printf("CvCaptureFromCAM ERROR\n");
	}
	cvReleaseCapture(&capture);
	return 0;


	
//	test_histogram( 1, "1.bmp");
	return 0;


}

// Function detectAndDisplay
void detectAndDisplay(Mat frame){
	std::vector<Rect> faces;
	std::vector<Rect> eyes;
	Mat frame_gray;
	Mat crop;
	Mat crop2;
	Mat res;
	Mat gray;
	Mat preprocessed;
	string text;
	stringstream sstm;


	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(60, 60));

	// Set Region of Interest
	cv::Rect roi_b;
	cv::Rect roi_c;

	size_t ic = 0; // ic is index of current element


	if (faces.size() != 0){
		for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
		{
			roi_b.x = faces[ic].x;
			roi_b.y = faces[ic].y;
			roi_b.width = faces[ic].width;
			roi_b.height = faces[ic].height;

			crop = frame(roi_b);
			
			resize(crop, res, Size(256, 256), 0, 0, INTER_LINEAR); // This will be needed later while saving images
			cvtColor(res, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

			eyes_cascade.detectMultiScale(gray, eyes, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(15, 15));
			if (eyes.size() == 2){
				if (eyes[0].x <= eyes[1].x){
					roi_c.x = eyes[0].x*0.75;
					roi_c.y = eyes[0].y*0.7;
					roi_c.width = (eyes[1].x + 65) - roi_c.x;
					roi_c.height = 190;
				}
				else if (eyes[0].x >= eyes[1].x) {
					roi_c.x = eyes[1].x*0.75;
					roi_c.y = eyes[1].y*0.7;
					roi_c.width = (eyes[0].x + 65) - roi_c.x;
					roi_c.height = 190;
				}
				crop2 = gray(roi_c);
 				preprocessed = tan_triggs_preprocessing(crop2, 0.1, 10.0, 0.2, 1, 2);
				//resize(crop2, crop2, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images
				//resize(preprocessed, preprocessed, Size(128, 128), 0, 0, INTER_LINEAR); // This will be needed later while saving images

				dst = LBP(preprocessed);

				int histSize = 256;
				float range[] = { 0, 256 };
				const float* histRange = { range };
				bool uniform = true; bool accumulate = true;
				Mat ndg_hist;
				calcHist(&frame, 1, 0, Mat(), ndg_hist, 1, &histSize, &histRange, uniform, accumulate);
				normalize(ndg_hist, ndg_hist, 0, 1, NORM_MINMAX, -1, Mat());
				

				for (int i = 0; i < 4; i++)
				{
					int compare_method = i;
					double h1_h2 = compareHist(ndg_hist, ndg_hist, compare_method);
					double h2_h2 = compareHist(ndg_hist, ndg_hist, compare_method);
					double h3_h2 = compareHist(ndg_hist, ndg_hist, compare_method);
					double h4_h2 = compareHist(ndg_hist, ndg_hist, compare_method);

					printf(" Method [%d] h1_h2, h2_h2, h3_h2, h4_h2 : %f, %f, %f, %f \n", i, h1_h2, h2_h2, h3_h2, h4_h2);
				}

				Point centerEye1(eyes[0].x + eyes[0].width*0.5, eyes[0].y + eyes[0].height*0.5);
				int radiusEye1 = cvRound((eyes[0].width + eyes[0].height)*0.25);
				circle(gray, centerEye1, radiusEye1, Scalar(0, 0, 255), 1, 8, 0);

				Point centerEye2(eyes[1].x + eyes[1].width*0.5, eyes[1].y + eyes[1].height*0.5);
				int radiusEye2 = cvRound((eyes[1].width + eyes[1].height)*0.25);
				circle(gray, centerEye2, radiusEye2, Scalar(0, 0, 255), 1, 8, 0);

			}

			Point pt1(faces[ic].x, faces[ic].y); // Display detected faces on main window - live stream from camera
			Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
			rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);

			putText(frame, "Auto-focused  " , cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
		}

	}

	imshow("Live Camera", frame);
	if (!crop2.empty())
	{
		imshow("Gray2", dst);
		//imshow("Gray3", crop2);
		imshow("Histo", histo);
		//imshow("Histo TanTriggs", histo2);
		imshow("TanTriggs Preprocessed Image", norm_0_255(preprocessed));
	
	}
	else{
		destroyWindow("Gray2");
		//destroyWindow("Gray3");
		destroyWindow("Histo");
		destroyWindow("TanTriggs Preprocessed Image");
	}
}
Mat LBP(Mat img){
	Mat dst = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			uchar center = img.at<uchar>(i, j);
			unsigned char code = 0;
			code |= ((img.at<uchar>(i - 1, j - 1)) > center) << 7;
			code |= ((img.at<uchar>(i - 1, j)) > center) << 6;
			code |= ((img.at<uchar>(i - 1, j + 1)) > center) << 5;
			code |= ((img.at<uchar>(i, j + 1)) > center) << 4;
			code |= ((img.at<uchar>(i + 1, j + 1)) > center) << 3;
			code |= ((img.at<uchar>(i + 1, j)) > center) << 2;
			code |= ((img.at<uchar>(i + 1, j - 1)) > center) << 1;
			code |= ((img.at<uchar>(i, j - 1)) > center) << 0;
			dst.at<uchar>(i - 1, j - 1) = code;
		}
	}
	histo = CalculHistogrammeNdg(dst);
	return dst;
}

// Gestion Histogramme
Mat CalculHistogrammeNdg(Mat frame)
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

	// Draw the histograms for B, G and R
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

// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat& src) {
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

//
// Calculates the TanTriggs Preprocessing as described in:
//
//      Tan, X., and Triggs, B. "Enhanced local texture feature sets for face
//      recognition under difficult lighting conditions.". IEEE Transactions
//      on Image Processing 19 (2010), 1635–650.
//
// Default parameters are taken from the paper.
//
Mat tan_triggs_preprocessing(InputArray src,
	float alpha, float tau , float gamma , int sigma0 ,
	int sigma1 ) {

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

//int main(int argc, const char *argv[]) {
//	// Get filename to the source image:
//	if (argc != 2) {
//		cout << "usage: " << argv[0] << " <image.ext>" << endl;
//		exit(1);
//	}
//	// Load image & get skin proportions:
//	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//	// Calculate the TanTriggs Preprocessed image with default parameters:
//	Mat preprocessed = tan_triggs_preprocessing(image);
//	// Draw it on screen:
//	imshow("Original Image", image);
//	imshow("TanTriggs Preprocessed Image", norm_0_255(preprocessed));
//	// Show the images:
//	waitKey(0);
//	// Success!
//	return 0;
//}

