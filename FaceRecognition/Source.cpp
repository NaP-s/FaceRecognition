// Libraries
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include "Traitements.h"
#include"Image.h"

#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>

// Déclaration des namespace
using namespace std;
using namespace cv;

// Function Headers
void Display();

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Image *imageCamera;

Mat image1 = imread("LBP_PP1.jpg", CV_LOAD_IMAGE_GRAYSCALE);

Mat ShowImageOverlay(Mat imageToDisplay)
{
	Mat mat_img(imageToDisplay);
	int stepSize = mat_img.rows / 8;

	int width = mat_img.size().width;
	int height = mat_img.size().height;

	for (int i = 0; i < height; i += stepSize)
		cv::line(mat_img, Point(0, i), Point(width, i), cv::Scalar(255, 0, 0));

	for (int i = 0; i < width; i += stepSize)
		cv::line(mat_img, Point(i, 0), Point(i, height), cv::Scalar(255, 0, 0));
	return mat_img;
}

#pragma region Fonction main : On lance nos Threads
void comparaison();

int main(){

	// Execution de cvStartWindowThread pour pouvoir créer des Threads
	cvStartWindowThread();

	// Initializing local variables
	int k = 1;
	CvCapture* capture;
	Mat frame;

	// Chargement des cascades de détection => Si on n'y arrive pas alors on ferme l'application
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading\n");
		return (-1);
	}
	if (!eyes_cascade.load(eyes_cascade_name)){
		printf("--(!)Error loading\n");
		return -1;
	};

	// On essaye de se connecter en priorité au lunette
	capture = cvCaptureFromCAM(1);
	if (!capture)	// Si NoK alors on se connecte à la webcam
	{
		capture = cvCaptureFromCAM(0);
	}



	if (capture != 0){
		while (k == 1){
			// On récupère une image depuis la caméra
			frame = cvQueryFrame(capture);
			cv::flip(frame, frame, 1);
			cv::resize(frame, frame, Size(256, 256));
			imageCamera = new Image(frame, 1, 1, 1, 1, 0, 0);


			// Si on a une image => Alors on detecte
			if (!frame.empty()){
				try
				{
					//detectAndDisplay();
					Display();
					//comparaison();

				}
				catch (exception e)
				{
					continue;
				}
			}
			else{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			// On appuie sur c pour quitter
			int c = waitKey(1);
			if (char(c) == 'c') {
				k = 0;
				destroyAllWindows();
				break;
			}
		}
	}
	else{
		printf("Erreur lors de la lecture du flux vidéo\n");
	}
	cvReleaseCapture(&capture);
	return 0;
}

#pragma endregion


void Display()
{
	// Affichage des differentes images
	imshow("WebCam", imageCamera->get_frameCouleur());
	imshow("WebCamNdg", imageCamera->get_frameNdg());
	cvMoveWindow("WebCam", 0, 0);
	Mat imgNdg;
	Traitements::ELBP(converr(imageCamera->get_frameCouleur(), imgNdg, CV_BGR2GRAY), 4, 16));
	imshow("WebcamLbp4_16", ;
	imshow("WebcamLbp1_8", Traitements::ELBP(imageCamera->get_frameNdg(), 1,8));
}