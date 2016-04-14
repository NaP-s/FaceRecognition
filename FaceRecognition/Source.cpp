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

// D�claration des namespace
using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay();

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Image *imageCamera;
Image *imageReduite;
Image *imagePourTraitement;
Image *imagePourTraitementAvecPretraitement;

Image *image;

int nImage = 1;
double h1_h2 = 0;
Mat image1 = imread("LBP_PP1.jpg", CV_LOAD_IMAGE_GRAYSCALE);




#pragma region Fonction main : On lance nos Threads
void comparaison();


int main(){
	// Execution de cvStartWindowThread pour pouvoir cr�er des Threads
	cvStartWindowThread();

	// Initializing local variables
	int k = 1;
	CvCapture* capture;
	Mat frame;

	cv::resize(image1, image1, Size(256, 256));


	// Chargement des cascades de d�tection => Si on n'y arrive pas alors on ferme l'application
	if (!face_cascade.load(face_cascade_name)){
		printf("--(!)Error loading\n");
		return (-1);
	}
	if (!eyes_cascade.load(eyes_cascade_name)){
		printf("--(!)Error loading\n");
		return -1;
	};

	// On essaye de se connecter en priorit� au lunette
	capture = cvCaptureFromCAM(1);
	if (!capture)	// Si NoK alors on se connecte � la webcam
	{
		capture = cvCaptureFromCAM(0);
	}



	if (capture != 0){
		while (k == 1){
			// On r�cup�re une image depuis la cam�ra
			frame = cvQueryFrame(capture);
			cv::flip(frame, frame, 1);
			imageCamera = new Image(frame, 1, 1, 0, 0, 0, 0);


			// Si on a une image => Alors on detecte
			if (!frame.empty()){
				try
				{
					detectAndDisplay();
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
			if (char(c) == 's') {
				if (imagePourTraitementAvecPretraitement != nullptr)
				{
					imwrite("NdgCharlot" + std::to_string(nImage) + ".jpg", imagePourTraitementAvecPretraitement->get_frameNdg());
					nImage++;
				}
			}
			if (char(c) == 'x') {
				if (imagePourTraitementAvecPretraitement != nullptr)
				{
					imwrite("NdgLio" + std::to_string(nImage) + ".jpg", imagePourTraitementAvecPretraitement->get_frameNdg());
					nImage++;
				}
			}
		}
	}
	else{
		printf("Erreur lors de la lecture du flux vid�o\n");
	}
	cvReleaseCapture(&capture);
	return 0;
}
#pragma endregion

#pragma region Fonction detectAndDisplay - On lance la d�tection
/// <summary>M�thode de d�tection et d'affichage
/// <para>frame : Image d'entr�e envoy� par la webcam</para>
/// </summary>
void detectAndDisplay(){
	// Vecteurs de rectangle => Chaque rectangle correspond � l'emplacement d'un visage / yeux 
	std::vector<Rect> faces;
	std::vector<Rect> eyes;


	// On d�finit des r�gions d'interet permettant d'isoler une partie de l'image et ainsi accelerer les temps de traitement
	Rect roi_b;
	Rect roi_c;

	// On convertit l'image de la webcam en Ndg puis on �galise son histogramme si n�cessaire
	if (imageCamera->get_frameNdg().empty())
		imageCamera->set_frameNdg(imageCamera->ConvertToNdg(imageCamera->get_frameCouleur(), true));


	// D�tection du visage : CV_HAAR_FIND_BIGGEST_OBJECT On cherche le plus gros objet ; Size(60, 60) => De taille minimum 60*60 pixels
	face_cascade.detectMultiScale(imageCamera->get_frameNdg(), faces, 1.1, 4, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(60, 60));

	size_t ic = 0; // Index dans le tableau faces : Dans notre cas, on ne d�tecte qu'un seul visage
	if (faces.size() != 0){
		// On d�finit une r�gion d'interet autour de notre visage
		roi_b.x = faces[ic].x;
		roi_b.y = faces[ic].y;
		roi_b.width = faces[ic].width;
		roi_b.height = faces[ic].height;

		// On cr�er une nouvelle image avec juste le visage en d�coupant une partie de l'image de la webCam
		imageReduite = new Image(Image::resize(imageCamera->get_frameCouleur()(roi_b), Size(256, 256)), 1, 0, 1, 0, 0, 0);

		// On lance la d�tection des yeux : CV_HAAR_SCALE_IMAGE On cherche plusieurs objets ; Size(15, 15) => De taille minimum 15*15 pixels
		// La position des yeux vas nous permettre de pouvoir redecouper notre image en etant resserr� sur le visage. On ne voit donc plus le fond.
		// C'est cette image qui nous servira pour notre image LBP
		eyes_cascade.detectMultiScale(imageReduite->get_frameNdg(), eyes, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(15, 15));
		// Dans le cas ou on a bien d�tecter deux yeux
		if (eyes.size() == 2){
			// Si le premier oeil du vecteur est l'oeil gauche
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
			//imagePourTraitement = new Image(Image::resize(imageReduite->get_frameCouleur()(roi_c), Size(256, 256)), 1, 0, 1, 0, 0, 1);

			// On cr�e notre / nos images LBP
			//imagePourTraitementAvecPretraitement = new Image(imagePourTraitement->get_frameCouleur(), 1, 0, 0, 0, 0, 0);
			imagePourTraitementAvecPretraitement = new Image(Image::resize(imageReduite->get_frameCouleur()(roi_c), Size(256, 256)), 1, 0, 0, 0, 0, 0);
			imagePourTraitementAvecPretraitement->set_frameNdg(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->PreprocessingWithTanTrigs(imagePourTraitementAvecPretraitement->get_frameNdg())));

			imagePourTraitementAvecPretraitement->set_frameLbp(imagePourTraitementAvecPretraitement->CreateLbpImage(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->PreprocessingWithTanTrigs(imagePourTraitementAvecPretraitement->get_frameNdg()))));
			// On trace notre histogramme depuis notre image LBP
			imagePourTraitementAvecPretraitement->set_histogramLbp(*new Histogram(imagePourTraitementAvecPretraitement->get_frameLbp()));

		}

		// Dessin du visage d�tect� sur l'image principale
		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		// ReSharper disable once CppMsExtBindingRValueToLvalueReference
		rectangle(imageCamera->get_frameCouleur(), pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);
		if (imagePourTraitementAvecPretraitement != NULL)
		{
			int histSize = 256;
			float range[] = { 0, 256 };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = true;
			Mat ndg_lbp;
			Mat ndg_lbpREf;


			calcHist(&imagePourTraitementAvecPretraitement->get_frameLbp(), 1, 0, Mat(), ndg_lbp, 1, &histSize, &histRange, uniform, accumulate);
			calcHist(&image1, 1, 0, Mat(), ndg_lbpREf, 1, &histSize, &histRange, uniform, accumulate);

			normalize(ndg_lbp, ndg_lbp, 0, 1, NORM_MINMAX, -1, Mat());
			normalize(ndg_lbpREf, ndg_lbpREf, 0, 1, NORM_MINMAX, -1, Mat());

			h1_h2 = compareHist(ndg_lbp, ndg_lbpREf, CV_COMP_CHISQR);
		}

		// ReSharper disable CppMsExtBindingRValueToLvalueReference
		putText(imageCamera->get_frameCouleur(), "Visage Detecte ici : Score " + std::to_string(h1_h2), cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	}

	// Affichage des differentes images
	imshow("WebCam", imageCamera->get_frameCouleur());
	cvMoveWindow("WebCam", 0, 0);
	if (imagePourTraitementAvecPretraitement != NULL)
	{
		//imshow("imageLBP", imagePourTraitement->get_frameLbp());
		imshow("imageLBPAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameLbp());
		//imshow("imageNDG_PourTraitement", imagePourTraitement->get_frameNdg());
		imshow("imageNDG_PourTraitementAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameNdg());
		//imshow("imageHistogrammeLBP", imagePourTraitement->get_frameHistogramLbp().get_graphHistogram());
		imshow("imageHistogrammeLBPAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameHistogramLbp().get_graphHistogram());
		//cvMoveWindow("imageLBP", 800, 0);
		//cvMoveWindow("imageNDG_PourTraitement", 1100, 0);
		//cvMoveWindow("imageHistogrammeLBP", 1400, 0);
		cvMoveWindow("imageLBPAvecPretraitement", 800, 500);
		cvMoveWindow("imageNDG_PourTraitementAvecPretraitement", 1100, 500);
		cvMoveWindow("imageHistogrammeLBPAvecPretraitement", 1400, 500);

		imshow("imageRef", image1);
		cvMoveWindow("imageRef", 0, 600);

	}
	else{
		//destroyWindow("imageLBP");
		destroyWindow("imageLBPAvecPretraitement");
		//destroyWindow("imageNDG_PourTraitement");
		destroyWindow("imageNDG_PourTraitementAvecPretraitement");
		//destroyWindow("imageHistogrammeLBP");
		destroyWindow("imageHistogrammeLBPAvecPretraitement");
	}
}
#pragma endregion


#pragma region Comparaison
void comparaison()
{
	Image image = Image(image1, 1, 1, 1, 0, 0, 1);
	//Image image1 = Image(image21, 1, 1, 1, 0, 0, 1);
	Mat imageMat;
	stringstream text1, text2, text3, fichier;
	int val = 0;
	int i = 0;
	int j = 0;
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = true;
	uchar intensity = 0;
	unsigned char *input = reinterpret_cast<unsigned char*>(&image.get_frameLbp());


	imshow("Image", image.get_frameNdg());
	imshow("ImageLbp", image.get_frameLbp());
	imshow("ImageHisto", image.get_frameHistogramLbp().get_graphHistogram());

	/*imshow("Image1", image1.get_frameNdg());
	imshow("ImageLbp1", image1.get_frameLbp());
	imshow("ImageHisto1", image1.get_frameHistogramLbp().get_graphHistogram());
	*/
	Mat HistoJu1;
	Mat HistoLu1;

	calcHist(&image.get_frameLbp(), 1, 0, Mat(), HistoJu1, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&image1.get_frameLbp(), 1, 0, Mat(), HistoLu1, 1, &histSize, &histRange, uniform, accumulate);

	double lu1_lu2 = compareHist(HistoLu1, HistoJu1, CV_COMP_CHISQR);

	imageMat = image.get_frameLbp();

	fichier << "Histo.txt";
	ofstream f(fichier.str().c_str());

	if (!f.is_open())
		std::cout << "Impossible d'ouvrir le fichier en ecriture !" << std::endl;
	else
	{
		int Histo[256] = { 0 };

		i = 0;
		j = 0;

		for (i = 0; i < imageMat.rows; i++)
			for (j = 0; j < imageMat.cols; j++)
			{
				intensity = imageMat.at<uchar>(i, j);
				Histo[intensity]++;
			}
		for (int k = 0; k < 256; k++)
		{
			f << Histo[k] << " ";

		}
		f << endl;


		/*
		Mat HistoJu1;
		Mat HistoJu2;
		Mat HistoJu3;
		Mat HistoJu4;

		Mat HistoLu1;
		Mat HistoLu2;
		Mat HistoLu3;
		Mat HistoLu4;

		Mat imageJu1 = imread("Julien/LBP_PP19.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageJu2 = imread("Julien/LBP_PP20.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageJu3 = imread("Julien/LBP_PP21.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageJu4 = imread("Julien/LBP_PP22.jpg", CV_LOAD_IMAGE_GRAYSCALE);


		Mat imageLu1 = imread("Lucas/LBP_PP1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageLu2 = imread("Lucas/LBP_PP2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageLu3 = imread("Lucas/LBP_PP3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		Mat imageLu4 = imread("Lucas/LBP_PP4.jpg", CV_LOAD_IMAGE_GRAYSCALE);

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true; bool accumulate = true;

		calcHist(&imageJu1, 1, 0, Mat(), HistoJu1, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageJu2, 1, 0, Mat(), HistoJu2, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageJu3, 1, 0, Mat(), HistoJu3, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageJu4, 1, 0, Mat(), HistoJu4, 1, &histSize, &histRange, uniform, accumulate);

		calcHist(&imageLu1, 1, 0, Mat(), HistoLu1, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageLu2, 1, 0, Mat(), HistoLu2, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageLu3, 1, 0, Mat(), HistoLu3, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&imageLu4, 1, 0, Mat(), HistoLu4, 1, &histSize, &histRange, uniform, accumulate);

		imshow("LBP_PP1_JU", imageJu1);
		imshow("LBP_PP2_JU", imageJu2);
		imshow("LBP_PP3JU", imageJu3);
		imshow("LBP_PP4_JU", imageJu4);
		imshow("LBP_PP1_LU", imageLu1);
		imshow("LBP_PP2_LU", imageLu2);
		imshow("LBP_PP3_LU", imageLu3);
		imshow("LBP_PP4_LU", imageLu4);

		double lu1_lu2 = compareHist(HistoLu1, HistoLu2, CV_COMP_CHISQR);
		double ju1_lu1 = compareHist(HistoJu1, HistoLu1, CV_COMP_CHISQR);
		double ju2_lu2 = compareHist(HistoJu2, HistoLu2, CV_COMP_CHISQR);

		Ptr<FaceRecognizer>  createLBPHFaceRecognizer(int radius = 1, int neighbors = 8, int grid_x = 8, int grid_y = 8, double threshold = DBL_MAX);*/
	}
	f.close();
}

#pragma endregion 

