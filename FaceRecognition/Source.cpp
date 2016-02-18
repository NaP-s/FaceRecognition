// Libraries
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "Traitements.h"
#include"Image.h"

// Déclaration des namespace
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

int nImage = 1;
Mat image = imread("LBP_PP19.jpg", 1);

#pragma region Fonction main : On lance nos Threads
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
			imageCamera = new Image(frame, 1, 1, 0, 0, 0, 0);


			// Si on a une image => Alors on detecte
			if (!frame.empty()){
				try
				{
					detectAndDisplay();

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
					imwrite("LBP_PP" + std::to_string(nImage) + ".jpg", imagePourTraitementAvecPretraitement->get_frameLbp());
					nImage++;
				}
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

#pragma region Fonction detectAndDisplay - On lance la détection
/// <summary>Méthode de détection et d'affichage
/// <para>frame : Image d'entrée envoyé par la webcam</para>
/// </summary>
void detectAndDisplay(){
	// Vecteurs de rectangle => Chaque rectangle correspond à l'emplacement d'un visage / yeux 
	std::vector<Rect> faces;
	std::vector<Rect> eyes;


	// On définit des régions d'interet permettant d'isoler une partie de l'image et ainsi accelerer les temps de traitement
	Rect roi_b;
	Rect roi_c;

	// On convertit l'image de la webcam en Ndg puis on égalise son histogramme si nécessaire
	if (imageCamera->get_frameNdg().empty())
		imageCamera->set_frameNdg(imageCamera->ConvertToNdg(imageCamera->get_frameCouleur(), true));


	// Détection du visage : CV_HAAR_FIND_BIGGEST_OBJECT On cherche le plus gros objet ; Size(60, 60) => De taille minimum 60*60 pixels
	face_cascade.detectMultiScale(imageCamera->get_frameNdg(), faces, 1.1, 4, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(60, 60));

	size_t ic = 0; // Index dans le tableau faces : Dans notre cas, on ne détecte qu'un seul visage
	if (faces.size() != 0){
		// On définit une région d'interet autour de notre visage
		roi_b.x = faces[ic].x;
		roi_b.y = faces[ic].y;
		roi_b.width = faces[ic].width;
		roi_b.height = faces[ic].height;

		// On créer une nouvelle image avec juste le visage en découpant une partie de l'image de la webCam
		imageReduite = new Image( Image::resize(imageCamera->get_frameCouleur()(roi_b),Size(256,256)),1,0,1,0,0,0);
		//resize(imageReduite->get_frameCouleur(), imageReduite->get_frameCouleur(), Size(256, 256), 0, 0, INTER_LINEAR);

		// On lance la détection des yeux : CV_HAAR_SCALE_IMAGE On cherche plusieurs objets ; Size(15, 15) => De taille minimum 15*15 pixels
		// La position des yeux vas nous permettre de pouvoir redecouper notre image en etant resserré sur le visage. On ne voit donc plus le fond.
		// C'est cette image qui nous servira pour notre image LBP
		eyes_cascade.detectMultiScale(imageReduite->get_frameNdg(), eyes, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(15, 15));
		// Dans le cas ou on a bien détecter deux yeux
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
			imagePourTraitement = new Image(Image::resize(imageReduite->get_frameCouleur()(roi_c),Size(128, 128)), 1, 0, 1, 0, 0, 1);
			////resize(imagePourTraitement->get_frameNdg(), imagePourTraitement->get_frameNdg(), Size(128, 128), 0, 0, INTER_LINEAR);

			// On crée notre / nos images LBP
			imagePourTraitementAvecPretraitement = new Image(imagePourTraitement->get_frameCouleur(), 1, 0, 0, 0, 0, 0);
			imagePourTraitementAvecPretraitement->set_frameNdg(imagePourTraitement->Normalize(imagePourTraitement->PreprocessingWithTanTrigs(imagePourTraitement->get_frameNdg())));

			imagePourTraitementAvecPretraitement->set_frameLbp(imagePourTraitement->CreateLbpImage(imagePourTraitement->Normalize(imagePourTraitement->PreprocessingWithTanTrigs(imagePourTraitement->get_frameNdg()))));
			// On trace notre histogramme depuis notre image LBP
			imagePourTraitementAvecPretraitement->set_histogramLbp(*new Histogram(imagePourTraitementAvecPretraitement->get_frameLbp()));

		}

		// Dessin du visage détecté sur l'image principale
		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		// ReSharper disable once CppMsExtBindingRValueToLvalueReference
		rectangle(imageCamera->get_frameCouleur(), pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);

		// ReSharper disable CppMsExtBindingRValueToLvalueReference
		putText(imageCamera->get_frameCouleur(), "Visage Detecte ici", cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);

		/*	int histSize = 256;
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
				}*/



	}

	// Affichage des differentes images
	imshow("WebCam", imageCamera->get_frameCouleur());
	cvMoveWindow("WebCam", 0, 0);
	if (imagePourTraitementAvecPretraitement != NULL)
	{
		imshow("imageLBP", imagePourTraitement->get_frameLbp());
		imshow("imageLBPAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameLbp());
		imshow("imageNDG_PourTraitement", imagePourTraitement ->get_frameNdg());
		imshow("imageNDG_PourTraitementAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameNdg());
		imshow("imageHistogrammeLBP", imagePourTraitementAvecPretraitement->get_frameHistogramLbp().get_graphHistogram());
		imshow("imageHistogrammeLBPAvecPretraitement", imagePourTraitement->get_frameHistogramLbp().get_graphHistogram());
		cvMoveWindow("imageLBP", 1000, 0);
		cvMoveWindow("imageNDG_PourTraitement", 1200, 0);
		cvMoveWindow("imageHistogrammeLBP", 1400, 0);
		cvMoveWindow("imageLBPAvecPretraitement", 1000, 500);
		cvMoveWindow("imageNDG_PourTraitementAvecPretraitement", 1200, 500);
		cvMoveWindow("imageHistogrammeLBPAvecPretraitement", 1400, 500);

		imshow("imageRef", image);
		cvMoveWindow("imageRef", 0, 600);

	}
	else{
		destroyWindow("imageLBP");
		destroyWindow("imageLBPAvecPretraitement");
		destroyWindow("imageNDG_PourTraitement");
		destroyWindow("imageNDG_PourTraitementAvecPretraitement");
		destroyWindow("imageHistogrammeLBP");
		destroyWindow("imageHistogrammeLBPAvecPretraitement");
	}
}
#pragma endregion




