// Libraries
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include "Traitements.h"

// Déclaration des namespace
using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay(Mat frame);

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

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


			// Si on a une image => Alors on detecte
			if (!frame.empty()){
				try
				{
					detectAndDisplay(frame);
				}
				catch (exception e)
				{
					//throw(e);
					continue;
				}
			}
			else{
				printf(" --(!) No captured frame -- Break!");
				break;
			}
			// On appuie sur c pour quitter
			int c = waitKey(1);
			if ((char)c == 'c') {
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

#pragma region Fonction detectAndDisplay - On lance la détection
/// <summary>Méthode de détection et d'affichage
/// <para>frame : Image d'entrée envoyé par la webcam</para>
/// </summary>
void detectAndDisplay(Mat frame){
	// Vecteurs de rectangle => Chaque rectangle correspond à l'emplacement d'un visage / yeux 
	std::vector<Rect> faces;
	std::vector<Rect> eyes;

	// Images de traitement ou d'affichage
	Mat imageCamNDG;
	Mat imageCouleurRedim;
	Mat imageNDG_Redim;
	Mat imageNDG_PourTraitement;
	Mat imageLBP;
	Mat imageHistogrammeLBP;

	// On définit des régions d'interet permettant d'isoler une partie de l'image et ainsi accelerer les temps de traitement
	Rect roi_b;
	Rect roi_c;

	// On convertit l'image de la webcam en Ndg puis on égalise son histogramme
	cvtColor(frame, imageCamNDG, COLOR_BGR2GRAY);
	equalizeHist(imageCamNDG, imageCamNDG);

	// Détection du visage : CV_HAAR_FIND_BIGGEST_OBJECT On cherche le plus gros objet ; Size(60, 60) => De taille minimum 60*60 pixels
	face_cascade.detectMultiScale(imageCamNDG, faces, 1.1, 4, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(60, 60));

	size_t ic = 0; // Index dans le tableau faces : Dans notre cas, on ne détecte qu'un seul visage
	if (faces.size() != 0){
		// On définit une région d'interet autour de notre visage
		roi_b.x = faces[ic].x;
		roi_b.y = faces[ic].y;
		roi_b.width = faces[ic].width;
		roi_b.height = faces[ic].height;

		// On créer une nouvelle image avec juste le visage en découpant une partie de l'image de la webCam
		imageCouleurRedim = frame(roi_b);
		resize(imageCouleurRedim, imageCouleurRedim, Size(256, 256), 0, 0, INTER_LINEAR);

		cvtColor(imageCouleurRedim, imageNDG_Redim, CV_BGR2GRAY); // On convertit notre image en NDG pour nos traitements

		// On lance la détection des yeux : CV_HAAR_SCALE_IMAGE On cherche plusieurs objets ; Size(15, 15) => De taille minimum 15*15 pixels
		// La position des yeux vas nous permettre de pouvoir redecouper notre image en etant resserré sur le visage. On ne voit donc plus le fond.
		// C'est cette image qui nous servira pour notre image LBP
		eyes_cascade.detectMultiScale(imageNDG_Redim, eyes, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(15, 15));
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
			imageNDG_PourTraitement = imageNDG_Redim(roi_c);
			resize(imageNDG_PourTraitement, imageNDG_PourTraitement, Size(128, 128), 0, 0, INTER_LINEAR);

			// On crée notre image LBP
			imageLBP = Traitements::LBP(imageNDG_PourTraitement);

			// On trace notre histogramme depuis notre image LBP
			imageHistogrammeLBP = Traitements::HistogrammeNDG(imageLBP);

		}

		// Dessin du visage détecté sur l'image principale
		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);

		putText(frame, "Visage Detecte ici", cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
	}

	// Affichage des differentes images
	imshow("WebCam", frame);
	if (!imageNDG_PourTraitement.empty())
	{
		imshow("imageLBP", imageLBP);
		imshow("imageNDG_PourTraitement", imageNDG_PourTraitement);
		imshow("imageHistogrammeLBP", imageHistogrammeLBP);

	}
	else{
		destroyWindow("imageLBP");
		destroyWindow("imageNDG_PourTraitement");
		destroyWindow("imageHistogrammeLBP");
	}
}
#pragma endregion




