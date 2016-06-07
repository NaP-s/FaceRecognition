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
void detectAndDisplay();
Mat ShowImageOverlay(Mat imageToDisplay);

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Image *imageCamera;
Image *imageReduite;
Image *imagePourTraitement;
Image *imagePourTraitementAvecPretraitement;

Image *imageRef;

Image *image;
int nImage = 1;
double h1_h2 = 0;
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


vector<double> ChiDeu(Mat img_VisageLBP1, Mat img_VisageLBP2, int splitX, int splitY)
{
	img_VisageLBP1.convertTo(img_VisageLBP1, CV_8UC1);
	img_VisageLBP2.convertTo(img_VisageLBP2, CV_8UC1);
	vector<double> score(64, 0);
	vector<int> mapPonderation{ 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 1, 1, 4, 4, 2, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 };
	int Nbcomp = splitY*splitX;

	int width = img_VisageLBP1.size().width;
	int height = img_VisageLBP1.size().height;
	int stepSize = img_VisageLBP1.rows / 8;

	int z = 0;
	int histSize = 255;
	float range[] = { 0, 255 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat a1_hist, a2_hist;

	//initialisation du tableau de scores (16 valeurs)

	for (int h = 0; h + stepSize < height; h += stepSize)
	{

		for (int g = 0; g + stepSize < width; g += stepSize)

		{
			// 1) on crée un rectangle qui va sélectionner la partie à découper
			// on le crée avec un point (x,y), une longueur, une largeur

			CvRect ROI = cvRect(h, g, stepSize, stepSize);
			Mat img_dest1 = img_VisageLBP1(ROI);
			Mat img_dest2 = img_VisageLBP2(ROI);
			/*namedWindow(std::to_string(z));
			imshow(std::to_string(z), img_dest1);
			*/


			/*cv::calcHist(&img_dest1, 1, 0, cv::Mat(), a1_hist, 1, &histSize, &histRange, uniform, accumulate);
			cv::calcHist(&img_dest2, 1, 0, cv::Mat(), a2_hist, 1, &histSize, &histRange, uniform, accumulate);*/

			//score[z] = cv::compareHist(a1_hist, a2_hist, CV_COMP_CHISQR);

			vector<int> hist1 = Traitements::CreateHistograme(img_dest1);

			vector<int> hist2 = Traitements::CreateHistograme(img_dest2);

			for (int nbBin = 0; nbBin <= 255; nbBin++)
			{
				if ((hist1[nbBin] + hist2[nbBin]) == 0)
					score[z] += 0;
				else
					score[z] += (((hist1[nbBin] - hist2[nbBin])*(hist1[nbBin] - hist2[nbBin])) / (hist1[nbBin] + hist2[nbBin])); // Calcul du Khi-deux et insertion dans un tableau scores : 16 valeurs à la fin
			}

			score[z] *= mapPonderation[z];
			z++;
		}

	}

	// Puis multiplication avec un tableau contenant le poids de chaque ROI pour avoir le scores final
	return score;

}



#pragma region Fonction main : On lance nos Threads

int main(){

	// Execution de cvStartWindowThread pour pouvoir créer des Threads
	cvStartWindowThread();

	// Initializing local variables
	int k = 1;
	CvCapture* capture;
	Mat frame;

	try
	{
		image1 = imread("COL_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cv::resize(image1, image1, Size(256, 256));
		imageRef = new Image(image1,0,0,0,0,0,0);
		imageRef->set_frameNdg(imageRef->get_frameCouleur());
		imageRef->set_frameLbp(imageRef->ConvertToLbp(imageRef->get_frameNdg()));
	}
	catch (Exception e)
	{
		printf("--(!)Error loading reference image\n");
	}
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
					imwrite("COL_" + std::to_string(nImage) + ".jpg", imagePourTraitementAvecPretraitement->get_frameCouleur());
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
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
		// On définit une région d'interet autour de notre visage
		roi_b.x = faces[ic].x;
		roi_b.y = faces[ic].y;
		roi_b.width = faces[ic].width;
		roi_b.height = faces[ic].height;

		// On créer une nouvelle image avec juste le visage en découpant une partie de l'image de la webCam
		imageReduite = new Image(Image::resize(imageCamera->get_frameCouleur()(roi_b), Size(256, 256)), 1, 0, 1, 0, 0, 0);

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

			// On crée notre / nos images LBP
			//imagePourTraitementAvecPretraitement = new Image(imagePourTraitement->get_frameCouleur(), 1, 0, 0, 0, 0, 0);
			imagePourTraitementAvecPretraitement = new Image(Image::resize(imageReduite->get_frameCouleur()(roi_c), Size(256, 256)), 1,1, 1, 0, 0, 0);
			//imagePourTraitementAvecPretraitement->set_frameNdg(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->PreprocessingWithTanTrigs(imagePourTraitementAvecPretraitement->get_frameNdg())));
			//imagePourTraitementAvecPretraitement->set_frameNdg(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->get_frameNdg()));

			//imagePourTraitementAvecPretraitement->set_frameLbp(imagePourTraitementAvecPretraitement->CreateLbpImage(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->PreprocessingWithTanTrigs(imagePourTraitementAvecPretraitement->get_frameNdg()))));
			
			//imagePourTraitementAvecPretraitement->set_frameLbp(imagePourTraitementAvecPretraitement->CreateLbpImage(imagePourTraitementAvecPretraitement->Normalize(imagePourTraitementAvecPretraitement->get_frameNdg())));

			// On trace notre histogramme depuis notre image LBP
			//imagePourTraitementAvecPretraitement->set_histogramLbp(*new Histogram(imagePourTraitementAvecPretraitement->get_frameLbp()));

		}

		// Dessin du visage détecté sur l'image principale
		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		// ReSharper disable once CppMsExtBindingRValueToLvalueReference
		rectangle(imageCamera->get_frameCouleur(), pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);
		if (imagePourTraitementAvecPretraitement != NULL && imageRef != NULL)
		{
			vector<double>scores = vector<double>(64);
			Mat imageDecale;
			//cv::resize(imagePourTraitementAvecPretraitement->get_frameLbp(), imageDecale, Size(250, 250));
			scores = ChiDeu(imagePourTraitementAvecPretraitement->get_frameLbp(), imageRef->get_frameLbp(), 8, 8);
			double scoreTotal = 0;
			for each (double score  in scores)
			{
				scoreTotal += score;
			}
			// ReSharper disable CppMsExtBindingRValueToLvalueReference
			putText(imageCamera->get_frameCouleur(), "Visage Detecte ici : Score " + std::to_string(scoreTotal), cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
		}

		// Affichage des differentes images
		imshow("WebCam", imageCamera->get_frameCouleur());
		imshow("WebCamNdg", imageCamera->get_frameNdg());
		cvMoveWindow("WebCam", 0, 0);
		if (imagePourTraitementAvecPretraitement != NULL)
		{
			//imshow("imageLBP", imagePourTraitement->get_frameLbp());
			imshow("imageLBPAvecPretraitement", ShowImageOverlay(imagePourTraitementAvecPretraitement->get_frameLbp()));
			//imshow("imageNDG_PourTraitement", imagePourTraitement->get_frameNdg());
			imshow("imageNDG_PourTraitementAvecPretraitement", ShowImageOverlay(imagePourTraitementAvecPretraitement->get_frameNdg()));
			//imshow("imageHistogrammeLBP", imagePourTraitement->get_frameHistogramLbp().get_graphHistogram());
			//imshow("imageHistogrammeLBPAvecPretraitement", imagePourTraitementAvecPretraitement->get_frameHistogramLbp().get_graphHistogram());
			//cvMoveWindow("imageLBP", 800, 0);
			//cvMoveWindow("imageNDG_PourTraitement", 1100, 0);
			//cvMoveWindow("imageHistogrammeLBP", 1400, 0);
			cvMoveWindow("imageLBPAvecPretraitement", 800, 500);
			cvMoveWindow("imageNDG_PourTraitementAvecPretraitement", 1100, 500);
			cvMoveWindow("imageHistogrammeLBPAvecPretraitement", 1400, 500);
			if (imageRef != NULL)
				imshow("imageRef", imageRef->get_frameLbp());
			cvMoveWindow("imageRef", 0, 600);

		}
		else{
			//destroyWindow("imageLBP");
			destroyWindow("imageLBPAvecPretraitement");
			//destroyWindow("imageNDG_PourTraitement");
			destroyWindow("imageNDG_PourTraitementAvecPretraitement");
			//destroyWindow("imageHistogrammeLBP");
			//destroyWindow("imageHistogrammeLBPAvecPretraitement");
		}
	}
#pragma endregion

}