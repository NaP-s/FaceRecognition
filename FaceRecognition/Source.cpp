// Libraries
#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include "Traitements.h"
#include "Image.h"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <Windows.h>




// Déclaration des namespace
using namespace std;
using namespace cv;

// Function Headers
void detectAndDisplay();
Mat ShowImageOverlay(Mat imageToDisplay);
Mat ShowImagePonderation(Mat imageToDisplay);

// Global variables
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Image *imageCamera;
Image *imageReduite;
Image *imagePourTraitement;

struct Person
{
    std::string name;
    std::string surname;
    std::string birthdate;
    std::string age;
    std::string company;
    Image img;
};

std::vector<Person> referencePersons;

int nImage = 1;

String detectedPerson_Name;
String detectedPerson_Surname;
String detectedPerson_Age;
String detectedPerson_Birthhdate;
String detectedPerson_Company;


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

Mat ShowImagePonderation(Mat imageToDisplay)
{
	vector<int> mapPonderation
	{
		0, 1, 1, 0, 0, 1, 1, 0,
		2, 4, 4, 1, 1, 4, 4, 2,
		1, 3, 3, 0, 0, 3, 3, 1,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 1, 2, 2, 1, 0, 0,
		0, 1, 2, 3, 3, 2, 1, 0,
		0, 1, 2, 3, 3, 2, 1, 0,
		0, 0, 0, 1, 1, 0, 0, 0
	};
	Mat mat_img(imageToDisplay);
	int stepSize = mat_img.rows / 8;

	int width = mat_img.size().width;
	int height = mat_img.size().height;
	int z = 0;

	//initialisation du tableau de scores (16 valeurs)

	for (int h = stepSize; h <= height; h += stepSize)
	{

		for (int g = stepSize; g  <= width; g += stepSize)

		{
			switch (mapPonderation[z])
			{
			case 0:
				for (int i = h - stepSize; i < h; i += 1)
					for (int j = g - stepSize; j < g; j += 1)
						mat_img.at<uchar>(i,j) = 0;
				break;
			case 1:
				for (int i = h - stepSize; i < h; i += 1)
					for (int j = g - stepSize; j < g; j += 1)
						mat_img.at<uchar>(i, j) = 40;
				break;
			case 2:
				for (int i = h - stepSize; i < h; i += 1)
					for (int j = g - stepSize; j < g; j += 1)
						mat_img.at<uchar>(i, j) = 100;
				break;
			case 3:
				for (int i = h - stepSize; i < h; i += 1)
					for (int j = g - stepSize; j < g; j += 1)
						mat_img.at<uchar>(i, j) = 150;
				break;
			case 4:
				for (int i = h - stepSize; i < h; i += 1)
					for (int j = g - stepSize; j < g; j += 1)
						mat_img.at<uchar>(i, j) = 255;
				break;
			default:
				break;
			}
			z++;
		}
	}
	return (mat_img);
}


vector<double> ChiDeu(Mat img_VisageLBP1, Mat img_VisageLBP2, int splitX, int splitY)
{
	img_VisageLBP1.convertTo(img_VisageLBP1, CV_8UC1);
	img_VisageLBP2.convertTo(img_VisageLBP2, CV_8UC1);
	vector<double> score(64, 0);
	vector<int> mapPonderation
	{
					0, 1, 1, 0, 0, 1, 1, 0,
					2, 4, 4, 1, 1, 4, 4, 2,
					1, 3, 3, 0, 0, 3, 3, 1,
					0, 1, 1, 1, 1, 1, 1, 0,
					0, 0, 1, 2, 2, 1, 0, 0,
					0, 1, 2, 3, 3, 2, 1, 0,
					0, 1, 2, 3, 3, 2, 1, 0,
					0, 0, 0, 1, 1, 0, 0, 0 
	};
	int Nbcomp = splitY*splitX;

	int width = img_VisageLBP1.size().width;
	int height = img_VisageLBP1.size().height;
	int stepSize = img_VisageLBP1.rows / 8;

	int z = 0;

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

	// Show image Ponderation
	imshow("ImagePonderation", ShowImageOverlay(ShowImagePonderation(Mat(255, 255, CV_8UC1))));

        try
        {
                struct RefInfo { const char* path; const char* name; const char* surname; const char* birth; const char* age; const char* company; };
                std::vector<RefInfo> infos = {
                        {"JZK\\crop_3.jpg", "ZARNIAK", "Julien", "27 juillet", "23 ans", "ACTEMIUM"},
                        {"LVT\\crop_1.jpg", "VEROT", "Lionel", "17 aout", "23 ans", "VALEO"},
                        {"LVE\\crop_1.jpg", "VOLAINE", "Lucas", "08 fevrier", "26 ans", "AREVA"},
                        {"CEVT\\crop_1.jpg", "VLIMANT", "Charles Etienne", "18 mars", "25 ans", "THALES"},
                        {"SMN\\crop_1.jpg", "MARTIN", "Sylvain", "21 novembre", "23 ans", "Tri qualite service"},
                        {"FGE\\crop_1.jpg", "GIRE", "Florian", "24 mars", "23 ans", "SCHNEIDER"},
                        {"MSR\\crop_1.jpg", "SAMOUILLER", "Martin", "16 mai", "23 ans", "CYXPLUS"},
                        {"GCD\\crop_1.jpg", "CHAMBOND", "Gregoire", "18 juin", "26 ans", "INSENSE"}
                };
                for (const auto& info : infos)
                {
                        Mat img = imread(info.path, CV_LOAD_IMAGE_GRAYSCALE);
                        cv::resize(img, img, Size(256, 256));
                        Person p;
                        p.name = info.name;
                        p.surname = info.surname;
                        p.birthdate = info.birth;
                        p.age = info.age;
                        p.company = info.company;
                        p.img = Image(img, 0, 0, 0, 0, 0, 0);
                        p.img.set_frameNdg(p.img.get_frameCouleur());
                        p.img.set_frameLbp(p.img.ConvertToLbp(p.img.get_frameNdg()));
                        referencePersons.push_back(p);
                }
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
				if (imagePourTraitement != nullptr)
				{
					imwrite("COL_" + std::to_string(nImage) + ".jpg", imagePourTraitement->get_frameCouleur());
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
	imagePourTraitement = NULL;
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
			imagePourTraitement = new Image(Image::resize(imageReduite->get_frameCouleur()(roi_c), Size(256, 256)), 1, 1, 1, 0, 0, 0);
		}

		// Dessin du visage détecté sur l'image principale
		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		// ReSharper disable once CppMsExtBindingRValueToLvalueReference
		rectangle(imageCamera->get_frameCouleur(), pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);
                if (imagePourTraitement != NULL && !referencePersons.empty())
                {
                        // Calcul du score pour chaque personne
                        double bestScore = std::numeric_limits<double>::max();
                                Person bestPerson;
                                for (const auto& ref : referencePersons)
                                {
                                        std::vector<double> scores = ChiDeu(imagePourTraitement->get_frameLbp(), ref.img.get_frameLbp(), 8, 8);
                                        double total = 0;
                                        for (double s : scores)
                                                total += s;
                                        if (total < bestScore)
                                        {
                                                bestScore = total;
                                                bestPerson = ref;
                                        }
                                }
                                detectedPerson_Name = bestPerson.name;
                                detectedPerson_Surname = bestPerson.surname;
                                detectedPerson_Birthhdate = bestPerson.birthdate;
                                detectedPerson_Age = bestPerson.age;
                                detectedPerson_Company = bestPerson.company;
                                putText(imageCamera->get_frameCouleur(), detectedPerson_Surname, cvPoint((faces[ic].x + faces[ic].width / 4), faces[ic].y - 10), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
                        }
		}
	}
	// Affichage des differentes images
	imshow("WebCam", imageCamera->get_frameCouleur());
	imshow("WebCamNdg", imageCamera->get_frameNdg());
	cvMoveWindow("WebCam", 0, 0);
	cvMoveWindow("WebCam", 1000, 0);
	if (imagePourTraitement != NULL)
	{
		imshow("imageLBP", ShowImageOverlay(imagePourTraitement->get_frameLbp()));
		imshow("imageNDG", ShowImageOverlay(imagePourTraitement->get_frameNdg()));
		cvMoveWindow("imageLBP", 800, 500);
		cvMoveWindow("imageNDG", 1100, 500);

		cvMoveWindow("imageRef", 0, 600);

	}
	else{
		destroyWindow("imageLBP");
		destroyWindow("imageNDG");
	}
}
#pragma endregion



