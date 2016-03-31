#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;


class Traitements
{
public:
	Traitements();
	~Traitements();
	//Fonctions
	static Mat HistogrammeCouleur(Mat);
	static Mat HistogrammeNDG(Mat);
	static Mat LBP(Mat);
	static vector<int> CreateHistograme(Mat);
	static int comp(Mat a, Mat b, Mat c);
};

