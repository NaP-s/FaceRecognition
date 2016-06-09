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
	static Mat ELBP(const Mat& src, int radius, int neighbors);
	static vector<int> CreateHistograme(Mat);
	static int comp(Mat a, Mat b, Mat c);
	static Mat PreprocessingWithTanTrigs(InputArray src, float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1, int sigma1 = 2);
};

