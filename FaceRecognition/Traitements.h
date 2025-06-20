#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>


class Traitements
{
public:
	Traitements();
	~Traitements();
	//Fonctions
	static cv::Mat HistogrammeCouleur(cv::Mat);
	static cv::Mat HistogrammeNDG(cv::Mat);
	static cv::Mat LBP(cv::Mat);
	static cv::Mat ELBP(const cv::Mat& src, int radius, int neighbors);
	static std::vector<int> CreateHistograme(cv::Mat, bool = false);
	static cv::Mat PreprocessingWithTanTrigs(cv::InputArray src, float alpha = 0.1, float tau = 10.0, float gamma = 0.2, int sigma0 = 1, int sigma1 = 2);
};

