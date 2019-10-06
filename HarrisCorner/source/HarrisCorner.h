#pragma once

#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// Harris corner algorithm implemented on CPU
class HarrisCornerCPU {

public:
	HarrisCornerCPU();
	~HarrisCornerCPU() {}
	void run(const Mat& inImage, Mat& outImage);

private:
	Mat convert2Grayscale(const Mat& inpput);
	Mat Gaussian(const Mat& input);
	Mat Gradient(const Mat& input, bool bVertical);
	Mat computSumOfProduct(const Mat& input1, const Mat& input2, const int windowSize);
	Mat computResponse(const Mat& Sxx, const Mat& Syy, const Mat& Sxy, float k = 0.04);
	Mat nonMaximumSuppression(const Mat& response);
	void getGausianKernel(const float sigma, Mat& kernel, int kernelSize = 3) const;
	template <class T>
	void filter(const Mat& input, Mat& output, Mat& kernel);

private:
	float sigma = 2.0;
	std::string fileName;
	float responseThreshold = 0.01;
	Mat harrisResponse;
};