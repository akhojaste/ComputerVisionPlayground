#include "iostream"
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "HarrisCorner.h"
#include "CudaKernels.h"

using namespace std;
using namespace cv;

//--- Utility functions
void showImageGPU(const Mat& image, string title = "") {

	if (!image.data)
	{
		printf("No image data \n");
	}

	Mat image2;
	cv::normalize(image, image2, 0, 255, cv::NORM_MINMAX, CV_8U);

	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, image2);

	waitKey(0);
}

int main(int argc, char* argv[])
{
	//Call the GPU version here
	std::cout << "----------------------------------------------" << std::endl;

	Mat inImage = cv::imread(argv[1]);
	if (!inImage.data) {

		std::cout << "No filename is provided. Usage: HarrisCorner_test.exe filename" << std::endl;
		return -1;
	}
	showImageGPU(inImage, "Original Image");

	HarrisCornerGPU gpuRun;
	Mat outImage(inImage.size(), CV_32F);

	gpuRun.run(inImage, outImage);
	showImageGPU(outImage, "HarrisCornerGPU");
	imwrite("GPUCorners.jpg", outImage);

	//Call the CPU version here
	std::cout << "----------------------------------------------" << std::endl;
	HarrisCornerCPU h;
	Mat outImageCPU(inImage.size(), CV_32F);
	h.run(inImage, outImageCPU);
	showImageGPU(outImageCPU, "HarrisCornerCPU");
	imwrite("CPUCorners.jpg", outImageCPU);

	return 0;
}
