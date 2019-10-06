#include "HarrisCorner.h"
#include <math.h>
#include <chrono>
#include <fstream>


using namespace std;
using namespace cv;

//--- Utility functions
void showImage(const Mat& image, string title = "") {

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

//--- Harris corner detection
HarrisCornerCPU::HarrisCornerCPU() {

	//std::cout << "Harris Corner Construction" << std::endl;
	//--- Input file reading
	fileName = "D:\\Zebra\\GPU_and_CPU\\src\\reference.png";

}

Mat HarrisCornerCPU::Gaussian(const Mat &input) {

	Mat blur = Mat::zeros(cv::Size(input.cols, input.rows), CV_32F);

	Mat kernel = Mat::zeros(cv::Size(3, 3), CV_32F);
	getGausianKernel(sigma, kernel, 3);
	//std::cout << "kernel : " << kernel << std::endl;
	filter<float>(input, blur, kernel);

	return blur;

}

Mat HarrisCornerCPU::Gradient(const Mat& input, bool bVertical) {

	Mat Derivative = Mat(cv::Size(input.cols, input.rows), CV_32F);
	Mat kernel;
	//Vertical Edge
	if (bVertical) {
		kernel = (Mat_<float>(3, 3) << -1.0, -2.0, -1.0,
			0.0, 0.0, 0.0,
			1.0, 2.0, 1.0);
	}
	//Horizontal edge
	else {
		kernel = (Mat_<float>(3, 3) << -1.0, 0.0, 1.0,
			-2.0, 0.0, 2.0,
			-1.0, 0.0, 1.0);
	}

	filter<float>(input, Derivative, kernel);

	return Derivative;

}

template <class T>
void HarrisCornerCPU::filter(const Mat& input, Mat& output, Mat& kernel) {

	for (int i = 0; i < input.rows; ++i) {
		for (int j = 0; j < input.cols; ++j) {

			float fFilterOutput = 0.0;
			for (int ky = -1; ky < 2; ++ky) {
				for (int kx = -1; kx < 2; ++kx) {

					//boundary checking
					if ((i + ky) < 0 || (i + ky) >= input.rows) {
						continue;
					}

					if ((j + kx) < 0 || (j + kx) >= input.cols) {
						continue;
					}

					try {
						fFilterOutput += kernel.at<float>(kx + 1, ky + 1) * input.at<T>(i + ky, j + kx);
					}
					catch (...) {
						std::cout << "i + kx" << i + kx << ", j + ky" << j + ky;
						throw std::runtime_error("Error caught.");
						return;
					}
				}
			}

			output.at<T>(i, j) = (T)fFilterOutput;

		}
	}
}

Mat HarrisCornerCPU::computSumOfProduct(const Mat& input1, const Mat& input2, const int windowSize) {

	//TODO, check, input1 and input2 should have same size
	Mat output = Mat::zeros(cv::Size(input1.cols, input1.rows), CV_32F);

	int filterWidth = windowSize / 2;

	for (int i = 0; i < input1.rows; ++i) {
		for (int j = 0; j < input1.cols; ++j) {

			float fFilterOutput = 0.0;
			for (int ky = -filterWidth; ky <= filterWidth; ++ky) {
				for (int kx = -filterWidth; kx <= filterWidth; ++kx) {

					//boundary checking
					if ((i + ky) < 0 || (i + ky) >= input1.rows) {
						continue;
					}

					if ((j + kx) < 0 || (j + kx) >= input1.cols) {
						continue;
					}

					try {
						fFilterOutput += input1.at<float>(i + ky, j + kx) * input2.at<float>(i + ky, j + kx);
					}
					catch (...) {
						std::cout << "i + kx" << i + kx << ", j + ky" << j + ky;
						throw std::runtime_error("Error caught.");
					}
				}
			}

			output.at<float>(i, j) = (float)fFilterOutput;

		}
	}

	return output;

}

void HarrisCornerCPU::getGausianKernel(const float sigma, Mat& kernel, int kernelSize) const {

#define M_PI 3.14159265358979323846

	double r, s = 2.0 * sigma * sigma;

	// sum is for normalization 
	float sum = 0.0;

	int shift = kernelSize / 2;

	for (int x = -shift; x <= shift; x++) {
		for (int y = -shift; y <= shift; y++) {
			r = sqrt(x * x + y * y);
			kernel.at<float>(x + shift, y + shift) = (exp(-(r * r) / s)) / (M_PI * s);
			sum += kernel.at<float>(x + shift, y + shift);
		}
	}

	// normalising the Kernel 
	for (int i = 0; i < kernelSize; ++i)
		for (int j = 0; j < kernelSize; ++j)
			kernel.at<float>(i, j) /= sum;
}

Mat HarrisCornerCPU::convert2Grayscale(const Mat& input) {

	Mat greyscaleImg(input.rows, input.cols, CV_32F);

	for (int c = 0; c < input.cols; c++) {
		for (int r = 0; r < input.rows; r++) {
			greyscaleImg.at<float>(r, c) =
				0.2126 * input.at<cv::Vec3b>(r, c)[0] +
				0.7152 * input.at<cv::Vec3b>(r, c)[1] +
				0.0722 * input.at<cv::Vec3b>(r, c)[2];
		}
	}

	return greyscaleImg;
}

Mat HarrisCornerCPU::computResponse(const Mat& Sxx, const Mat& Syy, const Mat& Sxy, float k) {

	//TODO: check the shapes
	Mat response = Mat::zeros(cv::Size(Sxx.cols, Sxx.rows), CV_32F);

	for (int i = 0; i < Sxx.rows; ++i) {
		for (int j = 0; j < Sxx.cols; ++j) {

			float r = (Sxx.at<float>(i, j) * Syy.at<float>(i, j) - Sxy.at<float>(i, j) * Sxy.at<float>(i, j)) -
				k * std::pow((Sxx.at<float>(i, j) + Syy.at<float>(i, j)), 2);

			response.at<float>(i, j) = r;

		}
	}

	return response;

}

Mat HarrisCornerCPU::nonMaximumSuppression(const Mat& response) {

	double minVal, maxVal;

	//get the maximum and minimum intensities
	cv::minMaxLoc(response, &minVal, &maxVal);

	//Final image showing only the location of Harris Corners
	Mat output = Mat::zeros(cv::Size(response.cols, response.rows), CV_8U);

	double matchThreshold = responseThreshold * maxVal;

	for (int i = 0; i < response.rows; ++i) {
		for (int j = 0; j < response.cols; ++j) {

			if (response.at<float>(i, j) >= matchThreshold) {
				output.at<uchar>(i, j) = 255;
			}

		}
	}

	return output;

}

void HarrisCornerCPU::run(const Mat& inImage, Mat& outImage) {

	using namespace std::chrono;

	Mat inImageFloat(inImage.size(), CV_32FC3);
	inImage.convertTo(inImageFloat, CV_32FC3);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	//--- Harris Corner Detection
	Mat gray = convert2Grayscale(inImage);
	//showImage(gray, "grayscale");

	//Gaussian blur
	Mat blur = Gaussian(gray);
	//showImage(blur, "gaussianBlur");

	//std::fstream file("gray.bin", std::ios::out | std::ios::binary);
	//file.write((char*)gray.data, gray.cols * gray.rows * 4);
	//file.close();

	//Ix derivatives
	Mat Ix = Gradient(blur, false);
	//showImage(Ix, "Ix");

	//Iy derivatives
	Mat Iy = Gradient(blur, true);
	//showImage(Iy, "Iy");

	//Sum of product of derivates
	Mat Sxx = computSumOfProduct(Ix, Ix, 3);
	Mat Syy = computSumOfProduct(Iy, Iy, 3);
	Mat Sxy = computSumOfProduct(Ix, Iy, 3);

	//get the response
	Mat R = computResponse(Sxx, Syy, Sxy);
	//showImage(R, "Harris Responses");

	//apply non-maximum suppression
	Mat corners = nonMaximumSuppression(R);
	//showImage(corners, "Harris Corners");

	//Timing
	high_resolution_clock::time_point end = high_resolution_clock::now();
	duration<double> totalCpuTime = duration_cast<duration<double>> (end - start);
	std::cout << "Total time took on CPU: " << totalCpuTime.count() << " seconds." << std::endl;

	//showImage(corners, "HarrisCornerCPU");

	outImage = corners;
}