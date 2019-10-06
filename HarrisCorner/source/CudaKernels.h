#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

// Harris corner algorithm implemented on CUDA
class HarrisCornerGPU {

public:

	HarrisCornerGPU() 
	{
		//initialize the memories
		d_rgb = d_gray = d_blur = d_kernel = d_Ix = d_Iy = 
			d_Sxx = d_Sxy = d_Syy = d_Response = d_corners = d_maxResponseIntermediate = d_maxResponse = nullptr;

	}
	~HarrisCornerGPU() {}

	void run(const Mat& inImage, Mat& outImage);

private:

	void allocateDeviceMemory(float** d_buffer, size_t bufferSize);
	void allocateAllMemories();
	void copy2Device(float* h_buffer, float* d_buffer, size_t size2Copy);
	void runKernelCascade(Mat& inImage, Mat& outImage);
	void freeMemory();
	void getGaussianKernel(const float sigma, float* kernel, int kernelSize = 3);

private:

	Mat inputImage;
	float sigma = 2.0;
	float responseThreshold = 0.01;

	//The buffers in the GPU 
	float *d_rgb, *d_gray, *d_blur, *d_kernel, *d_Ix, *d_Iy,
		*d_Sxx, *d_Sxy, *d_Syy, *d_Response, *d_corners, *d_maxResponse, *d_maxResponseIntermediate;

	size_t graySize, rgbSize;


};