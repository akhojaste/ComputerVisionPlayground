#include "CudaKernels.h"

#include "iostream"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

#define BLOCK_SIZE 8

//--- Utility functions
void showImg(const Mat& image, string title = "") {

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

void testDebug(const float* d_image, size_t size) {

	cudaError_t cudaStatus;

	float* h_image = new float[size];
	cudaStatus = cudaMemcpy((void *)h_image, (void *)d_image, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}

	Mat img(512, 512, CV_32F, h_image);
	showImg(img);
}

void __global__ filter(const float* inputImg, float* outputImg, const int width, const int height, const float* kernel, const int kernelSize);
void __global__ rgb2gray(const float* pfimagIn, float* pfimgOut, const int width, const int height, const int channels);
void __global__ computSumOfProduct(const float* inputImg1, const float* inputImg2, float* outputImg,
	const int width, const int height, const int windowSize);
void __global__ computResponse(const float* Sxx, const float* Syy, const float* Sxy, float* response, int width, int height, float k = 0.04);
void __global__ Array_Reduction_Kernel(float * g_iData, float *g_Intermediat);
void __global__ nonMaximumSuppression(const float* response, float* corners, const int width, const int height, const float threshold, const float* maxR);


//GPU Kernel
void __global__ filter(const float* inputImg, float* outputImg, const int width, const int height, const float* kernel, const int kernelSize)
{

	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	int kernelWidth = (kernelSize) / 2;
	int i = iRow, j = iCol;
	float fFilterOutput = 0.0;
	for (int ky = -kernelWidth; ky <= kernelWidth; ++ky) {
		for (int kx = -kernelWidth; kx <= kernelWidth; ++kx) {

			//boundary checking
			if ((i + ky) < 0 || (i + ky) >= height) {
				continue;
			}

			if ((j + kx) < 0 || (j + kx) >= width) {
				continue;
			}

			fFilterOutput += kernel[(ky + 1) * 3 + (kx + 1)] * inputImg[(i + ky) * width + (j + kx)];

			//Testing, just output the inputImg, PASSED.
			//fFilterOutput = inputImg[i * width + j];
		}
	}

	outputImg[i * width + j] = fFilterOutput;

	//Testing, just output the inputImg, PASSED.
	//outputImg[i * width + j] = inputImg[i * width + j];
}

//GPU Kernel
void __global__ rgb2gray(const float* pfimagIn, float* pfimgOut, const int width, const int height, const int channels)
{
	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	// The B value of the pixel, images in the OpenCV are in BGR format not RGB
	long long lPixelIdx = (iRow * width + iCol) * channels;

	float B = pfimagIn[lPixelIdx];
	float G = pfimagIn[lPixelIdx + 1];
	float R = pfimagIn[lPixelIdx + 2];

	//Weighted method or luminosity method. The controbution of each color
	//to the final gray scale image is different. So if we take simply the
	//average of RGB, i.e. (R + G + B) / 3, then the image has low contrast
	//and it is mostly black. Also our eyes interpret different colors differently
	//float temp = 0.11 * B + 0.59 * G + 0.3 * R;
	float temp = 0.2126 * B + 0.7152 * G + 0.0722 * R;

	//Convert the RGB colorful image to the grayscale
	pfimgOut[iRow * width + iCol] = temp;
}

void __global__ computSumOfProduct(const float* inputImg1, const float* inputImg2, float* outputImg,
	const int width, const int height, const int windowSize) {

	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	int kernelWidth = (windowSize) / 2;
	int i = iRow, j = iCol;
	float fFilterOutput = 0.0;
	for (int ky = -kernelWidth; ky <= kernelWidth; ++ky) {
		for (int kx = -kernelWidth; kx <= kernelWidth; ++kx) {

			//boundary checking
			if ((i + ky) < 0 || (i + ky) >= height) {
				continue;
			}

			if ((j + kx) < 0 || (j + kx) >= width) {
				continue;
			}

			fFilterOutput += inputImg1[(i + ky) * width + (j + kx)] * inputImg2[(i + ky) * width + (j + kx)];

			//Testing, just output the inputImg, PASSED.
			//fFilterOutput = inputImg[i * width + j];
		}
	}

	outputImg[i * width + j] = fFilterOutput;

	//Testing, just output the inputImg, PASSED.
	//outputImg[i * width + j] = inputImg[i * width + j];

}


void __global__ computResponse(const float* Sxx, const float* Syy, const float* Sxy, float* response, int width, int height, float k) {

	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	int i = iRow, j = iCol;
	float fFilterOutput = 0.0;

	float r = Sxx[i * width + j] * Syy[i * width + j] - Sxy[i * width + j] * Sxy[i * width + j] -
		k * (Sxx[i * width + j] + Syy[i * width + j]) * (Sxx[i * width + j] + Syy[i * width + j]);

	response[i * width + j] = r;

	//Testing, just output the inputImg, PASSED.
	//outputImg[i * width + j] = inputImg[i * width + j];
}


void __global__ Array_Reduction_Kernel(float * g_iData, float *g_Intermediat)
{
	extern __shared__ float sData[]; //Size is determined by the host

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int tId = threadIdx.x;

	//First every thread in the block puts its value intor the shared memory
	sData[tId] = g_iData[index];
	__syncthreads();

	for (unsigned int idx = blockDim.x / 2; idx > 0; idx >>= 1)
	{
		if (tId < idx)
		{
			sData[tId] = max(sData[tId], sData[idx + tId]);
		}

		__syncthreads();
	}

	g_Intermediat[blockIdx.x] = sData[0];
}

void __global__ nonMaximumSuppression(const float* response, float* corners, const int width, const int height, const float threshold, const float* maxR) {

	int iRow = blockIdx.y * blockDim.y + threadIdx.y;
	int iCol = blockIdx.x * blockDim.x + threadIdx.x;

	int i = iRow, j = iCol;

	float t = threshold * (*maxR);

	if (response[i * width + j] > t) {
		corners[i * width + j] = 255.0;
	}
	else {
		corners[i * width + j] = 0.0;
	}

}

void HarrisCornerGPU::allocateDeviceMemory(float** d_buffer, size_t bufferSize) {
	
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void **)d_buffer, rgbSize);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda. " << cudaStatus << std::endl;
		return;
	}

}


void HarrisCornerGPU::allocateAllMemories() {
	cudaError_t cudaStatus;

	float* rgb = (float*)inputImage.data;

	const int width = inputImage.cols;
	const int height = inputImage.rows;
	const int channels = 3;

	//Total number of Bytes of the image
	rgbSize = width * height * channels * sizeof(float);
	graySize = width * height * sizeof(float);

	//-- GPU memory allocation for bufferss

	//rgb image
	allocateDeviceMemory(&d_rgb, rgbSize);

	//grayscale image
	allocateDeviceMemory(&d_gray, graySize);

	//gaussian blur image
	allocateDeviceMemory(&d_blur, graySize);

	//kernel for filter
	allocateDeviceMemory(&d_kernel, 9 * sizeof(float));
	
	//Ix
	allocateDeviceMemory(&d_Ix, graySize);

	//Iy
	allocateDeviceMemory(&d_Iy, graySize);

	//Sxx
	allocateDeviceMemory(&d_Sxx, graySize);

	//Syy
	allocateDeviceMemory(&d_Syy, graySize);

	//Sxy
	allocateDeviceMemory(&d_Sxy, graySize);

	//Response
	allocateDeviceMemory(&d_Response, graySize);

	//Corners
	allocateDeviceMemory(&d_corners, graySize);

}

//Copy memory from host to device
void HarrisCornerGPU::copy2Device(float* h_buffer, float* d_buffer, size_t size2Copy) {

	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy((void *)d_buffer, (void *)h_buffer, size2Copy, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}
}


void HarrisCornerGPU::freeMemory() {
	cudaError_t cudaStatus;

	//*d_rgb, *d_gray, *d_blur, *d_kernel, *d_Ix, *d_Iy,
	//	*d_Sxx, *d_Sxy, *d_Syy, *d_Response, *d_corners;

	if (d_rgb != nullptr) {
		cudaStatus = cudaFree((void *)d_rgb);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_rgb = nullptr;
	}

	if (d_gray != nullptr) {
		cudaStatus = cudaFree((void *)d_gray);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_gray = nullptr;
	}

	if (d_blur != nullptr) {
		cudaStatus = cudaFree((void *)d_blur);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_blur = nullptr;
	}

	if (d_kernel != nullptr) {
		cudaStatus = cudaFree((void *)d_kernel);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_kernel = nullptr;
	}

	if (d_Ix != nullptr) {
		cudaStatus = cudaFree((void *)d_Ix);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Ix = nullptr;
	}

	if (d_Iy != nullptr) {
		cudaStatus = cudaFree((void *)d_Iy);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Iy = nullptr;
	}

	if (d_Sxx != nullptr) {
		cudaStatus = cudaFree((void *)d_Sxx);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Sxx = nullptr;
	}

	if (d_Sxy != nullptr) {
		cudaStatus = cudaFree((void *)d_Sxy);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Sxy = nullptr;
	}

	if (d_Syy != nullptr) {
		cudaStatus = cudaFree((void *)d_Syy);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Syy = nullptr;
	}

	if (d_Response != nullptr) {
		cudaStatus = cudaFree((void *)d_Response);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_Response = nullptr;
	}

	if (d_corners != nullptr) {
		cudaStatus = cudaFree((void *)d_corners);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_corners = nullptr;
	}

	if (d_maxResponseIntermediate != nullptr) {
		cudaStatus = cudaFree((void *)d_maxResponseIntermediate);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_maxResponseIntermediate = nullptr;
	}

	if (d_maxResponse != nullptr) {
		cudaStatus = cudaFree((void *)d_maxResponse);
		if (cudaStatus != cudaSuccess) {
			std::cout << "Failure on cuda" << std::endl;
			return;
		}
		d_maxResponse = nullptr;
	}
}


void HarrisCornerGPU::runKernelCascade(Mat& inImage, Mat& outImage) {

	cudaError_t cudaStatus;

	//Timing
	cudaEvent_t start, stop;
	float elapsed = 0.0;
	cudaStatus = cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}
	cudaStatus = cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}

	//Start timing
	cudaStatus = cudaEventRecord(start, 0);

	//allocate memory on device
	allocateAllMemories();
	
	//copy input image to GPU buffer
	float* rgb = (float*)inputImage.data;
	copy2Device(rgb, d_rgb, rgbSize);
	//testDebug(d_rgb, rgbSize);

	//-- Transfer data to GPU
	float h_kernel[9] = { 0 };
	getGaussianKernel(sigma, h_kernel);
	copy2Device(h_kernel, d_kernel, 9 * sizeof(float));


	//--Compue the results in GPU
	dim3 dNumThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE); //  Each thread block contains this much threads
													  // This amount of thread blocks
													  //Total number of threads that will be launched are dimGrid.x * dimGird.y * dimBlocks.x * dimBlocks.y
													  //NOTE: the toal numer of thread per block, i.e. dimBlock.x * dimBlock.y should not excede 1024 and
													  //in some system 512
	dim3 dNumBlocks(inputImage.cols / dNumThreadsPerBlock.x, inputImage.rows / dNumThreadsPerBlock.y);

	//GPU Kernel
	rgb2gray << < dNumBlocks, dNumThreadsPerBlock >> >  (d_rgb, d_gray, inputImage.cols, inputImage.rows, 3);
	//testDebug(d_gray, graySize);

	//GPU Kernel
	filter << < dNumBlocks, dNumThreadsPerBlock >> >  (d_gray, d_blur, inputImage.cols, inputImage.rows, d_kernel, 3);
	//testDebug(d_blur, graySize);

	//horizontalEdgeKernel
	float h_horizontalEdgeKernel[9] = { -1.0, 0.0, 1.0,
		-2.0, 0.0, 2.0,
		-1.0, 0.0, 1.0 };

	//verticalEdgeKernel
	float h_verticalEdgeKernel[9] = { -1.0, -2.0, -1.0,
		0.0, 0.0, 0.0,
		1.0, 2.0, 1.0 };


	//Gradient, Ix
	copy2Device(h_horizontalEdgeKernel, d_kernel, 9 * sizeof(float));
	filter << < dNumBlocks, dNumThreadsPerBlock >> >  (d_blur, d_Ix, inputImage.cols, inputImage.rows, d_kernel, 3);
	//testDebug(d_Ix, graySize);

	//Gradient, Ix
	copy2Device(h_verticalEdgeKernel, d_kernel, 9 * sizeof(float));
	filter << < dNumBlocks, dNumThreadsPerBlock >> >  (d_blur, d_Iy, inputImage.cols, inputImage.rows, d_kernel, 3);
	//testDebug(d_Iy, graySize);

	//Sxx
	computSumOfProduct << < dNumBlocks, dNumThreadsPerBlock >> > (d_Ix, d_Ix, d_Sxx, inputImage.cols, inputImage.rows, 3);
	//testDebug(d_Sxx, graySize);

	//Sxy
	computSumOfProduct <<< dNumBlocks, dNumThreadsPerBlock >>> (d_Ix, d_Iy, d_Sxy, inputImage.cols, inputImage.rows, 3);
	//testDebug(d_Sxy, graySize);

	//Syy
	computSumOfProduct <<< dNumBlocks, dNumThreadsPerBlock >>> (d_Iy, d_Iy, d_Syy, inputImage.cols, inputImage.rows, 3);
	//testDebug(d_Syy, graySize);


	//Harris respone
	computResponse <<< dNumBlocks, dNumThreadsPerBlock >>> (d_Sxx, d_Syy, d_Sxy, d_Response, inputImage.cols, inputImage.rows, 0.04);
	//testDebug(d_Response, graySize);

	//Get the maximum valud of Response
	allocateDeviceMemory(&d_maxResponseIntermediate, inputImage.rows * sizeof(float));
	allocateDeviceMemory(&d_maxResponse, 1 * sizeof(float));

	//Need to get the maximum of input image, this happens in 2 steps:
	//first, we reduce each columns of the image, so for lets say an image of size 512x512
	//we get a reduced array of 512 values after this step. Then we reduce this to 1 value
	//with launching 512 threads to do the reduction
	int blocks = inputImage.rows;
	int threadPerBlock = inputImage.cols;
	int iMemSize = threadPerBlock * sizeof(float); // to pass to the cuda kernel so it allocates shared memory there
	Array_Reduction_Kernel << <blocks, threadPerBlock, iMemSize >> >(d_Response, d_maxResponseIntermediate);

	//now reduce it again to one value
	//note this "1" here, only 1 block of threads is called. 
	//with "g_Intermediat[blockIdx.x]" call, the blockIdx.x will be 0
	blocks = 1; 
	threadPerBlock = inputImage.rows;
	iMemSize = threadPerBlock * sizeof(float);
	Array_Reduction_Kernel << <blocks, threadPerBlock, iMemSize >> >(d_maxResponseIntermediate, d_maxResponse);

	//Debug, check the maximum response
	float h_maxR = 0.0;
	cudaStatus = cudaMemcpy((void *)(&h_maxR), (void *)d_maxResponseIntermediate, sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}

	//Non-maximum suppression
	nonMaximumSuppression <<< dNumBlocks, dNumThreadsPerBlock >> > (d_Response, d_corners, inputImage.cols, inputImage.rows, responseThreshold, d_maxResponse);
	//testDebug(d_corners, graySize);

	//Transfer the results to CPU
	cudaStatus = cudaMemcpy((void *)outImage.data, (void *)d_corners, graySize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}

	//Stop timing and measure elapsed
	cudaStatus = cudaEventRecord(stop, 0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}
	cudaStatus = cudaEventSynchronize(stop);
	if (cudaStatus != cudaSuccess) {
		std::cout << "Failure on cuda" << std::endl;
		return;
	}

	cudaStatus = cudaEventElapsedTime(&elapsed, start, stop);

	//Destroy event recorders
	cudaStatus = cudaEventDestroy(start);
	cudaStatus = cudaEventDestroy(stop);

	std::cout << "Total elapsedd processing time in GPU: " << elapsed << " ms." << std::endl;

	freeMemory();
}


void HarrisCornerGPU::getGaussianKernel(const float sigma, float* kernel, const int kernelSize) {

#define M_PI 3.14159265358979323846

		double r, s = 2.0 * sigma * sigma;

		// sum is for normalization 
		float sum = 0.0;

		int shift = kernelSize / 2;

		for (int x = -shift; x <= shift; x++) {
			for (int y = -shift; y <= shift; y++) {
				r = sqrt(x * x + y * y);
				kernel[(y + shift) * kernelSize + x + shift] = (exp(-(r * r) / s)) / (M_PI * s);
				//kernel.at<float>(x + shift, y + shift) = (exp(-(r * r) / s)) / (M_PI * s);
				sum += kernel[(y + shift) * kernelSize + x + shift];
			}
		}

		// normalising the Kernel 
		for (int i = 0; i < kernelSize; ++i)
			for (int j = 0; j < kernelSize; ++j)
				kernel[i * kernelSize + j] /= sum;
}

//--------------------------------
void HarrisCornerGPU::run(const Mat& inImage, Mat& outImage) {

	//making sure image is rgb (or bga to be more exact!)
	Mat inImageFloat(inImage.rows, inImage.cols, CV_32FC3);
	Mat bga;
	if (inImage.channels() == 4) {
		cvtColor(inImage, bga, cv::COLOR_BGRA2BGR);
		bga.convertTo(inImageFloat, CV_32FC3);
	}
	else {
		inImage.convertTo(inImageFloat, CV_32FC3);
	}


	inputImage = inImageFloat;

	runKernelCascade(inImageFloat, outImage);

}