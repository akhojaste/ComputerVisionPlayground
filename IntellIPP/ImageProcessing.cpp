#include "ImageProcessing.h"



ImageProcessing::ImageProcessing()
{
}


ImageProcessing::~ImageProcessing()
{
}

void ImageProcessing::ReadAndFilterIPP()
{
	long long lStart, lEnd, lFreq;
	double dbTime = 0.0;

	unsigned short usHeight = 962;
	unsigned short usWidth  = 1372;
	std::string InputImage = "F:\\data\\binImageIn3.dat";
	std::string OutputImage = "F:\\data\\binImageOut3.dat";

	//Read the image here
	std::ifstream file(InputImage, std::ios::ate | std::ios::binary);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	unsigned char *pSrcBuff = new unsigned char[size];
	unsigned char *pDstBuff = new unsigned char[size];

	if (file.is_open())
	{
		file.read((char *)pSrcBuff, size);
		file.close();
	}
	else
		return;

	//////get sizes of the spec structure and the work buffer
	//IppStatus status = ippStsNoErr;
	//IppiSize FilterSize = { 3, 3 };
	//IppiSize SrcSize = { usHeight, usWidth};
	//Ipp8u *pSrc = (Ipp8u *)pSrcBuff;
	//Ipp8u *pDst = (Ipp8u *)pDstBuff;
	//Ipp8u *pBuffer = nullptr;
	//int srcStep = 256, dstStep = 256; //what is this?!!
	//int iBufferSize = 0;

	//status = ippiFilterBoxBorderGetBufferSize(SrcSize, FilterSize, IppDataType::ipp8u, 1, &iBufferSize);
	//pBuffer = ippsMalloc_8u(iBufferSize);

	//status = ippiFilterBoxBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, SrcSize, FilterSize, ippBorderRepl, NULL, pBuffer);

	//const char *Error;
	//Error = ippGetStatusString(status);

	//if (pBuffer) ippsFree(pBuffer);

	////SHARPENING FILTER, the kernel is different

	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);

	IppStatus status = ippStsNoErr;
	IppiMaskSize FilterSize = ippMskSize3x3;
	IppiSize SrcSize = { usWidth, usHeight };
	Ipp8u *pSrc = (Ipp8u *)pSrcBuff;
	Ipp8u *pDst = (Ipp8u *)pDstBuff;
	Ipp8u *pBuffer = nullptr;
	int srcStep = usWidth, dstStep = usWidth; //This is the distance in bytes between consecutive lines of source and destination
	int iBufferSize = 0;

	//Getting the buffer size
	status = ippiFilterSharpenBorderGetBufferSize(SrcSize, FilterSize, IppDataType::ipp8u, IppDataType::ipp8u, 1, &iBufferSize);
	pBuffer = ippsMalloc_8u(iBufferSize);

	//Actual filter
	IppiSize RoiSize = { usWidth, usHeight};
	status = ippiFilterSharpenBorder_8u_C1R(pSrc, srcStep, pDst, dstStep, RoiSize, FilterSize, IppiBorderType::ippBorderConst, 0, pBuffer);

	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);
	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);
	
	dbTime = ((double)lEnd - (double)lStart) / ((double)lFreq);
	dbTime *= 1000.0; //ms

	std::cout << "Time with Intel 3x3 filter: " << dbTime << " ms"<< std::endl;

	const char *Error;
	Error = ippGetStatusString(status);

	if (pBuffer) ippsFree(pBuffer);

	//Write the data to the file
	std::ofstream outFile(OutputImage, std::ios::out | std::ios::binary);
	outFile.write((char *)pDst, usWidth * usHeight);
	outFile.close();

	delete pSrcBuff;
	delete pDstBuff;

	return;
}

void ImageProcessing::ReadAndFilterRaw()
{
	long long lStart, lEnd, lFreq;
	double dbTime = 0.0;

	unsigned short usHeight = 1372;
	unsigned short usWidth  = 1372;
	std::string InputImage = "F:\\data\\binImageIn2.dat";
	std::string OutputImage = "F:\\data\\binImageOut2.dat";

	//Read the image here
	std::ifstream file(InputImage, std::ios::ate | std::ios::binary);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	unsigned char *pSrcBuff = new unsigned char[size];
	unsigned char *pDstBuff = new unsigned char[size];

	if (file.is_open())
	{
		file.read((char *)pSrcBuff, size);
		file.close();
	}

	//Intel IPPs sharpenning filter kernel
	// -1 -1 -1
	// -1 16 -1  X  1/8
	// -1 -1 -1

	//A 3x3 Sharpening filter
	const short sFilterW = 3;
	const short sFilterH = 3;
	const short sFilterSize = 9;
	double dbFilterCoeff[sFilterSize] = { -1.0/8.0, -1.0/8.0, -1.0/8.0,
										  -1.0/8.0, 16.0/8.0, -1.0/8.0,
										  -1.0/8.0, -1.0/8.0, -1.0/8.0};


	BYTE *pline = nullptr;
	long lRow = 0, lCol = 0;

	long lLineWidth = (long)usWidth;

	short sFRow = 0, sFCol = 0;
	short sFRowShift = sFilterH / 2;
	short sFColShift = sFilterW / 2;
	short sKImage = 1;
	short sKFilter = 1;

	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);
 
	for (int i = 0; i < usHeight * usWidth; ++i)
	{
		//Taking Modes and Dividing is slowing down the filter so we use a simple counter instead
		if (i == sKImage * usWidth) //We are at the next line
		{
			lRow++;
			lCol = 0;
			sKImage++;
		}
		/*lRow = i / usWidth;
		lCol = i % usWidth;*/

		double dbFilterOutput = 0.0;
		sFCol = 0;
		sFRow = 0;

		for (int iFilterCount = 0; iFilterCount < sFilterSize; ++iFilterCount)
		{
			//Taking Modes and Dividing is slowing down the filter so we use a simple counter instead
			if (iFilterCount == sKFilter * sFilterW)
			{
				sFRow++;
				sFCol++;
				sKFilter++;
			}

			////sFRow = iFilterCount / sFilterW;
			////sFCol = iFilterCount % sFilterW;

			long lImageRow = lRow + sFRow - sFRowShift;
			long lImageCol = lCol + sFCol - sFColShift;

			if (lImageRow < 0 || lImageRow >= usHeight || lImageCol < 0 || lImageCol >= usWidth)
			{
				sFCol++;
				continue;
			}
				

			dbFilterOutput += dbFilterCoeff[sFRow * sFilterW + sFCol] * (float)pSrcBuff[lImageRow * lLineWidth + lImageCol];

			sFCol++;
		}

		//The filtered values might be out of range, either be negative or bigger than 255. We can't show them.
		if (dbFilterOutput < 0.0)
			dbFilterOutput = 0.0;
		else if (dbFilterOutput >= 255.0)
			dbFilterOutput = 255.0;

		pDstBuff[lRow * lLineWidth + lCol] = dbFilterOutput; //R

		lCol++;
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);
	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);

	dbTime = ((double)lEnd - (double)lStart) / ((double)lFreq);
	dbTime *= 1e3; //ms

	std::cout << "Time with Raw 3x3 filter: " << dbTime << " ms" << std::endl;

	delete pSrcBuff;
	delete pDstBuff;
}
