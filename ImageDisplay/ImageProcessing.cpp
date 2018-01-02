#include "stdafx.h" //STUPID, this should be included first before any other including
#include "ImageProcessing.h"


ImageProcessing::ImageProcessing()
{
	
}


ImageProcessing::~ImageProcessing()
{
}

void ImageProcessing::Smooth(BYTE *pbOutBuffer, const BYTE *pbInBuffer, const unsigned short usHeight, const unsigned short usWidth, unsigned short usBytesPerPixel)
{

	//A 5x5 Gaussian smotthing filter
	//const short sFilterW = 5;
	//const short sFilterH = 5;
	//const short sFilterSize = 25;
	//double dbFilterCoeff[sFilterSize] = { 0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944,
	//								 0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223,
	//								 0.0383275593839039, 0.0557662698468495, 0.0631914624102647, 0.0557662698468495, 0.0383275593839039,
	//								 0.0338239524399223, 0.0492135604085414, 0.0557662698468495, 0.0492135604085414, 0.0338239524399223,
	//								 0.0232468398782944, 0.0338239524399223, 0.0383275593839039, 0.0338239524399223, 0.0232468398782944 };

	//A 3x3 Sharpening filter
	const short sFilterW = 3;
	const short sFilterH = 3;
	const short sFilterSize = 9;
	double dbFilterCoeff[sFilterSize] = { -0.0833333333333333, -0.0833333333333333, -0.0833333333333333, 
										  -0.0833333333333333, 1.66666666666667, -0.0833333333333333, 
										  -0.0833333333333333, -0.0833333333333333, -0.0833333333333333 };

	


	long long lStart, lEnd, lFreq; 
	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);

	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);

	BYTE *pline = nullptr;
	long lRow = 0, lCol = 0;

	long lLineWidth = (long) usWidth * usBytesPerPixel;

	short sFRow = 0, sFCol = 0;

//#pragma omp parallel for //why this does not make it better? because the buffer is shared between thread and this will actually add the overhead of waiting for the buffer to get released.
	for (int i = 0; i < usHeight * usWidth; ++i)
	{
		
		lRow = i / usWidth;
		lCol = i %usWidth;
	
		double dbFilterOutput = 0.0;
		
		for (int iFilterCount = 0; iFilterCount < sFilterSize; ++iFilterCount)
		{
			sFRow = iFilterCount / sFilterW;
			sFCol = iFilterCount % sFilterW;

			long lImageRow = lRow + sFRow - sFilterH / 2.0;
			long lImageCol = lCol + sFCol - sFilterW / 2.0;

			if (lImageRow < 0 || lImageRow >= usWidth || lImageCol < 0 || lImageCol >= usHeight)
				continue;

			dbFilterOutput += dbFilterCoeff[sFRow * sFilterW + sFCol] * (float)pbInBuffer[lImageRow * lLineWidth + lImageCol * usBytesPerPixel];
		}

		//The filtered values might be out of range, either be negative or bigger than 255. We can't show them.
		if (dbFilterOutput < 0.0)
			dbFilterOutput = 0.0;
		else if (dbFilterOutput >= 255.0)
			dbFilterOutput = 255.0;

		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 0] = dbFilterOutput; //R
		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 1] = dbFilterOutput; //G
		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 2] = dbFilterOutput; //B
		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 3] = 255; //Alpha
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);

	double dbTime = ((double)lEnd - (double)lStart) / (double)lFreq;

	dbTime *= 1000.0; //ms
}



//1D with for loop for filters
////{
////	double m_dbGaussainCoeff[9] = { 0.077847, 0.123317, 0.077847, 0.123317, 0.195346, 0.123317, 0.077847, 0.123317, 0.077847 };
////
////	short sFilterSize = 9;
////	short sFilterW = 3;
////	short sFilterH = 3;
////
////	long long lStart, lEnd, lFreq;
////	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);
////
////	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);
////
////	BYTE *pline = nullptr;
////	long lRow = 0, lCol = 0;
////
////	long lLineWidth = (long)usWidth * usBytesPerPixel;
////
////	short sFRow = 0, sFCol = 0;
////
////	//#pragma omp parallel for
////	for (int i = 0; i < usHeight * usWidth; ++i)
////	{
////
////		lRow = i / usWidth;
////		lCol = i %usWidth;
////
////		//For now, lets just skip these pixels
////		if (lRow < 2 || lRow >= (usHeight - 2) || lCol < 2 || lCol >= (usWidth - 2))
////			continue;
////
////		double dbFilterOutput = 0.0;
////
////		for (int iFilterCount = 0; iFilterCount < sFilterSize; ++iFilterCount)
////		{
////			sFRow = iFilterCount / sFilterW;
////			sFCol = iFilterCount % sFilterW;
////
////			long lImageRow = lRow + sFRow - sFilterH / 2;
////			long lImageCol = lCol + sFCol - sFilterW / 2;
////
////			dbFilterOutput += m_dbGaussainCoeff[sFRow * sFilterW + sFCol] * (float)pbInBuffer[lImageRow * lLineWidth + lImageCol * usBytesPerPixel];
////		}
////
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 0] = dbFilterOutput; //R
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 1] = dbFilterOutput; //G
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 2] = dbFilterOutput; //B
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 3] = 255; //Alpha
////	}
////
////	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);
////
////	double dbTime = ((double)lEnd - (double)lStart) / (double)lFreq;
////
////	dbTime *= 1000.0; //ms
////}



///1D works
////{
////	double m_dbGaussainCoeff[3][3] = { { 0.077847, 0.123317, 0.077847 },
////	{ 0.123317, 0.195346, 0.123317 },
////	{ 0.077847, 0.123317, 0.077847 }
////	};
////
////	long long lStart, lEnd, lFreq;
////	QueryPerformanceCounter((LARGE_INTEGER*)&lStart);
////
////	QueryPerformanceFrequency((LARGE_INTEGER*)&lFreq);
////
////	BYTE *pline = nullptr;
////	long lRow = 0, lCol = 0;
////
////	long lLineWidth = (long)usWidth * usBytesPerPixel;
////
////	//#pragma omp parallel for
////	for (int i = 0; i < usHeight * usWidth; ++i)
////	{
////
////		lRow = i / usWidth;
////		lCol = i %usWidth;
////
////		//For now, lets just skip these pixels
////		if (lRow < 2 || lRow >= (usHeight - 2) || lCol < 2 || lCol >= (usWidth - 2))
////			continue;
////
////		double dbRow1 = m_dbGaussainCoeff[0][0] * (float)pbInBuffer[(lRow - 1) * lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[0][1] * (float)pbInBuffer[(lRow - 1) * lLineWidth + lCol * usBytesPerPixel] +
////			m_dbGaussainCoeff[0][2] * (float)pbInBuffer[(lRow - 1) * lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////		double dbRow2 = m_dbGaussainCoeff[1][0] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[1][1] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol)* usBytesPerPixel] +
////			m_dbGaussainCoeff[1][2] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////		double dbRow3 = m_dbGaussainCoeff[2][0] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[2][1] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol)* usBytesPerPixel] +
////			m_dbGaussainCoeff[2][2] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////
////
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 0] = dbRow1 + dbRow2 + dbRow3; //R
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 1] = dbRow1 + dbRow2 + dbRow3; //G
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 2] = dbRow1 + dbRow2 + dbRow3; //B
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 3] = 255; //Alpha
////	}
////
////	QueryPerformanceCounter((LARGE_INTEGER*)&lEnd);
////
////	double dbTime = ((double)lEnd - (double)lStart) / (double)lFreq;
////
////	dbTime *= 1000.0; //ms
////}



//2D, Not Optimum
//WORKS!
////BYTE *pline = nullptr;
////long lRow = 0, lCol = 0;
////
////long lLineWidth = (long)usWidth * usBytesPerPixel;
////
////for (int lRow = 0; lRow < usHeight; ++lRow)
////{
////
////	for (int lCol = 0; lCol < usWidth; ++lCol)
////	{
////		if (lRow < 2 || lRow >= (usHeight - 2) || lCol < 2 || lCol >= (usWidth - 2))
////			continue;
////
////		double dbRow1 = m_dbGaussainCoeff[0][0] * (float)pbInBuffer[(lRow - 1) * lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[0][1] * (float)pbInBuffer[(lRow - 1) * lLineWidth + lCol * usBytesPerPixel] +
////			m_dbGaussainCoeff[0][2] * (float)pbInBuffer[(lRow - 1) * lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////		double dbRow2 = m_dbGaussainCoeff[1][0] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[1][1] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol)* usBytesPerPixel] +
////			m_dbGaussainCoeff[1][2] * (float)pbInBuffer[(lRow)* lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////		double dbRow3 = m_dbGaussainCoeff[2][0] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol - 1) * usBytesPerPixel] +
////			m_dbGaussainCoeff[2][1] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol)* usBytesPerPixel] +
////			m_dbGaussainCoeff[2][2] * (float)pbInBuffer[(lRow + 1) * lLineWidth + (lCol + 1) * usBytesPerPixel];
////
////
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 0] = dbRow1 + dbRow2 + dbRow3; //R
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 1] = dbRow1 + dbRow2 + dbRow3; //G
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 2] = dbRow1 + dbRow2 + dbRow3; //B
////		pbOutBuffer[lRow * lLineWidth + lCol * usBytesPerPixel + 3] = 255; //Alpha
////
////	}
////
////
////
////	//pbOutBuffer[lRow * lLineWidth + lCol * (long)usBytesPerPixel + 0] = lRow;
////	//pbOutBuffer[lRow * lLineWidth + lCol * (long)usBytesPerPixel + 1] = lRow;
////	//pbOutBuffer[lRow * lLineWidth + lCol * (long)usBytesPerPixel + 2] = lRow;
////	//pbOutBuffer[lRow * lLineWidth + lCol * (long)usBytesPerPixel + 3] = 255; //Alpha
////}
