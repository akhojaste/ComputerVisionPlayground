///This is the ImageProcessing header file.
///This class is responsible for image I/O and processing.

#pragma once

#include <math.h>
#include <windef.h>

class ImageProcessing
{
public:
	ImageProcessing();
	~ImageProcessing();

	void Smooth(BYTE *pbOutBuffer, const BYTE *pbInBuffer, const unsigned short usHeight, const unsigned short usWidth, unsigned short usBytesPerPixel);
};

