#pragma once

#include <ipp.h>
#include <fstream>
#include <iostream>
#include <Windows.h>

class ImageProcessing
{
public:
	ImageProcessing();
	~ImageProcessing();

	//Filter the image using Intel IPP Library
	void ReadAndFilterIPP();

	//Filter the image using the compiled C++ library
	void ReadAndFilterRaw();
};

