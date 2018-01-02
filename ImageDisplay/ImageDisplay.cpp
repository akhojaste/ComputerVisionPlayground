#include "stdafx.h"
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <stdio.h>

#include "ImageProcessing.h"

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

#define BUTTON_IDENTIFIER 1

#define IP_BPP 4 //Bytes per Pixel

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
void ReadAndProcessImage();
void GetThePixelFormat(Gdiplus::PixelFormat pixelFormat);

VOID OnPaint(HDC hdc)
{
	Graphics graphics(hdc);
	//Pen      pen(Color(255, 0, 0, 255));
	//graphics.DrawLine(&pen, 0, 0, 200, 100); 

	// Load the image and show it on the screen
	wchar_t *filename = L"F:\\data\\cameraman.png";
	Gdiplus::Image * pImage = Gdiplus::Image::FromFile(filename);
	const Gdiplus::Rect rectImage = { 0, 0, (INT)pImage->GetWidth() , (INT)pImage->GetHeight() };
	graphics.DrawImage(pImage, rectImage);
}

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

INT WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, PSTR, INT iCmdShow)
{
	HWND                hWnd;
	MSG                 msg;
	WNDCLASS            wndClass;
	GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR           gdiplusToken;

	// Initialize GDI+.
	GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	wndClass.style = CS_HREDRAW | CS_VREDRAW;
	wndClass.lpfnWndProc = WndProc;
	wndClass.cbClsExtra = 0;
	wndClass.cbWndExtra = 0;
	wndClass.hInstance = hInstance;
	wndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	wndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wndClass.lpszMenuName = NULL;
	wndClass.lpszClassName = TEXT("ImageDisplay");

	RegisterClass(&wndClass);

	//Creating the MainWindow
	hWnd = CreateWindow(
		TEXT("ImageDisplay"),   // window class name
		TEXT("ImageDisplay"),  // window caption
		WS_OVERLAPPEDWINDOW,      // window style
		CW_USEDEFAULT,            // initial x position
		CW_USEDEFAULT,            // initial y position
		600,            // initial x size
		800,            // initial y size
		NULL,                     // parent window handle
		NULL,                     // window menu handle
		hInstance,                // program instance handle
		NULL);                    // creation parameters

	//Adding a button inside the MainWindow
	HWND hwndButton = CreateWindow(
		L"BUTTON",  // Predefined class; Unicode assumed 
		L"Filter",      // Button text 
		WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,  // Styles 
		300,         // x position 
		10,         // y position 
		100,        // Button width
		50,        // Button height
		hWnd,     // Parent window
		(HMENU)BUTTON_IDENTIFIER,
		(HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE),
		NULL);      // Pointer not needed.

	ShowWindow(hWnd, iCmdShow);
	UpdateWindow(hWnd);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	GdiplusShutdown(gdiplusToken);
	return msg.wParam;
}  // WinMain

LRESULT CALLBACK WndProc(HWND hWnd, UINT message,
	WPARAM wParam, LPARAM lParam)
{
	HDC          hdc;
	PAINTSTRUCT  ps;
	TCHAR greeting[] = _T("Original Image");

	switch (message)
	{
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		OnPaint(hdc);

		TextOut(hdc,
			5, 5,
			greeting, _tcslen(greeting));

		EndPaint(hWnd, &ps);
		return 0;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;

	case WM_COMMAND:
	{
		switch (LOWORD(wParam))
		{
			case BUTTON_IDENTIFIER:
			{
				//User click on the button
				if (HIWORD(wParam) == BN_CLICKED)
				{
					//UINT nButton = (UINT)LOWORD(wParam);
					//HWND hButtonWnd = (HWND)lParam;

					//Do the image processing
					ReadAndProcessImage();

					//Attach it to the view
					HDC hdc = GetDC(hWnd);
					Graphics graphics(hdc);
					wchar_t *filename = L"F:\\data\\cameramanFiltered.png";
					Gdiplus::Image * pImage = Gdiplus::Image::FromFile(filename);
					const Gdiplus::Rect rectImage = { 0, 280, (INT)pImage->GetWidth(), (INT)pImage->GetHeight() };
					graphics.DrawImage(pImage, rectImage);

					TCHAR greeting[] = _T("Filtered Image");

					TextOut(hdc,
						rectImage.GetLeft(), rectImage.GetTop() + 5,
						greeting, _tcslen(greeting));

				}
			}
			break;
		}
	}
	break;
		
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
} // WndProc

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;          // number of image encoders
	UINT  size = 0;         // size of the image encoder array in bytes

	ImageCodecInfo* pImageCodecInfo = NULL;

	GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}

	free(pImageCodecInfo);
	return -1;  // Failure
}

void ReadAndProcessImage()
{
	// LOAD the image and show it on the screen
	wchar_t *filename = L"F:\\data\\cameraman.png";
	Gdiplus::Image * pImage = Gdiplus::Image::FromFile(filename);
	const Gdiplus::Rect rectImage = { 0, 0, (INT)pImage->GetWidth(), (INT)pImage->GetHeight() };
	Gdiplus::PixelFormat pixelFormat = pImage->GetPixelFormat();

	int iHeight = pImage->GetHeight();
	int iWidth = pImage->GetWidth();

	//Now lets try and get the fucking pixel data from this stupid Gdiplus
	//BYTE *pbImageBuffer = new BYTE[iHeight * iWidth * IP_BPP]; 
	Gdiplus::Bitmap bmp(filename);
	
	Gdiplus::BitmapData ImageData;
	Gdiplus::Status status = bmp.LockBits(&rectImage, Gdiplus::ImageLockModeRead, pixelFormat, &ImageData);

	BYTE *pSrc = (BYTE*)ImageData.Scan0;

	//for (int i = 0; i < iHeight; ++i)
	//{
	//	memcpy(pbImageBuffer, pSrc, IP_BPP * iWidth);
	//	pSrc += ImageData.Stride; //This is the gap between each line of the image in bytes (here image is row wise)
	//	pbImageBuffer += ImageData.Stride;
	//	
	//	//for (int j = 0; j < iWidth; ++j)
	//	//{
	//	//	pbImageBuffer[i * iWidth * IP_BPP + j * IP_BPP] = pSrc[i * iWidth * IP_BPP + j * IP_BPP];
	//	//}
	//}
	//bmp.UnlockBits(&ImageData);

	//pbImageBuffer -= iHeight * ImageData.Stride;

	//IMAGE PROCESSING: Do some processing on the buffer and then return it
	ImageProcessing IP;
	BYTE *pbImageBufferSmoothed = new BYTE[iHeight * iWidth * IP_BPP]; //I'm just interested in the gray scale for now

	IP.Smooth(pbImageBufferSmoothed, pSrc, iHeight, iWidth, IP_BPP);

	//SAVE the image to the file
	Gdiplus::Bitmap bmpSave(iWidth, iHeight, PixelFormat32bppARGB);
	Gdiplus::BitmapData bmpData;
	status = bmpSave.LockBits(&rectImage, Gdiplus::ImageLockModeWrite, PixelFormat32bppARGB, &bmpData);

	BYTE *pSavedImageBuffer = (BYTE*)bmpData.Scan0;

	//for (int i = 0; i < iHeight; ++i)
	//{
	//	memcpy(pSavedImageBuffer, pbImageBufferSmoothed + i * bmpData.Stride, bmpData.Stride);
	//	pSavedImageBuffer += bmpData.Stride;
	//}

	memcpy(pSavedImageBuffer, pbImageBufferSmoothed, iHeight * iWidth * IP_BPP);

	bmpSave.UnlockBits(&bmpData);

	CLSID pngClsid;
	GetEncoderClsid(L"image/png", &pngClsid);
	status = bmpSave.Save(L"F:\\data\\cameramanFiltered.png", &pngClsid, nullptr);



	//Free the buffer
	delete pbImageBufferSmoothed;
	//delete pbImageBuffer;
}

void GetThePixelFormat(Gdiplus::PixelFormat pixelFormat)
{
	switch (pixelFormat)
	{
	case  PixelFormatIndexed:
	{
		int k = 1;
		break;
	}

	case PixelFormatGDI:
	{
		int k = 1;
		break;
	}

	case PixelFormatAlpha:
	{
		int k = 1;
		break;

	}

	case    PixelFormatPAlpha:
	{
		int k = 1;
		break;
	}


	case    PixelFormatExtended:

	{
		int k = 1;
		break;
	}

	case     PixelFormatCanonical:
	{
		int k = 1;
		break;
	}


	case PixelFormatUndefined:
	{
		int k = 1;
		break;
	}

	case     PixelFormat1bppIndexed:
	{
		int k = 1;
		break;
	}

	case  PixelFormat4bppIndexed:
	{
		int k = 1;
		break;
	}

	case PixelFormat8bppIndexed:
	{
		int k = 1;
		break;
	}

	case PixelFormat16bppGrayScale:
	{
		int k = 1;
		break;
	}

	case PixelFormat16bppRGB555:
	{
		int k = 1;
		break;
	}

	case    PixelFormat16bppRGB565:
	{
		int k = 1;
		break;
	}

	case     PixelFormat16bppARGB1555:
	{
		int k = 1;
		break;
	}

	case     PixelFormat24bppRGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat32bppRGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat32bppARGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat32bppPARGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat48bppRGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat64bppARGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat64bppPARGB:
	{
		int k = 1;
		break;
	}

	case     PixelFormat32bppCMYK:
	{
		int k = 1;
		break;
	}

	case     PixelFormatMax:
	{
		int k = 1;
		break;
	}

	default:
		break;
	}
}
