#pragma once
#include "ColorTransform.h"

using namespace GCL;

class ColorTransNative
{
	ColorTransform * colormodel;
	int width = NULL;
	int height = NULL;
	int deviceid = 0;
public:
	ColorTransNative(int _width, int _height, int _deviceid);
	~ColorTransNative();

	int Native_ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32);

	int Native_ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32);
};