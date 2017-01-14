#pragma once
#include "ColorTransform.h"

using namespace GCL;

class ColorTransNative
{
	ColorTransform * colormodel;
	int width = NULL;
	int height = NULL;

public:
	ColorTransNative(int _width, int _height);
	~ColorTransNative();

	int Native_ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid);
};