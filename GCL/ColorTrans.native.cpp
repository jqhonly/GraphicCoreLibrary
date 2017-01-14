#include "stdafx.h"

using namespace GCL;

ColorTransNative::ColorTransNative(int _width, int _height)
{
	width = _width;
	height = _height;
	colormodel = new ColorTransform(width, height);
}

ColorTransNative::~ColorTransNative()
{
	if (colormodel)
	{
		delete colormodel;
		colormodel = nullptr;
	}
}

int ColorTransNative::Native_ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid)
{
	return colormodel->ColorTrans_YV12toARGB32(h_YV12, h_RGBA32, deviceid);
}



