#include "stdafx.h"

using namespace GCL;

ColorTransNative::ColorTransNative(int _width, int _height, int _deviceid)
{
	width = _width;
	height = _height;
	deviceid = _deviceid;
	colormodel = new ColorTransform(width, height, deviceid);
}

ColorTransNative::~ColorTransNative()
{
	if (colormodel)
	{
		delete colormodel;
		colormodel = nullptr;
	}
}

int ColorTransNative::Native_ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32)
{
	return colormodel->ColorTrans_YV12toARGB32(h_YV12, h_RGBA32);
}

int ColorTransNative::Native_ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32)
{
	return colormodel->ColorTrans_YV12toARGB32_RetineX(h_YV12, h_RGBA32);
}



