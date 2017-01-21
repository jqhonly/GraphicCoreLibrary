#include "iostream"
#include "CpuBitmap.h"

namespace GCL
{
	CpuBitmap::CpuBitmap(unsigned char* input, int _width, int _height, int _depth)
	{
		width = _width;
		height = _height;
		depth = _depth;
		memcpy(h_CpuData, input, width*height*depth * sizeof(unsigned char));
		h_float_CpuData = (float*)malloc(width*height*depth * sizeof(float));
	}

	CpuBitmap::~CpuBitmap()
	{
		delete[] h_CpuData;
		delete[] h_float_CpuData;
	}

}