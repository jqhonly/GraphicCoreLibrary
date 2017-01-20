#pragma once
#include "CpuBitmap.h"

namespace GCL
{
	class GpuBitmap
	{
	public:
		unsigned char* d_CpuData;
		float * d_float_CpuData;
		int width;
		int height;
		int depth;
		GpuBitmap();
		GpuBitmap(CpuBitmap cpuimage);
		GpuBitmap(unsigned char* input, int _width, int _height, int _depth);
		~GpuBitmap();

	public:

	};
}