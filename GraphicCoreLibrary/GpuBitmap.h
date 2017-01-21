#pragma once
#include "CpuBitmap.h"

namespace GCL
{
	class GpuBitmap
	{
	public:
		unsigned char* d_GpuData;
		float * d_float_GpuData;
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