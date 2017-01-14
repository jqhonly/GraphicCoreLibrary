#pragma once
#include <stdio.h>
#include <helper_math.h>
#include <helper_cuda.h>
#include <helper_timer.h>

extern "C"
int YV12toARGB32(unsigned char* d_YV12, unsigned char* d_RGBA32, int width, int height, int deviceid);

namespace GCL
{
	class ColorTransform
	{
		int width = NULL;
		int height = NULL;
		unsigned char* d_rgba32 = nullptr;
		unsigned char* d_yv12 = nullptr;
		/*unsigned char* h_rgba32 = nullptr;
		unsigned char* h_yv12 = nullptr;*/
	public:
		ColorTransform(int _width, int _height);
		~ColorTransform();
		int ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid);
	};
}