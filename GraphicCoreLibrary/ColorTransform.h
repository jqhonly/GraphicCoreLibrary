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
		unsigned char* d_o_rgba32 = nullptr;//fake test
		unsigned char* d_yv12 = nullptr;
		//************Multi-Scale RetineX
		float sigma1 = 20.0f;
		float sigma2 = 100.0f;
		float sigma3 = 300.0f;
		int order = 0;
		int nthreads = 64;  // number of threads per block
		uint* d_result1 = nullptr;
		uint* d_result2 = nullptr;
		uint* d_result3 = nullptr;
		uint* d_temp1 = nullptr;
		uint* d_temp2 = nullptr;
		uint* d_temp3 = nullptr;
		uint* d_img = nullptr;
		float* d_logave = nullptr;
		//temp host data
		float* h_logave = nullptr;
		unsigned int size;
		//************MaxMin VALUE
		float max1 = NULL;
		float max2 = NULL;
		float max3 = NULL;
		float min1 = NULL;
		float min2 = NULL;
		float min3 = NULL;

		void GetManMinValue(float * logAveImage);
	public:
		ColorTransform(int _width, int _height);
		~ColorTransform();
		int ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid);

		int ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid);

		int ColorTrans_RetineX(unsigned char* h_o_RGBA32, unsigned char* h_RGBA32, int deviceid);
	};
}