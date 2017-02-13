#pragma once
#ifndef COLORTRANSFORM_H
#define COLORTRANSFORM_H

#include <stdio.h>



#include <helper_cuda.h>
#include <helper_timer.h>
#include <helper_math.h>
#include "cublas_v2.h"

extern "C"
int YV12toARGB32(unsigned char* d_YV12, unsigned char* d_RGBA32, int width, int height, int deviceid);

namespace GCL
{
	class ColorTransform
	{
		int width = NULL;
		int height = NULL;
		int deviceid = 0;
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
		//************MaxMin Value
		float max1 = NULL;
		float max2 = NULL;
		float max3 = NULL;
		float min1 = NULL;
		float min2 = NULL;
		float min3 = NULL;
		//*************
		cublasHandle_t handle;
		cublasStatus_t stat;

		void GetManMinValue(float * logAveImage);
		int ColorTrans_YV12toARGB32_CPU(unsigned char *yv12, unsigned char *rgba32);
	public:
		ColorTransform(int _width, int _height, int _deviceid);

		~ColorTransform();

		int ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32);

		int ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32);

		int ColorTrans_RetineX(unsigned char* h_o_RGBA32, unsigned char* h_RGBA32);

		int ResetDeviceID(int _deviceid);

		int GetCurrentDeviceID();
	};
}

#endif // COLORTRANSFORM_H