#include <stdio.h>
#include <helper_math.h>
#include "GpuBitmap.h"

namespace GCL
{
	GpuBitmap::GpuBitmap(unsigned char* input, int _width, int _height, int _depth)
	{
		width = _width;
		height = _height;
		depth = _depth;
		cudaMalloc((void**)&d_GpuData, width*height*depth * sizeof(unsigned char));
		cudaMalloc((void**)&d_float_GpuData, width*height*depth * sizeof(float));
		cudaMemcpy(d_GpuData, input, width*height*depth * sizeof(unsigned char), cudaMemcpyHostToDevice);
	}

	GpuBitmap::GpuBitmap(CpuBitmap cpuimage)
	{
		width = cpuimage.width;
		height = cpuimage.height;
		depth = cpuimage.depth;
		cudaMalloc((void**)&d_GpuData, width*height*depth * sizeof(unsigned char));
		cudaMalloc((void**)&d_float_GpuData, width*height*depth * sizeof(float));
		cudaMemcpy(d_GpuData, cpuimage.h_CpuData, width*height*depth * sizeof(unsigned char), cudaMemcpyHostToDevice);
	}

	GpuBitmap::~GpuBitmap()
	{
		cudaFree(d_GpuData);
		cudaFree(d_float_GpuData);
	}

}