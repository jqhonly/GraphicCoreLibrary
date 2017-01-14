#include "ColorTransform.h"
#include <windows.h>  

namespace GCL
{
	ColorTransform::ColorTransform(int _width, int _height)
	{
		width = _width;
		height = _height;
		/*cudaHostAlloc((void**)&h_rgba32, width * height * 4 * sizeof(unsigned char), cudaHostAllocWriteCombined | cudaHostAllocMapped);
		cudaHostAlloc((void**)&h_yv12, width * height * 3 / 2 * sizeof(unsigned char), cudaHostAllocMapped);*/
		/*cudaHostGetDevicePointer(&d_rgba32, h_RGBA32, 0);
		cudaHostGetDevicePointer(&d_yv12, h_YV12, 0);*/
		cudaMalloc((void**)&d_yv12, width * height * 3 / 2 * sizeof(unsigned char));
		cudaMalloc((void**)&d_rgba32, width * height * 4 * sizeof(unsigned char));
	}

	ColorTransform::~ColorTransform()
	{
		/*cudaFreeHost(h_rgba32);
		cudaFreeHost(h_yv12);
		h_rgba32 = nullptr;
		h_yv12 = nullptr;*/
		cudaFree(d_yv12);
		cudaFree(d_rgba32);
		/*d_rgba32 = nullptr;
		d_yv12 = nullptr;*/
	}

	int ColorTransform::ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid)
	{
		LARGE_INTEGER Freq;
		LARGE_INTEGER start;
		LARGE_INTEGER end;
		QueryPerformanceFrequency(&Freq);
		
		//***
		int result;
		//QueryPerformanceCounter(&start);

		
		//memcpy(h_yv12, h_YV12, width * height * 3 / 2 * sizeof(unsigned char));
		cudaMemcpy(d_yv12, h_YV12, width * height * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);

		result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
		
		//cudaDeviceSynchronize();
		cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//memcpy(h_RGBA32, h_rgba32, width * height * 4 * sizeof(unsigned char));
		
		/*QueryPerformanceCounter(&end);
		printf("execute time: %d\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);*/
		return result;
	}
}

