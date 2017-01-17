#include "ColorTransform.h"
#include <windows.h>  

extern "C"
int gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads, int deviceid);

extern "C"
int ARGB32toUINT32(unsigned char* pARGB, uint* pUINT, int width, int height, int deviceid);

extern "C"
int logAve(float* logAveImage, uint* rgbaImage, uint* BlurImage1, uint* BlurImage2, uint* BlurImage3, int Width, int Height, int deviceid);

extern "C"
int h_Rescale(unsigned char* reScaledImage, float* logAveImage, float max1, float min1, float max2, float min2, float max3, float min3, int Width, int Height, int deviceid);


namespace GCL
{
	ColorTransform::ColorTransform(int _width, int _height)
	{
		width = _width;
		height = _height;
		size = width * height;
		/*cudaHostAlloc((void**)&h_rgba32, width * height * 4 * sizeof(unsigned char), cudaHostAllocWriteCombined | cudaHostAllocMapped);
		cudaHostAlloc((void**)&h_yv12, width * height * 3 / 2 * sizeof(unsigned char), cudaHostAllocMapped);*/
		/*cudaHostGetDevicePointer(&d_rgba32, h_RGBA32, 0);
		cudaHostGetDevicePointer(&d_yv12, h_YV12, 0);*/
		cudaMalloc((void**)&d_yv12, size * 3 / 2 * sizeof(unsigned char));//input
		cudaMalloc((void**)&d_o_rgba32, size * 4 * sizeof(unsigned char));//fake input
		cudaMalloc((void**)&d_rgba32, size * 4 * sizeof(unsigned char));//output
		//inter vars
		cudaMalloc((void **)&d_img, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_temp1, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_temp2, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_temp3, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_result1, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_result2, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_result3, size * sizeof(unsigned int));
		cudaMalloc((void **)&d_logave, sizeof(float)* size * 3);
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
		cudaFree(d_img);
		cudaFree(d_temp1);
		cudaFree(d_temp2);
		cudaFree(d_temp3);
		cudaFree(d_result1);
		cudaFree(d_result2);
		cudaFree(d_result3);
		cudaFree(d_logave);
	}

	int ColorTransform::ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid)
	{
		int result;
		//memcpy(h_yv12, h_YV12, width * height * 3 / 2 * sizeof(unsigned char));
		cudaMemcpy(d_yv12, h_YV12, width * height * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
		//cudaDeviceSynchronize();
		cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//memcpy(h_RGBA32, h_rgba32, width * height * 4 * sizeof(unsigned char));
		return result;
	}

	int ColorTransform::ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32, int deviceid)
	{
		LARGE_INTEGER Freq;
		LARGE_INTEGER start;
		LARGE_INTEGER end;
		QueryPerformanceFrequency(&Freq);
		int result;
		//QueryPerformanceCounter(&start);
		/*cudaHostAlloc((void**)&h_RGBA32, width*height * 4 * sizeof(unsigned char), cudaHostAllocMapped);
		cudaHostGetDevicePointer(&d_rgba32, h_RGBA32, 0);*/
		cudaMemcpy(d_yv12, h_YV12, width * height * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
		result = ARGB32toUINT32(d_rgba32, d_img, width, height, deviceid);
		result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
		result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
		result = h_Rescale(d_rgba32, d_logave, 1.0f, -1.3f, 0.95f, -1.25f, 0.85f, -1.1f, width, height, deviceid);
		//cudaDeviceSynchronize();
		cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//delete[] d_results;
		//QueryPerformanceCounter(&end);
		//printf("execute time: %lld\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);
		return result;
	}

	int ColorTransform::ColorTrans_RetineX(unsigned char* h_o_RGBA32, unsigned char* h_RGBA32, int deviceid)
	{
		int result;
		cudaMemcpy(d_o_rgba32, h_o_RGBA32, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = ARGB32toUINT32(d_o_rgba32, d_img, width, height, deviceid);
		result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
		result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
		result = h_Rescale(d_rgba32, d_logave, 1.0f, -1.3f, 0.95f, -1.25f, 0.85f, -1.1f, width, height, deviceid);
		cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		return result;
	}

}

