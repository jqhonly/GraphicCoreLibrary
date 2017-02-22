#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include "cublas_v2.h"

#include "recursiveGaussian_kernel.cuh"

#define USE_SIMPLE_FILTER 0

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

/*
Transpose a 2D array (see SDK transpose example)
*/
extern "C"
void transpose(uint *d_src, uint *d_dest, uint width, int height)
{
	dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
	dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
	d_transpose << < grid, threads >> >(d_dest, d_src, width, height);
	getLastCudaError("Kernel execution failed");
}

/*
Perform Gaussian filter on a 2D image using CUDA

Parameters:
d_src  - pointer to input image in device memory
d_dest - pointer to destination image in device memory
d_temp - pointer to temporary storage in device memory
width  - image width
height - image height
sigma  - sigma of Gaussian
order  - filter order (0, 1 or 2)
*/

// 8-bit RGBA version
extern "C"
int gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads, int deviceid)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	// compute filter coefficients
	const float
		nsigma = sigma < 0.1f ? 0.1f : sigma,
		alpha = 1.695f / nsigma,
		ema = (float)std::exp(-alpha),
		ema2 = (float)std::exp(-2 * alpha),
		b1 = -2 * ema,
		b2 = ema2;

	float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;

	switch (order)
	{
	case 0:
	{
		const float k = (1 - ema)*(1 - ema) / (1 + 2 * alpha*ema - ema2);
		a0 = k;
		a1 = k*(alpha - 1)*ema;
		a2 = k*(alpha + 1)*ema;
		a3 = -k*ema2;
	}
	break;

	case 1:
	{
		const float k = (1 - ema)*(1 - ema) / ema;
		a0 = k*ema;
		a1 = a3 = 0;
		a2 = -a0;
	}
	break;

	case 2:
	{
		const float
			ea = (float)std::exp(-alpha),
			k = -(ema2 - 1) / (2 * alpha*ema),
			kn = (-2 * (-1 + 3 * ea - 3 * ea*ea + ea*ea*ea) / (3 * ea + 1 + 3 * ea*ea + ea*ea*ea));
		a0 = kn;
		a1 = -kn*(1 + k*alpha)*ema;
		a2 = kn*(1 - k*alpha)*ema;
		a3 = -kn*ema2;
	}
	break;

	default:
		fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
		return -1;
	}

	coefp = (a0 + a1) / (1 + b1 + b2);
	coefn = (a2 + a3) / (1 + b1 + b2);

	// process columns
#if USE_SIMPLE_FILTER
	d_simpleRecursive_rgba << < iDivUp(width, nthreads), nthreads >> >(d_src, d_temp, width, height, ema);
#else
	d_recursiveGaussian_rgba << < iDivUp(width, nthreads), nthreads >> >(d_src, d_temp, width, height, a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
	getLastCudaError("Kernel execution failed");

	transpose(d_temp, d_dest, width, height);
	getLastCudaError("transpose: Kernel execution failed");

	// process rows
#if USE_SIMPLE_FILTER
	d_simpleRecursive_rgba << < iDivUp(height, nthreads), nthreads >> >(d_dest, d_temp, height, width, ema);
#else
	d_recursiveGaussian_rgba << < iDivUp(height, nthreads), nthreads >> >(d_dest, d_temp, height, width, a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
	getLastCudaError("Kernel execution failed");
	
	transpose(d_temp, d_dest, height, width);
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

extern "C"
int ARGB32toUINT32(unsigned char* pARGB, uint* pUINT, int width, int height, int deviceid)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	const dim3 blockSize(24, 24, 1);
	const dim3 gridSize((width / 16), (height / 16), 1);
	uchar_to_uint << <gridSize, blockSize >> >(pARGB, pUINT, width, height);
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

extern "C"
int logAve(float* logAveImage, uint* rgbaImage, uint* BlurImage1, uint* BlurImage2, uint* BlurImage3, int Width, int Height, int deviceid)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	const dim3 blockSize(24, 24, 1);
	const dim3 gridSize((Width / 16), (Height / 16), 1);
	LogAve << < gridSize, blockSize >> >(logAveImage, rgbaImage, BlurImage1, BlurImage2, BlurImage3, Width, Height);
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

extern "C"
int h_Rescale(unsigned char* reScaledImage, float* logAveImage, float max1, float min1, float max2, float min2, float max3, float min3, int Width, int Height, int deviceid)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	const dim3 blockSize(24, 24, 1);
	const dim3 gridSize((Width / 16), (Height / 16), 1);
	Rescale << < gridSize, blockSize >> > (reScaledImage, logAveImage, max1, min1, max2, min2, max3, min3, Width, Height);
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

extern "C"
int h_GetManMinValue(float* d_logave, float &max1, float &min1, float &max2, float &min2, float &max3, float &min3, int Width, int Height, int deviceid)
{
	cublasHandle_t handle;
	cublasStatus_t stat;
	cudaError_t cudaStatus;
	stat = cublasCreate(&handle);

	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	const dim3 blockSize(24, 24, 1);
	const dim3 gridSize((Width / 16), (Height / 16), 1);

	float* channel1 = nullptr;
	float* channel2 = nullptr;
	float* channel3 = nullptr;
	float* one_const = nullptr;
	cudaMalloc((void **)&channel1, sizeof(float)* (Width * Height + 1));
	cudaMalloc((void **)&channel2, sizeof(float)* (Width * Height + 1));
	cudaMalloc((void **)&channel3, sizeof(float)* (Width * Height + 1));
	cudaMalloc((void **)&one_const, sizeof(float)* (Width * Height + 1));
	float sum1;
	float sum2;
	float sum3;
	SlicetoBLAS << < gridSize, blockSize >> > (d_logave, channel1, channel2, channel3, one_const, Width, Height);

	cublasSdot(handle, Width*Height, channel1, 1, one_const, 1, &sum1);
	cublasSdot(handle, Width*Height, channel2, 1, one_const, 1, &sum2);
	cublasSdot(handle, Width*Height, channel3, 1, one_const, 1, &sum3);

	float mean1 = sum1 / (Width * Height);
	float mean2 = sum2 / (Width * Height);
	float mean3 = sum3 / (Width * Height);

	PrepareVal << < gridSize, blockSize >> > (channel1, mean1, Width, Height);
	PrepareVal << < gridSize, blockSize >> > (channel2, mean2, Width, Height);
	PrepareVal << < gridSize, blockSize >> > (channel3, mean3, Width, Height);

	float var1;
	float var2;
	float var3;

	cublasSdot(handle, Width*Height, channel1, 1, channel1, 1, &var1);
	cublasSdot(handle, Width*Height, channel2, 1, channel2, 1, &var2);
	cublasSdot(handle, Width*Height, channel3, 1, channel3, 1, &var3);

	var1 = sqrtf(var1 / (Width * Height - 1));
	var2 = sqrtf(var2 / (Width * Height - 1));
	var3 = sqrtf(var3 / (Width * Height - 1));

	max1 = mean1 + 2 * var1;
	max2 = mean2 + 2 * var2;
	max3 = mean3 + 2 * var3;
	min1 = mean1 - 2 * var1;
	min2 = mean2 - 2 * var2;
	min3 = mean3 - 2 * var3;

	cudaFree(channel1);
	cudaFree(channel2);
	cudaFree(channel3);
	cudaFree(one_const);
	stat = cublasDestroy(handle);

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}