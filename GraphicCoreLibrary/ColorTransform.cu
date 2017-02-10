#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_math.h>

#include "ColorTransform.cuh"

extern "C"
int YV12toARGB32(unsigned char* d_YV12, unsigned char* d_RGBA32, int width, int height, int deviceid)
{
	cudaError_t cudaStatus;
	
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	dim3 block(32, 8);
	int gridx = (width + 2 * block.x - 1) / (2 * block.x);
	int gridy = (height + 2 * block.y - 1) / (2 * block.y);
	dim3 grid(gridx, gridy);
	
	YV12ToARGB_FourPixel << <grid, block >> >(d_YV12, (unsigned int*)d_RGBA32, width, height);
	
	cudaStatus = cudaGetLastError();
	
	return cudaStatus;
}