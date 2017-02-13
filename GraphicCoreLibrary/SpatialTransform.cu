#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>

#include "SpatialTransform.cuh"

extern "C"
int Resize_Host(unsigned char*h_src, int srcWidth, int srcHeight, unsigned char *h_dst, int dstWidth, int dstHeight, int deviceid)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(deviceid);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	unsigned char * d_src;
	unsigned char * d_dst;
	cudaMalloc((void**)&d_src, srcWidth * srcHeight * 4 * sizeof(unsigned char));
	cudaMalloc((void**)&d_dst, dstHeight * dstWidth * 4 * sizeof(unsigned char));

	cudaMemcpy(d_src, h_src, srcWidth * srcHeight * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	int uint = 16;//Don't change!
	dim3 grid((dstWidth + uint - 1) / uint, (dstHeight + uint - 1) / uint);
	dim3 block(uint, uint);

	Resize_Device << <grid, block >> >(d_src, srcWidth, srcHeight, d_dst, dstWidth, dstHeight);

	cudaMemcpy(h_dst, d_dst, dstHeight * dstWidth * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_dst);
	cudaFree(d_src);

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}