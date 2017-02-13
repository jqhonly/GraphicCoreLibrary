#pragma once
#ifndef _SPATIAL_TRANSFORM_CUH_
#define _SPATIAL_TRANSFORM_CUH_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <helper_cuda.h>
#include <helper_math.h>

__global__ void Resize_Device(unsigned char*d_src, int srcWidth, int srcHeight, unsigned char *d_dst, int dstWidth, int dstHeight)
{
	double srcXf;
	double srcYf;
	int srcX;
	int srcY;
	double u;
	double v;
	int dstOffset;

	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if (x >= dstWidth || y >= dstHeight) 
		return;

	srcXf = x* ((float)srcWidth / dstWidth);
	srcYf = y* ((float)srcHeight / dstHeight);
	srcX = (int)srcXf;
	srcY = (int)srcYf;
	u = srcXf - srcX;
	v = srcYf - srcY;

	//r chanel  
	dstOffset = (y*dstWidth + x) * 4;
	d_dst[dstOffset] = 0;
	d_dst[dstOffset] += (1 - u)*(1 - v)*d_src[(srcY*srcWidth + srcX) * 4];
	d_dst[dstOffset] += (1 - u)*v*d_src[((srcY + 1)*srcWidth + srcX) * 4];
	d_dst[dstOffset] += u*(1 - v)*d_src[(srcY*srcWidth + srcX + 1) * 4];
	d_dst[dstOffset] += u*v*d_src[((srcY + 1)*srcWidth + srcX + 1) * 4];

	//g chanel  
	dstOffset = (y*dstWidth + x) * 4 + 1;
	d_dst[dstOffset] = 0;
	d_dst[dstOffset] += (1 - u)*(1 - v)*d_src[(srcY*srcWidth + srcX) * 4 + 1];
	d_dst[dstOffset] += (1 - u)*v*d_src[((srcY + 1)*srcWidth + srcX) * 4 + 1];
	d_dst[dstOffset] += u*(1 - v)*d_src[(srcY*srcWidth + srcX + 1) * 4 + 1];
	d_dst[dstOffset] += u*v*d_src[((srcY + 1)*srcWidth + srcX + 1) * 4 + 1];

	//b chanel  
	dstOffset = (y*dstWidth + x) * 4 + 2;
	d_dst[dstOffset] = 0;
	d_dst[dstOffset] += (1 - u)*(1 - v)*d_src[(srcY*srcWidth + srcX) * 4 + 2];
	d_dst[dstOffset] += (1 - u)*v*d_src[((srcY + 1)*srcWidth + srcX) * 4 + 2];
	d_dst[dstOffset] += u*(1 - v)*d_src[(srcY*srcWidth + srcX + 1) * 4 + 2];
	d_dst[dstOffset] += u*v*d_src[((srcY + 1)*srcWidth + srcX + 1) * 4 + 2];

	//a chanel  
	dstOffset = (y*dstWidth + x) * 4 + 3;
	d_dst[dstOffset] = 255;
}

#endif // #ifndef _SPATIAL_TRANSFORM_CUH_