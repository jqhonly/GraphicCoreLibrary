#include "cuda.h"
#include "Wapper.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


#define COLOR_COMPONENT_BIT_SIZE 10

#define COLOR_COMPONENT_MASK     0x3FF



__constant__ float constHueColorSpaceMat[9] = { 1.1644f, 0.0f, 1.596f, 1.1644f, -0.3918f, -0.813f, 1.1644f, 2.0172f, 0.0f };



__device__ static void YUV2RGB(const int* yuvi, float* red, float* green, float* blue)

{

	float luma, chromaCb, chromaCr;



	// Prepare for hue adjustment

	luma = (float)yuvi[0];

	chromaCb = (float)((int)yuvi[1] - 512.0f);

	chromaCr = (float)((int)yuvi[2] - 512.0f);



	// Convert YUV To RGB with hue adjustment

	*red = (luma     * constHueColorSpaceMat[0]) +

		(chromaCb * constHueColorSpaceMat[1]) +

		(chromaCr * constHueColorSpaceMat[2]);



	*green = (luma     * constHueColorSpaceMat[3]) +

		(chromaCb * constHueColorSpaceMat[4]) +

		(chromaCr * constHueColorSpaceMat[5]);



	*blue = (luma     * constHueColorSpaceMat[6]) +

		(chromaCb * constHueColorSpaceMat[7]) +

		(chromaCr * constHueColorSpaceMat[8]);

}



__device__ static int RGBA_pack_10bit(float red, float green, float blue, int alpha)

{

	int ARGBpixel = 0;



	// Clamp final 10 bit results

	red =::fmin(::fmax(red, 0.0f), 1023.f);

	green = ::fmin(::fmax(green, 0.0f), 1023.f);

	blue = ::fmin(::fmax(blue, 0.0f), 1023.f);



	// Convert to 8 bit unsigned integers per color component

	ARGBpixel = (((int)blue >> 2) |

		(((int)green >> 2) << 8) |

		(((int)red >> 2) << 16) |

		(int)alpha);



	return ARGBpixel;

}



__global__ void YV12ToARGB_FourPixel(const unsigned char* pYV12, unsigned int* pARGB, int width, int height)

{

	// Pad borders with duplicate pixels, and we multiply by 2 because we process 4 pixels per thread

	const int x = blockIdx.x *(blockDim.x << 1) + (threadIdx.x << 1);

	const int y = blockIdx.y *(blockDim.y << 1) + (threadIdx.y << 1);



	if ((x + 1) >= width || (y + 1) >= height)

		return;



	// Read 4 Luma components at a time

	int yuv101010Pel[4];

	yuv101010Pel[0] = (pYV12[y * width + x]) << 2;

	yuv101010Pel[1] = (pYV12[y * width + x + 1]) << 2;

	yuv101010Pel[2] = (pYV12[(y + 1)* width + x]) << 2;

	yuv101010Pel[3] = (pYV12[(y + 1)* width + x + 1]) << 2;



	const unsigned int vOffset = width * height;

	const unsigned int uOffset = vOffset + (vOffset >> 2);

	const unsigned int vPitch = width >> 1;

	const unsigned int uPitch = vPitch;

	const int x_chroma = x >> 1;

	const int y_chroma = y >> 1;



	int chromaCb = pYV12[uOffset + y_chroma * uPitch + x_chroma];      //U

	int chromaCr = pYV12[vOffset + y_chroma * vPitch + x_chroma];      //V



	yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));

	yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

	yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));

	yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

	yuv101010Pel[2] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));

	yuv101010Pel[2] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

	yuv101010Pel[3] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE + 2));

	yuv101010Pel[3] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));



	// this steps performs the color conversion

	int yuvi[12];

	float red[4], green[4], blue[4];



	yuvi[0] = (yuv101010Pel[0] & COLOR_COMPONENT_MASK);

	yuvi[1] = ((yuv101010Pel[0] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);

	yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1))& COLOR_COMPONENT_MASK);



	yuvi[3] = (yuv101010Pel[1] & COLOR_COMPONENT_MASK);

	yuvi[4] = ((yuv101010Pel[1] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);

	yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1))& COLOR_COMPONENT_MASK);



	yuvi[6] = (yuv101010Pel[2] & COLOR_COMPONENT_MASK);

	yuvi[7] = ((yuv101010Pel[2] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);

	yuvi[8] = ((yuv101010Pel[2] >> (COLOR_COMPONENT_BIT_SIZE << 1))& COLOR_COMPONENT_MASK);



	yuvi[9] = (yuv101010Pel[3] & COLOR_COMPONENT_MASK);

	yuvi[10] = ((yuv101010Pel[3] >> COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);

	yuvi[11] = ((yuv101010Pel[3] >> (COLOR_COMPONENT_BIT_SIZE << 1))& COLOR_COMPONENT_MASK);



	// YUV to RGB Transformation conversion

	YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);

	YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

	YUV2RGB(&yuvi[6], &red[2], &green[2], &blue[2]);

	YUV2RGB(&yuvi[9], &red[3], &green[3], &blue[3]);



//    pARGB[y * width + x] = RGBA_pack_10bit(red[0], green[0], blue[0], ((int)0xff << 24));

//    pARGB[y * width + x + 1] = RGBA_pack_10bit(red[1], green[1], blue[1], ((int)0xff << 24));

//    pARGB[(y + 1)* width + x] = RGBA_pack_10bit(red[2], green[2], blue[2], ((int)0xff << 24));

//    pARGB[(y + 1)* width + x + 1] = RGBA_pack_10bit(red[3], green[3], blue[3], ((int)0xff << 24));

    pARGB[y * width + x] = RGBA_pack_10bit(red[0], green[0], blue[0], ((int)0xff << 24));

    pARGB[y * width + x + 1] = RGBA_pack_10bit(red[1], green[1], blue[1], ((int)0xff << 24));

    pARGB[(y + 1)* width + x] = RGBA_pack_10bit(red[2], green[2], blue[2], ((int)0xff << 24));

    pARGB[(y + 1)* width + x + 1] = RGBA_pack_10bit(red[3], green[3], blue[3], ((int)0xff << 24));

}



extern "C" bool YV12ToARGB(unsigned char* pYV12, unsigned char* pARGB, int width, int height)//extern "C" bool __declspec(dllexport) __stdcall

{

	unsigned char* d_src;

	unsigned char* d_dst;

	unsigned int srcMemSize = sizeof(unsigned char)* width * height * 3 / 2;

    unsigned int dstMemSize = sizeof(unsigned char)* width * height * 4;



	cudaMalloc((void**)&d_src, srcMemSize);

	cudaMalloc((void**)&d_dst, dstMemSize);

	cudaMemcpy(d_src, pYV12, srcMemSize, cudaMemcpyHostToDevice);



    dim3 block(32, 8);

	int gridx = (width + 2 * block.x - 1) / (2 * block.x);

	int gridy = (height + 2 * block.y - 1) / (2 * block.y);

	dim3 grid(gridx, gridy);



    YV12ToARGB_FourPixel << <grid, block >> >((const unsigned char*)d_src, (unsigned int*)d_dst, width, height);//(unsigned int*)

	cudaMemcpy(pARGB, d_dst, dstMemSize, cudaMemcpyDeviceToHost);
	cudaFree(d_src);
	cudaFree(d_dst);

	return true;

}

//int main()
//{
//	//int* pa = (int*)malloc(sizeof(int) * 128);;
//	//int arraySize = 1280 * 720;
//	unsigned char *input = (unsigned char*)malloc(1280 * 720 * 3 / 2 * sizeof(unsigned char));
//	for (size_t i = 0; i < 1280 * 720 * 3 / 2; i++)
//	{
//		input[i] = i*177%13;
//	}
//	/*input[1280 * 720 * 3 / 2-1] = (unsigned char)"a";
//	printf("%d\n", input[1280 * 720 * 3 / 2-1]);*/
//	//unsigned char input[1280 * 720 * 3 / 2] = { 0 };
//	unsigned char *output=(unsigned char*)malloc(1280 * 720 * 4 * sizeof(unsigned char));
//	bool x = YV12ToARGB(input, output, 1280, 720);
//	for (size_t i = 0; i < 10; i++)
//	{
//		printf("input: %d\n", input[1280 * 720 * 3 / 2-11+i]);
//		printf("output: %d\n\n", output[i]);
//	}
//	printf("%d\n", x);
//	std::cin >> x;
//	free(input);
//	free(output);
//	return 0;
//}
