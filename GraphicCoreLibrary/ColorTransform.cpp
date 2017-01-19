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
	ColorTransform::ColorTransform(int _width, int _height, int _deviceid)
	{
		width = _width;
		height = _height;
		size = width * height;
		deviceid = _deviceid;
		cudaSetDevice(deviceid);//error check require!
		cudaMalloc((void**)&d_yv12, size * 3 / 2 * sizeof(unsigned char));//input
		cudaMalloc((void**)&d_o_rgba32, size * 4 * sizeof(unsigned char));//fake input//Detete before release
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

		h_logave = (float*)malloc(sizeof(float)* size * 3);
	}

	ColorTransform::~ColorTransform()
	{
		cudaFree(d_yv12);
		cudaFree(d_rgba32);
		cudaFree(d_o_rgba32);
		cudaFree(d_img);
		cudaFree(d_temp1);
		cudaFree(d_temp2);
		cudaFree(d_temp3);
		cudaFree(d_result1);
		cudaFree(d_result2);
		cudaFree(d_result3);
		cudaFree(d_logave);

		delete[] h_logave;//delete temp host data
	}

	void ColorTransform::GetManMinValue(float* h_logave)
	{
		float mean1 = 0, var1 = 0;
		float mean2 = 0, var2 = 0;
		float mean3 = 0, var3 = 0;
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				mean1 += h_logave[(i*height + j) * 3];
				mean2 += h_logave[(i*height + j) * 3 + 1];
				mean3 += h_logave[(i*height + j) * 3 + 2];
			}
		}
		mean1 /= width*height;
		mean2 /= width*height;
		mean3 /= width*height;
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				var1 += (h_logave[(i*height + j) * 3] - mean1)*(h_logave[(i*height + j) * 3] - mean1);
				var2 += (h_logave[(i*height + j) * 3 + 1] - mean2)*(h_logave[(i*height + j) * 3 + 1] - mean2);
				var3 += (h_logave[(i*height + j) * 3 + 2] - mean3)*(h_logave[(i*height + j) * 3 + 2] - mean3);
			}
		}
		var1 = sqrt(var1 / (width*height - 1));
		var2 = sqrt(var2 / (width*height - 1));
		var3 = sqrt(var3 / (width*height - 1));
		max1 = mean1 + 2 * var1;
		max2 = mean2 + 2 * var2;
		max3 = mean3 + 2 * var3;
		min1 = mean1 - 2 * var1;
		min2 = mean2 - 2 * var2;
		min3 = mean3 - 2 * var3;
	}


	int ColorTransform::ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32)
	{
		int result;
		cudaMemcpy(d_yv12, h_YV12, size * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
		cudaMemcpy(h_RGBA32, d_rgba32, size * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		return result;
	}

	int ColorTransform::ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32)
	{
		int result;
		cudaMemcpy(d_yv12, h_YV12, size * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
		result = ARGB32toUINT32(d_rgba32, d_img, width, height, deviceid);
		result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
		result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
		cudaMemcpy(h_logave, d_logave, size * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		GetManMinValue(h_logave);
		result = h_Rescale(d_rgba32, d_logave, max1, min1, max2, min2, max3, min3, width, height, deviceid);
		cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		return result;
	}

	//Test function, hid this kind of detials before release
	int ColorTransform::ColorTrans_RetineX(unsigned char* h_o_RGBA32, unsigned char* h_RGBA32)
	{
		int result;
		cudaMemcpy(d_o_rgba32, h_o_RGBA32, size * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
		result = ARGB32toUINT32(d_o_rgba32, d_img, width, height, deviceid);
		result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
		result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
		result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
		cudaMemcpy(h_logave, d_logave, size * 3 * sizeof(float), cudaMemcpyDeviceToHost);//Adaptive MaxMin value to host to avoid cross thread error
		GetManMinValue(h_logave);//Not nessessary for every time
		result = h_Rescale(d_rgba32, d_logave, max1, min1, max2, min2, max3, min3, width, height, deviceid);
		cudaMemcpy(h_RGBA32, d_rgba32, size * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		return result;
	}

}

