#include "ColorTransform.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <windows.h>  

#define CLIP(value) (unsigned char)(((value)>0xFF)?0xff:(((value)<0)?0:(value)))
//#define CUBLASAPI

extern "C"
int gaussianFilterRGBA(uint *d_src, uint *d_dest, uint *d_temp, int width, int height, float sigma, int order, int nthreads, int deviceid);

extern "C"
int ARGB32toUINT32(unsigned char* pARGB, uint* pUINT, int width, int height, int deviceid);

extern "C"
int logAve(float* logAveImage, uint* rgbaImage, uint* BlurImage1, uint* BlurImage2, uint* BlurImage3, int Width, int Height, int deviceid);

extern "C"
int h_Rescale(unsigned char* reScaledImage, float* logAveImage, float max1, float min1, float max2, float min2, float max3, float min3, int Width, int Height, int deviceid);

extern "C"
int h_GetMaxMinValue(float* d_logave, float &max1, float &min1, float &max2, float &min2, float &max3, float &min3, int Width, int Height, int deviceid);

namespace GCL
{
	ColorTransform::ColorTransform(int _width, int _height, int _deviceid)
	{
		width = _width;
		height = _height;
		size = width * height;
		deviceid = _deviceid;
		if (deviceid>=0)
		{
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
		//stat = cublasCreate(&handle);
	}

	ColorTransform::~ColorTransform()
	{
		cudaFree(d_yv12);
		cudaFree(d_rgba32);
		cudaFree(d_o_rgba32);//Detete before release
		cudaFree(d_img);
		cudaFree(d_temp1);
		cudaFree(d_temp2);
		cudaFree(d_temp3);
		cudaFree(d_result1);
		cudaFree(d_result2);
		cudaFree(d_result3);
		cudaFree(d_logave);

		delete[] h_logave;//delete temp host data

		//stat = cublasDestroy(handle);
	}

	void ColorTransform::GetMaxMinValue(float* h_logave)
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

	int ColorTransform::ResetDeviceID(int _deviceid)
	{
		if (_deviceid>=0)
		{
			deviceid = _deviceid;
			cudaSetDevice(deviceid);//error check require!
			//re-malloc
			cudaMalloc((void**)&d_yv12, size * 3 / 2 * sizeof(unsigned char));//input
			cudaMalloc((void**)&d_o_rgba32, size * 4 * sizeof(unsigned char));//fake input//Detete before release
			cudaMalloc((void**)&d_rgba32, size * 4 * sizeof(unsigned char));//output
			cudaMalloc((void **)&d_img, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_temp1, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_temp2, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_temp3, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_result1, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_result2, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_result3, size * sizeof(unsigned int));
			cudaMalloc((void **)&d_logave, sizeof(float)* size * 3);

			h_logave = (float*)malloc(sizeof(float)* size * 3);
			return cudaGetLastError();
		}
		else
		{
			deviceid = _deviceid;
			return 0;
		}
	}

	int ColorTransform::GetCurrentDeviceID()
	{
		return deviceid;
	}

	int ColorTransform::ColorTrans_YV12toARGB32_CPU(unsigned char *yv12, unsigned char *rgba32)
	{
		unsigned char *py;
		unsigned char *pu;
		unsigned char *pv;
		unsigned int linesize = width * 3;
		unsigned int uvlinesize = width / 2;
		unsigned int offset = 0;
		unsigned int offset1 = 0;
		unsigned int offsety = 0;
		unsigned int offsety1 = 0;
		unsigned int offsetuv = 0;

		py = yv12;
		pv = py + (width*height);
		pu = pv + ((width*height) / 4);

		unsigned int h = 0;
		unsigned int w = 0;

		unsigned int wy = 0;
		unsigned int huv = 0;
		unsigned int wuv = 0;

		for (h = 0;h<height;h += 2)
		{
			wy = 0;
			wuv = 0;
			offset = h * linesize;
			offset1 = (h + 1) * linesize;
			offsety = h * width;
			offsety1 = (h + 1) * width;
			offsetuv = huv * uvlinesize;

			for (w = 0;w<linesize;w += 6)
			{
				/* standart: r = y0 + 1.402 (v-128) */
				/* logitech: r = y0 + 1.370705 (v-128) */
				rgba32[w + offset] = CLIP(py[wy + offsety] + 1.402 * (pv[wuv + offsetuv] - 128));
				/* standart: g = y0 - 0.34414 (u-128) - 0.71414 (v-128)*/
				/* logitech: g = y0 - 0.337633 (u-128)- 0.698001 (v-128)*/
				rgba32[(w + 1) + offset] = CLIP(py[wy + offsety] - 0.34414 * (pu[wuv + offsetuv] - 128) - 0.71414*(pv[wuv + offsetuv] - 128));
				/* standart: b = y0 + 1.772 (u-128) */
				/* logitech: b = y0 + 1.732446 (u-128) */
				rgba32[(w + 2) + offset] = CLIP(py[wy + offsety] + 1.772 *(pu[wuv + offsetuv] - 128));

				rgba32[(w + 3) + offset] = CLIP(py[wy + 1 + offsety] + 1.402 * (pv[wuv + offsetuv] - 128));
				rgba32[(w + 4) + offset] = CLIP(py[wy + 1 + offsety] - 0.34414 * (pu[wuv + offsetuv] - 128) - 0.71414*(pv[wuv + offsetuv] - 128));
				rgba32[(w + 5) + offset] = CLIP(py[wy + 1 + offsety] + 1.772 *(pu[wuv + offsetuv] - 128));

				rgba32[w + offset1] = CLIP(py[wy + offsety1] + 1.402 * (pv[wuv + offsetuv] - 128));
				rgba32[(w + 1) + offset1] = CLIP(py[wy + offsety1] - 0.34414 * (pu[wuv + offsetuv] - 128) - 0.71414*(pv[wuv + offsetuv] - 128));
				rgba32[(w + 2) + offset1] = CLIP(py[wy + offsety1] + 1.772 *(pu[wuv + offsetuv] - 128));

				rgba32[(w + 3) + offset1] = CLIP(py[wy + 1 + offsety1] + 1.402 * (pv[wuv + offsetuv] - 128));
				rgba32[(w + 4) + offset1] = CLIP(py[wy + 1 + offsety1] - 0.34414 * (pu[wuv + offsetuv] - 128) - 0.71414*(pv[wuv + offsetuv] - 128));
				rgba32[(w + 5) + offset1] = CLIP(py[wy + 1 + offsety1] + 1.772 *(pu[wuv + offsetuv] - 128));

				wuv++;
				wy += 2;
			}
			huv++;
		}
		return 0;
	}

	int ColorTransform::ColorTrans_YV12toARGB32(unsigned char* h_YV12, unsigned char* h_RGBA32)
	{
		if (deviceid>=0)
		{
			int result;
			cudaMemcpy(d_yv12, h_YV12, size * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
			cudaMemcpy(h_RGBA32, d_rgba32, size * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			return result;
		}
		else
		{
			return ColorTrans_YV12toARGB32_CPU(h_YV12, h_RGBA32);
		}
	}

	int ColorTransform::ColorTrans_YV12toARGB32_RetineX(unsigned char* h_YV12, unsigned char* h_RGBA32)
	{
		if (deviceid >= 0)//GPU
		{
			/*LARGE_INTEGER Freq;
			LARGE_INTEGER start;
			LARGE_INTEGER end;
			QueryPerformanceFrequency(&Freq);
			QueryPerformanceCounter(&start);*/
			int result;
			cudaMemcpy(d_yv12, h_YV12, size * 3 / 2 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			result = YV12toARGB32(d_yv12, d_rgba32, width, height, deviceid);
			result = ARGB32toUINT32(d_rgba32, d_img, width, height, deviceid);
			result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
			result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
			result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
			result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
			cudaMemcpy(h_logave, d_logave, size * 3 * sizeof(float), cudaMemcpyDeviceToHost);
			GetMaxMinValue(h_logave);//CPU implementation
			//result = h_GetMaxMinValue(d_logave, max1, min1, max2, min2, max3, min3, width, height, deviceid);//GPU implementation
			result = h_Rescale(d_rgba32, d_logave, max1, min1, max2, min2, max3, min3, width, height, deviceid);
			cudaMemcpy(h_RGBA32, d_rgba32, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			/*QueryPerformanceCounter(&end);
			printf("execution time: %lld\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);*/
			return result;
		}
		else
		{
			return -1;//TODO: CPU implementation here
		}
	}

	//Test function, hid this kind of detials before release
	int ColorTransform::ColorTrans_RetineX(unsigned char* h_o_RGBA32, unsigned char* h_RGBA32)
	{
		if (deviceid >= 0)//GPU
		{
			int result;
			cudaMemcpy(d_o_rgba32, h_o_RGBA32, size * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			result = ARGB32toUINT32(d_o_rgba32, d_img, width, height, deviceid);
			result = gaussianFilterRGBA(d_img, d_result1, d_temp1, width, height, sigma1, order, nthreads, deviceid);
			result = gaussianFilterRGBA(d_img, d_result2, d_temp2, width, height, sigma2, order, nthreads, deviceid);
			result = gaussianFilterRGBA(d_img, d_result3, d_temp3, width, height, sigma3, order, nthreads, deviceid);
			result = logAve(d_logave, d_img, d_result1, d_result2, d_result3, width, height, deviceid);
			cudaMemcpy(h_logave, d_logave, size * 3 * sizeof(float), cudaMemcpyDeviceToHost);//Adaptive MaxMin value to host to avoid cross thread error
			GetMaxMinValue(h_logave);//Not nessessary for every time
			result = h_Rescale(d_rgba32, d_logave, max1, min1, max2, min2, max3, min3, width, height, deviceid);
			cudaMemcpy(h_RGBA32, d_rgba32, size * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
			return result;
		}
		else
		{
			return -1;//TODO: CPU implementation here
		}
	}

}

