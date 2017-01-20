#pragma once

namespace GCL
{
	class CpuBitmap
	{
	public:
		unsigned char* h_CpuData;
		float * h_float_CpuData;
		int width;
		int height;
		int depth;
		CpuBitmap();
		CpuBitmap(unsigned char* input, int _width, int _height, int _depth);
		~CpuBitmap();

	public:

	};
}