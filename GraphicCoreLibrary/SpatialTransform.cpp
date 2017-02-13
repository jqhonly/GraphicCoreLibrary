#include "SpatialTransform.h"

extern "C"
int Resize_Host(unsigned char*h_src, int srcWidth, int srcHeight, unsigned char *h_dst, int dstWidth, int dstHeight, int deviceid);

namespace GCL
{
	int SpatialTransform::Resize(unsigned char* h_src, int srcWidth, int srcHeight, unsigned char* h_dst, int dstWidth, int dstHeight, int deviceid)
	{
		return Resize_Host(h_src, srcWidth, srcHeight, h_dst, dstWidth, dstHeight, deviceid);
	}

}