#pragma once
#ifndef _SPATIAL_TRANSFORM_H_
#define _SPATIAL_TRANSFORM_H_


namespace GCL
{
	class SpatialTransform
	{
	public: 
		static int Resize(unsigned char*h_src, int srcWidth, int srcHeight, unsigned char *h_dst, int dstWidth, int dstHeight, int deviceid);
	};
}

#endif // #ifndef _SPATIAL_TRANSFORM_H_