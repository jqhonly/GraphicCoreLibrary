#pragma once
#ifndef GPUMANAGEMENT_H
#define GPUMANAGEMENT_H
#include "cuda_runtime.h"
#include <vector>

namespace GCL
{
	class GpuManagement
	{
	public:
		int DeviceCount;
		GpuManagement();
		std::vector<cudaDeviceProp> props;
		
		
	private:
		
		void GetDeviceCount();
		void AllDeviceQuery();
	};
}

#endif // GPUMANAGEMENT_H