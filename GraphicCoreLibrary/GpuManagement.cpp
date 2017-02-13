#include  "GpuManagement.h"

namespace GCL
{
	void GpuManagement::GetDeviceCount()
	{
		cudaGetDeviceCount(&DeviceCount);
	}

	GpuManagement::GpuManagement()
	{
		GetDeviceCount();
		AllDeviceQuery();
	}

	void GpuManagement::AllDeviceQuery()
	{
		for (int i = 0; i < DeviceCount; i++)
		{
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, i);
			props.push_back(prop);
		}
	}


}