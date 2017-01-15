#include <helper_math.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "ColorTransform.h"

#include <windows.h>  
#include "CameraManagement.h"

using namespace GCL;
//using namespace Camera;

void decoder()
{
	LARGE_INTEGER Freq;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	QueryPerformanceFrequency(&Freq);
	//**********
	//init
	ColorTransform ctrans(1280, 720);
	//malloc memory space
	unsigned char *input = (unsigned char*)malloc(1280 * 720 * 3 / 2 * sizeof(unsigned char));
	unsigned char *output = (unsigned char*)malloc(1280 * 720 * 4 * sizeof(unsigned char));
	for (size_t j = 0; j < 10; j++)
	{
		for (size_t i = 0; i < 1280 * 720 * 3 / 2; i++)
		{
			input[i] = i*(10 - j) % 255;
		}
		QueryPerformanceCounter(&start);
		ctrans.ColorTrans_YV12toARGB32(input, output, 0);
		QueryPerformanceCounter(&end);
		printf("execute time: %lld\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);
	}
	//fake data


	for (size_t i = 0; i < 10; i++)
	{
		printf("output: %d\n", output[i]);
	}
}

int main()
{
	/*HikVision hik = HikVision();
	hik.InitCamera();
	hik.Login("192.168.0.68", 8000, "admin", "hk123456");
	hik.Activte();*/

	Camera::InitCamera();
	Camera::Login("192.168.0.66", 8000, "admin", "hk123456");
	Camera::Activte();

	while (true)
	{
		Sleep(10);
	}
	
	return 0;
}