#include <helper_math.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include "ColorTransform.h"

#include <windows.h>  
#include "lodepng.h"
//#include "CameraManagement.h"

using namespace GCL;
//using namespace Camera;

unsigned char* LoadPng(std::string filepath, unsigned int &sizeX, unsigned int &sizeY)
{
	std::vector<unsigned char> vec;
	lodepng::decode(vec, sizeX, sizeY, filepath);
	//"C:\\Research\\test_project_cuda\\Debug\\g.png"
	unsigned char* imdata = new unsigned char[sizeX * sizeY * 4];
	for (int i = 0; i < sizeX*sizeY; i++)
	{
		imdata[i * 4] = vec[i * 4];
		imdata[i * 4 + 1] = vec[i * 4 + 1];
		imdata[i * 4 + 2] = vec[i * 4 + 2];
		imdata[i * 4 + 3] = vec[i * 4 + 3];
	}
	return imdata;
}

void SavePng(std::string SvaePath, unsigned char* puint, unsigned int sizeX, unsigned int sizeY)
{
	std::vector<unsigned char> vec(sizeX*sizeY * 4);
	for (int i = 0; i < sizeX*sizeY * 4; i++)
	{
		vec[i] = puint[i];
	}
	lodepng::encode(SvaePath, vec, sizeX, sizeY);
}

void decoder()
{
	LARGE_INTEGER Freq;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	QueryPerformanceFrequency(&Freq);
	//**********
	unsigned int width, height;
	unsigned char* t_rgba = LoadPng("C:\\Research\\test_project_cuda\\x64\\Debug\\d.png", width, height);
	/*width = 1280;
	height = 720;*/
	//init
	ColorTransform ctrans(width, height);
	//malloc memory space
	unsigned char *input = (unsigned char*)malloc(width * height * 3 / 2 * sizeof(unsigned char));
	unsigned char *output = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
	
	/*for (int i = 0; i < 10; i++)
	{
		QueryPerformanceCounter(&start);
		ctrans.ColorTrans_RetineX(t_rgba, output, 0);
		QueryPerformanceCounter(&end);
		printf("execute time: %lld\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);
	}*/
	
	for (size_t j = 0; j < 10; j++)
	{
		for (size_t i = 0; i < width * height * 3 / 2; i++)
		{
			input[i] = i*(10 - j) % 255;
		}
		QueryPerformanceCounter(&start);
		ctrans.ColorTrans_YV12toARGB32_RetineX(input, output, 0);
		QueryPerformanceCounter(&end);
		printf("execute time: %lld\n", (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);
	}
	//fake data
	SavePng("C:\\Research\\test_project_cuda\\x64\\Debug\\color_shf_.png", output, width, height);

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

	/*Camera::InitCamera();
	Camera::Login("192.168.0.66", 8000, "admin", "hk123456");
	Camera::Activte();

	while (true)
	{
		Sleep(10);
	}
	*/

	decoder();
	return 0;
}