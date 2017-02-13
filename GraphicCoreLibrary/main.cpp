#include "ColorTransform.h"
#include "CameraManagement.h"
#include "SpatialTransform.h"
#include <windows.h>  
#include "lodepng.h"
#include <thread> 

using namespace GCL;

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
	//unsigned char* t_rgba = LoadPng("C:\\Research\\test_project_cuda\\x64\\Debug\\d.png", width, height);
	width = 1280;
	height = 720;
	//init
	ColorTransform ctrans(width, height, -1);
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
		ctrans.ColorTrans_YV12toARGB32(input, output);
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

void Execution(int id, Camera& camera)
{
	LARGE_INTEGER Freq;
	LARGE_INTEGER start;
	LARGE_INTEGER end;
	QueryPerformanceFrequency(&Freq);
	LONGLONG temp = 0;
	for (int i = 0; i < 10; i++)
	{
		QueryPerformanceCounter(&start);
		auto frame1 = camera.getFrame();
		QueryPerformanceCounter(&end);
		printf("camera%d execution time: %lld\n", id, (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart);
		temp += (end.QuadPart - start.QuadPart) * 1000 / Freq.QuadPart;
	}
	printf("Average execution time if camera%d: %lld\n", id , temp / 50);
	printf("\n");
}

int main()
{
	

	Camera camera1("192.168.0.68", 8000, "admin", "hk123456");
	Camera camera2("192.168.0.67", 8000, "admin", "hk123456");
	camera1.login();
	camera2.login();

	camera1.play();
	camera2.play();

	/*std::thread t1(Execution, 1, camera1);
	std::thread t2(Execution, 2, camera2);
	
	t1.join();
	t2.join();*/
	for(int i = 0; i < 50; i++)
	{
		
		auto frame1 = camera1.getFrame();
		
		if(frame1.get() != nullptr)  //如果运行报double free， 这里改成frame1 != nullptr
		{
			/*unsigned char * temp = new unsigned char[640 * 360 * 4 * sizeof(unsigned char)];
			int xx = SpatialTransform::Resize(frame1->h_CpuData, 1280, 720, temp, 640, 360, 0);
			SavePng("C:\\Research\\GraphicCoreLibrary\\x64\\Debug\\test.png", temp, 640, 360);*/
			printf("frame1: ");
			for(int i = 0; i < 10; i++)
				printf("%x ", frame1->h_CpuData[i]);
			printf("\n");
		}
		auto frame2 = camera2.getFrame();
		if (frame2.get() != nullptr) //如果运行报double free， 这里改成frame2 != nullptr
		{
			printf("frame2: ");
			for (int i = 0; i < 10; i++)
				printf("%x ", frame2->h_CpuData[i]);
			printf("\n");
		}
		printf("\n");
		//Sleep(1);
	}
	
	camera1.stop();
	camera2.stop();

	camera1.logout();
	camera2.logout();

	//decoder();
	return 0;
}