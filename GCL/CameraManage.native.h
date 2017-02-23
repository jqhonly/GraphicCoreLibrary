#pragma once
#ifndef CAMERAMANAGE_NATIVE_H
#define CAMERAMANAGE_NATIVE_H
#include "CameraManagement.h"

using namespace GCL;

class CameraManage
{
	Camera * camera;
public:
	CameraManage(std::string ip, short port, std::string user, std::string pwd);
	~CameraManage();
	bool login_native();
	bool play_native(int display_hwnd);
	bool stop_native();
	bool logout_native();
	unsigned char * getFrame_native();
	int Width;
	int Height;
};

#endif // CAMERAMANAGE_NATIVE_H