#include "stdafx.h"

using namespace GCL;

CameraManage::CameraManage(std::string ip, short port, std::string user, std::string pwd)
{
	camera = new Camera(ip.c_str(), (unsigned short)port, user.c_str(), pwd.c_str());
}

CameraManage::~CameraManage()
{
	delete[] camera;
}

bool CameraManage::login_native()
{
	return camera->login();
}

bool CameraManage::play_native(int display_hwnd)
{
	return camera->play(((HWND)display_hwnd));
}

bool CameraManage::stop_native()
{
	return camera->stop();
}

bool CameraManage::logout_native()
{
	return camera->logout();
}

unsigned char * CameraManage::getFrame_native()
{
	if (camera->getFrame()!=nullptr)
	{
		Width = camera->getFrame()->width;
		Height = camera->getFrame()->height;
		return camera->getFrame()->h_CpuData;
	}
	else
	{
		return nullptr;
	}
}