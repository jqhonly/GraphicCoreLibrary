#include "CameraManagement.h"

namespace Camera
{
	void CALLBACK HikVision::DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, void *pUser, void* nReserved2)
	{
		printf("pUser:%p\n", pUser);
		//加静态读写锁
		backdata.pBuf = pBuf;
		backdata.pUser = pUser;
		//释放静态读写锁
	}

	///实时流回调
	void CALLBACK HikVision::fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
	{
		DWORD dRet;
		switch (dwDataType)
		{
		case NET_DVR_SYSHEAD:    //系统头
			if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
			{
				break;
			}
			if (dwBufSize > 0)
			{
				if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 2 * 1024 * 1024))
				{
					dRet = PlayM4_GetLastError(nPort);
					printf("PlayM4_OpenStream Failed!:%d\n", dRet);
					break;
				}
				//设置解码回调函数 只解码不显示
				if (!PlayM4_SetDecCallBackExMend(nPort, HikVision::DecCBFun, NULL, 0, pUser))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}

				if (!PlayM4_ThrowBFrameNum(nPort, 2))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}

				//打开视频解码
				if (!PlayM4_Play(nPort, hWnd))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}
			}
			break;

		case NET_DVR_STREAMDATA:   //码流数据
			if (dwBufSize > 0 && nPort != -1)
			{
				BOOL inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
				while (!inData)
				{
					Sleep(10);
					inData = PlayM4_InputData(nPort, pBuffer, dwBufSize);
					printf("PlayM4_InputData failed \n");
				}
			}
			break;
		default:
			break;
		}
	}

	void CALLBACK HikVision::g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
	{
		switch (dwType)
		{
		case EXCEPTION_RECONNECT:    //预览时重连
			printf("----------reconnect--------\n");
			break;
		default:
			break;
		}
	}

	void HikVision::play() {
		//---------------------------------------
		// 初始化
		NET_DVR_Init();
		//设置连接时间与重连时间
		NET_DVR_SetConnectTime(2000, 1);
		NET_DVR_SetReconnect(10000, true);

		//---------------------------------------
		// 获取控制台窗口句柄
		//HMODULE hKernel32 = GetModuleHandle((LPCWSTR)"kernel32");
		//GetConsoleWindow = (PROCGETCONSOLEWINDOW)GetProcAddress(hKernel32,"GetConsoleWindow");

		//---------------------------------------
		// 注册设备
		NET_DVR_DEVICEINFO_V30 struDeviceInfo;
		lUserID = NET_DVR_Login_V30("192.168.0.65", 8000, "admin", "hk123456", &struDeviceInfo);
		if (lUserID < 0)
		{
			printf("Login error, %d\n", NET_DVR_GetLastError());
			NET_DVR_Cleanup();
			return;
		}
		std::string SerialNumber = std::string((char *)struDeviceInfo.sSerialNumber);
		char *spSerialNumber = new char[SerialNumber.length()];
		strcpy(spSerialNumber, SerialNumber.c_str());
		pSerialNumber = (const char*)spSerialNumber;

		//---------------------------------------
		//设置异常消息回调函数
		NET_DVR_SetExceptionCallBack_V30(0, hWnd, HikVision::g_ExceptionCallBack, (void*)pSerialNumber);

		NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
		struPlayInfo.hPlayWnd = hWnd;         //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
		struPlayInfo.lChannel = 1;       //预览通道号
		struPlayInfo.dwStreamType = 0;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
		struPlayInfo.dwLinkMode = 0;       //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP

		lRealPlayHandle = NET_DVR_RealPlay_V40(lUserID, &struPlayInfo, HikVision::fRealDataCallBack, (void*)pSerialNumber);

		if (lRealPlayHandle<0)
		{
			NET_DVR_Logout(lUserID);
			printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
			return;
		}
	}

	void HikVision::stop()
	{
		//--------------------------------------
		//关闭预览
		if (!NET_DVR_StopRealPlay(lRealPlayHandle))
		{
			printf("NET_DVR_StopRealPlay error! Error number: %d\n", NET_DVR_GetLastError());
		}
		//注销用户
		NET_DVR_Logout(lUserID);
		NET_DVR_Cleanup();
	}

	const char *HikVision::getSerialNumber() const {
		return pSerialNumber;
	}

	HikVision::~HikVision() {
		delete[] pSerialNumber;
	}
}


//void main() {
//	Camera::HikVision HikVision;
//	HikVision.play();
//	while (1) {
//		//printf("SerialNumber: %s\n", HikVision.getSerialNumber());
//		Sleep(100);
//	}
//	HikVision.stop();
//}