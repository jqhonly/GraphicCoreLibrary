#pragma once
#include <HCNetSDK.h>
#include <PlayM4.h>
#include <string>

#define USECOLOR 1
namespace Camera
{
	LONG nPort = -1;
	LONG lRealPlayHandle;
	LONG lUserID;
	HWND hWnd = NULL;

	std::string IP;
	unsigned int port;
	std::string userName;
	std::string Pwd;
	void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser);
	void CALLBACK DecCBFun(LONG nPort, char * pBuf, LONG nSize, FRAME_INFO * pFrameInfo, void* nReserved1, void* nReserved2)
	{
		printf("%d\n", nSize);
	}

	void InitCamera()
	{
		// 初始化
		NET_DVR_Init();
		//设置连接时间与重连时间
		NET_DVR_SetConnectTime(2000, 1);
		NET_DVR_SetReconnect(10000, true);
	}

	void Login(std::string ip, unsigned int Port, std::string Username, std::string pwd)
	{
		NET_DVR_DEVICEINFO_V30 struDeviceInfo;
		lUserID = NET_DVR_Login_V30((char*)ip.c_str(), Port, (char*)Username.c_str(), (char*)pwd.c_str(), &struDeviceInfo);//"192.168.0.68", 8000, "admin", "hk123456"
		if (lUserID < 0)
		{
			printf("Registe Fail!\n");
			NET_DVR_Cleanup();
		}
	}

	void Activte()
	{
		NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
		struPlayInfo.hPlayWnd = hWnd;         //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
		struPlayInfo.lChannel = 1;       //预览通道号
		struPlayInfo.dwStreamType = 0;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
		struPlayInfo.dwLinkMode = 0;       //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP

		lRealPlayHandle = NET_DVR_RealPlay_V40(lUserID, &struPlayInfo, fRealDataCallBack, NULL);
		if (lRealPlayHandle < 0)
		{
			printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
			printf("Preview Fail!\n");
		}
		//DWORD aa = NET_DVR_GetLastError();
	}

	void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
	{
		unsigned int dRet;
		switch (dwDataType)
		{
		case NET_DVR_SYSHEAD:    //系统头
			if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
			{
				//DWORD aa = PlayM4_GetLastError(nPort);
				break;
			}
			if (dwBufSize > 0)
			{
				
				if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 2 * 1024 * 1024))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}
				//设置解码回调函数 只解码不显示
				if (!PlayM4_SetDecCallBack(nPort, DecCBFun))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}

				//设置解码回调函数 解码且显示
				//if (!PlayM4_SetDecCallBackEx(nPort,DecCBFun,NULL,NULL))
				//{
				//  dRet=PlayM4_GetLastError(nPort);
				//  break;
				//}
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

				//打开音频解码, 需要码流是复合流
				//          if (!PlayM4_PlaySound(nPort))
				//          {
				//              dRet=PlayM4_GetLastError(nPort);
				//              break;
				//          }
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
		}
	}
}