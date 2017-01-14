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
		// ��ʼ��
		NET_DVR_Init();
		//��������ʱ��������ʱ��
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
		struPlayInfo.hPlayWnd = hWnd;         //��ҪSDK����ʱ�����Ϊ��Чֵ����ȡ��������ʱ����Ϊ��
		struPlayInfo.lChannel = 1;       //Ԥ��ͨ����
		struPlayInfo.dwStreamType = 0;       //0-��������1-��������2-����3��3-����4���Դ�����
		struPlayInfo.dwLinkMode = 0;       //0- TCP��ʽ��1- UDP��ʽ��2- �ಥ��ʽ��3- RTP��ʽ��4-RTP/RTSP��5-RSTP/HTTP

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
		case NET_DVR_SYSHEAD:    //ϵͳͷ
			if (!PlayM4_GetPort(&nPort)) //��ȡ���ſ�δʹ�õ�ͨ����
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
				//���ý���ص����� ֻ���벻��ʾ
				if (!PlayM4_SetDecCallBack(nPort, DecCBFun))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}

				//���ý���ص����� ��������ʾ
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

				//����Ƶ����
				if (!PlayM4_Play(nPort, hWnd))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}

				//����Ƶ����, ��Ҫ�����Ǹ�����
				//          if (!PlayM4_PlaySound(nPort))
				//          {
				//              dRet=PlayM4_GetLastError(nPort);
				//              break;
				//          }
			}
			break;

		case NET_DVR_STREAMDATA:   //��������
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