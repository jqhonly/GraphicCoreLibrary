#include "CameraManagement.h"

namespace Camera
{
	void CALLBACK HikVision::DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, void *pUser, void* nReserved2)
	{
		printf("pUser:%p\n", pUser);
		//�Ӿ�̬��д��
		backdata.pBuf = pBuf;
		backdata.pUser = pUser;
		//�ͷž�̬��д��
	}

	///ʵʱ���ص�
	void CALLBACK HikVision::fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
	{
		DWORD dRet;
		switch (dwDataType)
		{
		case NET_DVR_SYSHEAD:    //ϵͳͷ
			if (!PlayM4_GetPort(&nPort)) //��ȡ���ſ�δʹ�õ�ͨ����
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
				//���ý���ص����� ֻ���벻��ʾ
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

				//����Ƶ����
				if (!PlayM4_Play(nPort, hWnd))
				{
					dRet = PlayM4_GetLastError(nPort);
					break;
				}
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
		default:
			break;
		}
	}

	void CALLBACK HikVision::g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
	{
		switch (dwType)
		{
		case EXCEPTION_RECONNECT:    //Ԥ��ʱ����
			printf("----------reconnect--------\n");
			break;
		default:
			break;
		}
	}

	void HikVision::play() {
		//---------------------------------------
		// ��ʼ��
		NET_DVR_Init();
		//��������ʱ��������ʱ��
		NET_DVR_SetConnectTime(2000, 1);
		NET_DVR_SetReconnect(10000, true);

		//---------------------------------------
		// ��ȡ����̨���ھ��
		//HMODULE hKernel32 = GetModuleHandle((LPCWSTR)"kernel32");
		//GetConsoleWindow = (PROCGETCONSOLEWINDOW)GetProcAddress(hKernel32,"GetConsoleWindow");

		//---------------------------------------
		// ע���豸
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
		//�����쳣��Ϣ�ص�����
		NET_DVR_SetExceptionCallBack_V30(0, hWnd, HikVision::g_ExceptionCallBack, (void*)pSerialNumber);

		NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
		struPlayInfo.hPlayWnd = hWnd;         //��ҪSDK����ʱ�����Ϊ��Чֵ����ȡ��������ʱ����Ϊ��
		struPlayInfo.lChannel = 1;       //Ԥ��ͨ����
		struPlayInfo.dwStreamType = 0;       //0-��������1-��������2-����3��3-����4���Դ�����
		struPlayInfo.dwLinkMode = 0;       //0- TCP��ʽ��1- UDP��ʽ��2- �ಥ��ʽ��3- RTP��ʽ��4-RTP/RTSP��5-RSTP/HTTP

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
		//�ر�Ԥ��
		if (!NET_DVR_StopRealPlay(lRealPlayHandle))
		{
			printf("NET_DVR_StopRealPlay error! Error number: %d\n", NET_DVR_GetLastError());
		}
		//ע���û�
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