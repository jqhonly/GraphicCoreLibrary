#pragma once
#include <HCNetSDK.h>
#include <PlayM4.h>
#include <string>

namespace Camera
{
	typedef struct {
		char *pBuf;
		void *pUser;
	}CallBackData;

	static LONG nPort; //1.这个参数在不同摄像头调用的时候怕是要做读写锁
	static HWND hWnd;
	static CallBackData backdata; //2.怕是也要做读写锁

	 //再声明两个静态读写锁

	class HikVision {

	public:
		void play();
		void stop();
		const char *getSerialNumber() const;
		~HikVision();

	private:
		static void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser);
		static void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser);
		static void CALLBACK DecCBFun(long nPort, char * pBuf, long nSize, FRAME_INFO * pFrameInfo, void *pUser, void* nReserved2);
		//static int yv12_to_rgb(unsigned char *yv12, unsigned char *rgb, unsigned int width, unsigned int height);

	private:
		//cv::Mat frame;
		LONG lRealPlayHandle;
		LONG lUserID;
		const char *pSerialNumber;
	};
}