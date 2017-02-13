/*
This Camera Management part is contributed by Yifu Zhang(zhangyifu@compilesense.com)

Copyright (c) 2014-2017 CompileSense& Glassix

*/
#include <HCNetSDK.h>
#include <PlayM4.h>
#include "CameraManagement.h"
#include "GpuManagement.h"
#include <unordered_map>

using namespace std;

namespace GCL {
	static std::unordered_map<long, long> portMap;
	static std::unordered_map<long, unique_ptr<CpuBitmap>> matMap;
	static std::unordered_map<long, unique_ptr<ColorTransform>> ctMap;
	static unique_ptr<CpuBitmap> nullMat;
	static GpuManagement *gm = new GpuManagement();

	//--------------------------------------------
	//解码回调 视频为YUV数据(YV12)，音频为PCM数据
	static void CALLBACK DecCBFun(LONG nPort, char * pBuf, LONG nSize, FRAME_INFO * pFrameInfo, void *nUser, void *nReserved2)
	{

		long lFrameType = pFrameInfo->nType;

		if (lFrameType == T_YV12)
		{
			unsigned char * rgba_data =new unsigned char[pFrameInfo->nWidth *pFrameInfo->nHeight *4*sizeof(unsigned char)];
			int x = ctMap[reinterpret_cast<long>(nUser)]->ColorTrans_YV12toARGB32(reinterpret_cast<unsigned char *>(pBuf), rgba_data);
			auto pImg = new CpuBitmap(rgba_data, pFrameInfo->nWidth, pFrameInfo->nHeight, 4);
			matMap[reinterpret_cast<long>(nUser)].reset(pImg);
			delete[] rgba_data;
		}
	}


	///实时流回调
	static void CALLBACK fRealDataCallBack(LONG lRealHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void *pUser)
	{
		DWORD dRet;
		switch (dwDataType)
		{
		case NET_DVR_SYSHEAD:    //系统头
		{
			LONG nPort;
			if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
			{
				dRet = PlayM4_GetLastError(nPort);
				printf("PlayM4_GetPort Failed!:%d\n", dRet);
				break;
			}

			portMap[reinterpret_cast<long>(pUser)] = nPort;

			if (dwBufSize > 0)
			{
				if (!PlayM4_OpenStream(nPort, pBuffer, dwBufSize, 2 * 1024 * 1024))
				{
					dRet = PlayM4_GetLastError(nPort);
					printf("PlayM4_OpenStream Failed!:%d\n", dRet);
					break;
				}
				//设置解码回调函数 只解码不显示
				if (!PlayM4_SetDecCallBackExMend(nPort, DecCBFun, NULL, 0, pUser))
				{
					dRet = PlayM4_GetLastError(nPort);
					printf("PlayM4_SetDecCallBackExMend Failed!:%d\n", dRet);
					break;
				}

				if (!PlayM4_ThrowBFrameNum(nPort, 2))
				{
					dRet = PlayM4_GetLastError(nPort);
					printf("PlayM4_ThrowBFrameNum Failed!:%d\n", dRet);
					break;
				}

				//打开视频解码
				if (!PlayM4_Play(nPort, NULL))
				{
					dRet = PlayM4_GetLastError(nPort);
					printf("PlayM4_Play Failed!:%d\n", dRet);
					break;
				}
			}
			break;
		}

		case NET_DVR_STREAMDATA:   //码流数据
		{
			long port = portMap[reinterpret_cast<long>(pUser)];
			if (dwBufSize > 0 && port != -1)
			{
				BOOL inData = PlayM4_InputData(port, pBuffer, dwBufSize);
				while (!inData)
				{
					Sleep(10);
					inData = PlayM4_InputData(port, pBuffer, dwBufSize);
					printf("PlayM4_InputData failed \n");
				}
			}
			break;
		}
		default:
			break;
		}
	}

	static void CALLBACK g_ExceptionCallBack(DWORD dwType, LONG lUserID, LONG lHandle, void *pUser)
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

	bool Camera::login()
	{
		if (!PlayFlag)
		{
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
			lUserID = NET_DVR_Login_V30(const_cast<char *>(ip.c_str()), port, const_cast<char *>(account.c_str()), const_cast<char *>(pwd.c_str()), &struDeviceInfo);
			if (lUserID < 0)
			{
				printf("Login error, %d\n", NET_DVR_GetLastError());
				NET_DVR_Cleanup();
				return false;
			}

			char *spSerialNumber = new char[SERIALNO_LEN];
			strcpy(spSerialNumber, (char *)struDeviceInfo.sSerialNumber);
			pSerialNumber = static_cast<const char*>(spSerialNumber);
			return true;
		}
		else
			return false;
	}


	bool Camera::play(HWND display_hwnd) {
		if (!PlayFlag) 
		{
			//---------------------------------------
			//设置异常消息回调函数
			NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, const_cast<char*>(pSerialNumber.c_str()));

			NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
			struPlayInfo.hPlayWnd = display_hwnd;         //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
			struPlayInfo.lChannel = 1;       //预览通道号
			struPlayInfo.dwStreamType = 0;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
			struPlayInfo.dwLinkMode = 0;       //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP
			
			if (gm->DeviceCount<=0)
			{
				//CPU Device Automatic Allocation
				ctMap[lUserID] = unique_ptr<ColorTransform>(new ColorTransform(1280, 720, -1));
			}
			else
			{
				//GPU Device Automatic Allocation
				ctMap[lUserID] = unique_ptr<ColorTransform>(new ColorTransform(1280, 720, lUserID%gm->DeviceCount));
			}

			lRealPlayHandle = NET_DVR_RealPlay_V40(lUserID, &struPlayInfo, fRealDataCallBack, reinterpret_cast<void*>(lUserID));

			if (lRealPlayHandle<0)
			{
				NET_DVR_Logout(lUserID);
				NET_DVR_Cleanup();
				printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
				return false;
			}


			matMap[lUserID] = unique_ptr<CpuBitmap>(nullptr);
			PlayFlag = true;

			return true;
		}

		else
			return false;
	}

	bool Camera::stop()
	{
		//--------------------------------------
		if (PlayFlag)
		{
			//关闭预览
			if (lRealPlayHandle < 0)
				return false;

			if (!NET_DVR_StopRealPlay(lRealPlayHandle))
			{
				printf("NET_DVR_StopRealPlay error! Error number: %d\n", NET_DVR_GetLastError());
				return false;
			}

			lRealPlayHandle = -1;
			PlayFlag = false;

			return true;
		}
		else
			return false;
	}

	bool Camera::logout()
	{
		//--------------------------------------
		if (!PlayFlag)
		{
			//注销用户
			if (NET_DVR_Logout(lUserID))
			{
				portMap.erase(lUserID);
				matMap[lUserID].reset(nullptr);
				matMap.erase(lUserID);

				/*ctMap[lUserID].reset(nullptr);
				ctMap.erase(lUserID);*/

				lUserID = -1;

				return NET_DVR_Cleanup();
			}
			else
				return false;
		}
		else
			return false;
	}

	bool Camera::isPlaying()
	{
		return PlayFlag;
	}



	unique_ptr<CpuBitmap>&& Camera::getFrame() const {
		std::unordered_map<long, unique_ptr<CpuBitmap>>::iterator ite = matMap.find(lUserID);
		if (PlayFlag) {
			while (ite->second == nullptr)
				Sleep(1);
			return std::move(ite->second);
		}
		else {
			return std::move(nullMat);
		}
	}

	std::string Camera::getSerialNumber() const {
		return pSerialNumber;
	}

	long Camera::getUserId() const {
		return lUserID;
	}


	Camera::Camera(const char* ip, WORD port, const char* account, const char* pwd) :
		lRealPlayHandle(-1),
		lUserID(-1),
		pSerialNumber(""),
		PlayFlag(false)
	{
		this->ip = ip;
		this->port = port;
		this->account = account;
		this->pwd = pwd;
		
	}


	Camera::~Camera() {
		//stop();
	}

}
