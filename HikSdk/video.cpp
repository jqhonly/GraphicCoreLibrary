#include "video.h"
#include "Wapper.h"

#include <string>
#include <unordered_map>

#include <unistd.h>
using namespace cv;
using namespace std;

namespace HikSdk {

#define CLIP(value) (uchar)(((value)>0xFF)?0xff:(((value)<0)?0:(value)))
#define USECOLOR 1

    static std::unordered_map<int, int> portMap;
    static std::unordered_map<int, unique_ptr<Mat>> matMap;
    static unique_ptr<cv::Mat> nullMat;

    static int yv12_to_rgb(unsigned char *yv12, unsigned char *rgb, unsigned int width, unsigned int height)
    {
        uchar *py;
        uchar *pu;
        uchar *pv;//此部分注释见另一篇yuyv_to_yv12
        uint linesize = width * 3;
        uint uvlinesize = width / 2;
        uint offset=0;
        uint offset1=0;
        uint offsety=0;
        uint offsety1=0;
        uint offsetuv=0;

        py = yv12;
        pv = py+(width*height);
        pu = pv+((width*height)/4);

        uint h=0;
        uint w=0;

        uint wy=0;
        uint huv=0;
        uint wuv=0;

        for(h=0;h<height;h+=2)
        {
            wy=0;
            wuv=0;
            offset = h * linesize;
            offset1 = (h + 1) * linesize;
            offsety = h * width;
            offsety1 = (h + 1) * width;
            offsetuv = huv * uvlinesize;

            for(w=0;w<linesize;w+=6)
            {
                /* standart: r = y0 + 1.402 (v-128) */
                /* logitech: r = y0 + 1.370705 (v-128) */
                rgb[w + offset] = CLIP(py[wy + offsety] + 1.402 * (pv[wuv + offsetuv]-128));
                /* standart: g = y0 - 0.34414 (u-128) - 0.71414 (v-128)*/
                /* logitech: g = y0 - 0.337633 (u-128)- 0.698001 (v-128)*/
                rgb[(w + 1) + offset] = CLIP(py[wy + offsety] - 0.34414 * (pu[wuv + offsetuv]-128) -0.71414*(pv[wuv + offsetuv]-128));
                /* standart: b = y0 + 1.772 (u-128) */
                /* logitech: b = y0 + 1.732446 (u-128) */
                rgb[(w + 2) + offset] = CLIP(py[wy + offsety] + 1.772 *(pu[wuv + offsetuv]-128));

                rgb[(w + 3) + offset] = CLIP(py[wy + 1 + offsety] + 1.402 * (pv[wuv + offsetuv]-128));
                rgb[(w + 4) + offset] = CLIP(py[wy +1  + offsety] - 0.34414 * (pu[wuv + offsetuv]-128) -0.71414*(pv[wuv + offsetuv]-128));
                rgb[(w + 5) + offset] = CLIP(py[wy + 1 + offsety] + 1.772 *(pu[wuv + offsetuv]-128));

                rgb[w + offset1] = CLIP(py[wy + offsety1] + 1.402 * (pv[wuv + offsetuv]-128));
                rgb[(w + 1) + offset1] = CLIP(py[wy + offsety1] - 0.34414 * (pu[wuv + offsetuv]-128) -0.71414*(pv[wuv + offsetuv]-128));
                rgb[(w + 2) + offset1] = CLIP(py[wy + offsety1] + 1.772 *(pu[wuv + offsetuv]-128));

                rgb[(w + 3) + offset1] = CLIP(py[wy + 1 + offsety1] + 1.402 * (pv[wuv + offsetuv]-128));
                rgb[(w + 4) + offset1] = CLIP(py[wy + 1 + offsety1] - 0.34414 * (pu[wuv + offsetuv]-128) -0.71414*(pv[wuv + offsetuv]-128));
                rgb[(w + 5) + offset1] = CLIP(py[wy + 1 + offsety1] + 1.772 *(pu[wuv + offsetuv]-128));

                wuv++;
                wy+=2;
            }
            huv++;
        }
        return 0;
    }

    //--------------------------------------------
    //解码回调 视频为YUV数据(YV12)，音频为PCM数据
    static void CALLBACK DecCBFun(int nPort,char * pBuf,int nSize,FRAME_INFO * pFrameInfo, int nUser,int nReserved2)
    {

        int lFrameType = pFrameInfo->nType;

        if(lFrameType == T_YV12)
        {
    #if USECOLOR
            //std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();

            //cpu decode
            //static IplImage* pImg= cvCreateImage(cvSize(pFrameInfo->nWidth,pFrameInfo->nHeight), 8, 3);//得到图像的Y分量
            Mat* pImg =new Mat(Size(pFrameInfo->nWidth,pFrameInfo->nHeight), CV_8UC4);
            //yv12_to_rgb((unsigned char *)pBuf,frame->data,pFrameInfo->nWidth,pFrameInfo->nHeight);

            //gpu decode
            //static IplImage* pImg= cvCreateImage(cvSize(pFrameInfo->nWidth,pFrameInfo->nHeight), 8, 4);//得到图像的Y分量
            bool x = YV12ToARGB((unsigned char *)pBuf, pImg->data, 1280, 720);

            //cpu decode
            //static IplImage* pImg= cvCreateImage(cvSize(pFrameInfo->nWidth,pFrameInfo->nHeight), 8, 3);//得到图像的Y分量
            //flag = cpuconvert.YV12ToRGB(pFrameInfo->nWidth,pFrameInfo->nHeight,pBuf,pImg->imageData);

            //qDebug() <<"decode time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - t0).count() / 1000 << "ms";
    #else
            static IplImage* pImg = cvCreateImage(cvSize(pFrameInfo->nWidth,pFrameInfo->nHeight), 8, 1);
            memcpy(pImg->imageData,pBuf,pFrameInfo->nWidth*pFrameInfo->nHeight);

    #endif
            //printf("%d\n",end-start);

            //lock.lockForWrite();
            //if(flag)
                //cv::Mat frame = cv::cvarrToMat(pImg, true);
            matMap[nUser].reset(pImg);
            //lock.unlock();
        }
    }


    ///实时流回调
    static void CALLBACK fRealDataCallBack(LONG lRealHandle,DWORD dwDataType,BYTE *pBuffer,DWORD dwBufSize,void *pUser)
    {
        DWORD dRet;
        switch (dwDataType)
        {
        case NET_DVR_SYSHEAD:    //系统头
        {
            LONG nPort;
            if (!PlayM4_GetPort(&nPort)) //获取播放库未使用的通道号
            {
                dRet=PlayM4_GetLastError(nPort);
                printf("PlayM4_GetPort Failed!:%d\n", dRet);
                break;
            }

            portMap[reinterpret_cast<long>(pUser)] = nPort;

            if(dwBufSize > 0)
            {
                if (!PlayM4_OpenStream(nPort,pBuffer,dwBufSize,2*1024*1024))
                {
                    dRet=PlayM4_GetLastError(nPort);
                    printf("PlayM4_OpenStream Failed!:%d\n", dRet);
                    break;
                }
                //设置解码回调函数 只解码不显示
                if (!PlayM4_SetDecCallBackExMend(nPort, DecCBFun, NULL, 0, reinterpret_cast<long>(pUser)))
                {
                    dRet=PlayM4_GetLastError(nPort);
                    printf("PlayM4_SetDecCallBackExMend Failed!:%d\n", dRet);
                    break;
                }

                if(!PlayM4_ThrowBFrameNum(nPort, 2))
                {
                    dRet=PlayM4_GetLastError(nPort);
                    printf("PlayM4_ThrowBFrameNum Failed!:%d\n", dRet);
                    break;
                }

                //打开视频解码
                if (!PlayM4_Play(nPort, NULL))
               {
                  dRet=PlayM4_GetLastError(nPort);
                  printf("PlayM4_Play Failed!:%d\n", dRet);
                  break;
                }
            }
            break;
        }

        case NET_DVR_STREAMDATA:   //码流数据
        {
            int port = portMap[reinterpret_cast<long>(pUser)];
            if (dwBufSize > 0 &&  port!= -1)
            {
                BOOL inData = PlayM4_InputData(port, pBuffer, dwBufSize);
                while (!inData)
                {
                    sleep(10);
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
        switch(dwType)
        {
        case EXCEPTION_RECONNECT:    //预览时重连
            printf("----------reconnect--------\n");
            break;
        default:
            break;
        }
    }

    void Camera::play() {
        if(!PlayFlag) {
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
            lUserID = NET_DVR_Login_V30((char *)ip.c_str(), port, (char *)account.c_str(), (char *)pwd.c_str(), &struDeviceInfo);
            if (lUserID < 0)
            {
                printf("Login error, %d\n", NET_DVR_GetLastError());
                NET_DVR_Cleanup();
                return;
            }

            char *spSerialNumber = new char[SERIALNO_LEN];
            strcpy(spSerialNumber, (const char *)struDeviceInfo.sSerialNumber);
            pSerialNumber = static_cast<const char*>(spSerialNumber);

            //---------------------------------------
            //设置异常消息回调函数
            NET_DVR_SetExceptionCallBack_V30(0, NULL, g_ExceptionCallBack, const_cast<char*>(pSerialNumber));

            NET_DVR_PREVIEWINFO struPlayInfo = { 0 };
            struPlayInfo.hPlayWnd = NULL;         //需要SDK解码时句柄设为有效值，仅取流不解码时可设为空
            struPlayInfo.lChannel = 1;       //预览通道号
            struPlayInfo.dwStreamType = 0;       //0-主码流，1-子码流，2-码流3，3-码流4，以此类推
            struPlayInfo.dwLinkMode = 0;       //0- TCP方式，1- UDP方式，2- 多播方式，3- RTP方式，4-RTP/RTSP，5-RSTP/HTTP

            lRealPlayHandle = NET_DVR_RealPlay_V40(lUserID, &struPlayInfo, fRealDataCallBack, reinterpret_cast<void*>(lUserID));

            if (lRealPlayHandle<0)
            {
                NET_DVR_Logout(lUserID);
                NET_DVR_Cleanup();
                printf("NET_DVR_RealPlay_V30 failed! Error number: %d\n", NET_DVR_GetLastError());
                return;
            }

            matMap.insert(std::unordered_map<int, unique_ptr<Mat>>::value_type(lUserID, unique_ptr<Mat>()));
            PlayFlag = true;
        }
    }

    void Camera::stop()
    {
        //--------------------------------------
        //关闭预览
        if(lRealPlayHandle < 0)
            return;

        if(!NET_DVR_StopRealPlay(lRealPlayHandle))
        {
            printf("NET_DVR_StopRealPlay error! Error number: %d\n",NET_DVR_GetLastError());
            return;
        }

        lRealPlayHandle = -1;
        PlayFlag = false;

        //注销用户
        NET_DVR_Logout(lUserID);

        portMap.erase(lUserID);
        matMap[lUserID].reset(nullptr);
        matMap.erase(lUserID);

        lUserID = -1;

        NET_DVR_Cleanup();
    }

    unique_ptr<cv::Mat>&& Camera::getFrame() const {
        std::unordered_map<int, unique_ptr<cv::Mat>>::iterator ite = matMap.find(lUserID);
        if(PlayFlag  && ite->second != nullptr ) {
            return std::move(ite->second);
        }
        else {
            return std::move(nullMat);
        }
    }

    const char *Camera::getSerialNumber() const {
        return pSerialNumber;
    }

    int Camera::getUserId() const {
        return lUserID;
    }

    Camera::Camera(const char *ip, int port, const char *account, const char *pwd):PlayFlag(false),
                                                                         pSerialNumber(NULL),
                                                                         lRealPlayHandle(-1),
                                                                         lUserID(-1) {
        this->ip = ip;
        this->port = port;
        this->account = account;
        this->pwd = pwd;
    }

    Camera::~Camera() {
        stop();
        delete[] pSerialNumber;
    }

}
