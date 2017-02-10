/*
This Camera Management part is contributed by Yifu Zhang(zhangyifu@compilesense.com)

Copyright (c) 2014-2017 CompileSense& Glassix

*/
#pragma once
#ifndef CAMERAMANAGEMENT_H
#define CAMERAMANAGEMENT_H


#include <memory>
#include <string>
#include "ColorTransform.h"
#include "CpuBitmap.h"

namespace GCL {

    class Camera {

    public:
        explicit Camera(const char *ip, WORD port, const char *account, const char *pwd);
        ~Camera();
        void play();
        void stop();
        std::unique_ptr<CpuBitmap>&& getFrame() const;
        const char *getSerialNumber() const;
        long getUserId() const;
        bool isPlay();

    private:
        long lRealPlayHandle;
		long lUserID;

        const char *pSerialNumber;

        std::string ip;
        WORD port;
        std::string account;
        std::string pwd;

        bool PlayFlag;
    };
}

#endif // CAMERAMANAGEMENT_H
