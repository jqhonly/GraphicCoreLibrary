#ifndef VIDEO_H
#define VIDEO_H

#include <opencv2/opencv.hpp>

#include <memory>

namespace HikSdk {

    class Camera {

    public:
        explicit Camera(const char *ip, int port, const char *account, const char *pwd);
        ~Camera();
        void play();
        void stop();
        std::unique_ptr<cv::Mat>&& getFrame() const;
        const char *getSerialNumber() const;
        int getUserId() const;
        bool isPlay();

    private:
        int lRealPlayHandle;
        int lUserID;

        const char *pSerialNumber;

        std::string ip;
        int port;
        std::string account;
        std::string pwd;

        bool PlayFlag;
    };
}

#endif // VIDEO_H
