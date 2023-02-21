
#ifndef _MV_JPEG_STREAM_H_
#define _MV_JPEG_STREAM_H_

#include <Arduino.h>
#include <esp_http_server.h>
#include "mbot-vision/framerate.h"
#include "mbot-vision/image/camera.h"
#include "mbot-vision/image/jpeg.h"
#include "mbot-vision/detection/detector.h"

class JpegStream {
    private:
        httpd_handle_t _handle;
        int _sockfd;

        TaskHandle_t _task;
        Framerate _framerate;
        bool _showDetector;
        Detector &_detector;

        bool _disconnected;

    public:
        JpegStream(httpd_req_t *request, bool showDetector, Detector &detector);
        void run();

    private:
        bool send(const void *data, size_t len);
        bool send(const char *data);
        static size_t sendStatic(void *p, size_t index, const void *data, size_t len);
};

#endif