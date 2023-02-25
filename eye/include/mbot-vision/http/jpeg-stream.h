
#ifndef _MV_JPEG_STREAM_H_
#define _MV_JPEG_STREAM_H_

#include <Arduino.h>
#include "mbot-vision/http/async-stream.h"
#include "mbot-vision/framerate.h"
#include "mbot-vision/image/camera.h"
#include "mbot-vision/image/jpeg.h"
#include "mbot-vision/detection/detector.h"

class JpegStream : public AsyncStream {
    private:
        TaskHandle_t _task;
        Framerate _framerate;
        bool _showDetector;
        Detector &_detector;

        bool _disconnected;

    public:
        JpegStream(httpd_req_t *request, bool showDetector, Detector &detector);
        void run();

    private:
        static size_t sendStatic(void *p, size_t index, const void *data, size_t len);
};

#endif