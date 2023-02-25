
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include "mbot-vision/framerate.h"
#include "mbot-vision/detection/detector.h"
#include "mbot-vision/http/async-stream.h"

class EventStream : public AsyncStream {
    private:
        TaskHandle_t _task;
        Framerate _framerate;
        Detector &_detector;

    public:
        EventStream(httpd_req_t *request, Detector &detector);
        void run();
};

#endif