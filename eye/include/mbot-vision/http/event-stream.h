
#ifndef _MV_EVENT_STREAM_H_
#define _MV_EVENT_STREAM_H_

#include <Arduino.h>
#include <esp_http_server.h>
#include "mbot-vision/framerate.h"
#include "mbot-vision/detection/detector.h"

class EventStream {
    private:
        httpd_handle_t _handle;
        int _sockfd;
        
        TaskHandle_t _task;
        Framerate _framerate;
        Detector &_detector;

    public:
        EventStream(httpd_req_t *request, Detector &detector);
        void run();

    private:
        bool send(const void *data, size_t len);
        bool send(const char *data);
};

#endif