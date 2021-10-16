#ifndef _MV_BLOBDETECTOR_H_
#define _MV_BLOBDETECTOR_H_

#include <Arduino.h>
#include "framerate.h"
#include "colorspace.h"

#define HUE_THRESHOLD           25
#define SATURATION_THRESHOLD    150

struct DetectedBlob {
    float x, y;
    float width, height;

    void serialize(String &out);
};

class BlobDetector {
    private:
        HsvColor _color;
        TaskHandle_t _task;
        Framerate _framerate;
        DetectedBlob _detected;
        SemaphoreHandle_t _signal;

    public:
        BlobDetector(RgbColor color);
        ~BlobDetector();

        void start();
        DetectedBlob get();

    private:
        void run();
        static void runStatic(void *p);
};

#endif