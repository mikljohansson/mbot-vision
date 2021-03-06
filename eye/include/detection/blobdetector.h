#ifndef _MV_BLOBDETECTOR_H_
#define _MV_BLOBDETECTOR_H_

#include <Arduino.h>
#include "framerate.h"
#include "image/colorspace.h"

struct DetectedBlob {
    float x, y;
    bool detected;
    RgbColor color;

    void serialize(String &out);
};

class BlobDetector {
    private:
        RgbColor _color;
        TaskHandle_t _task;
        Framerate _framerate;
        DetectedBlob _detected;
        SemaphoreHandle_t _signal;

    public:
        BlobDetector(RgbColor color);
        ~BlobDetector();

        void start();
        DetectedBlob wait();
        DetectedBlob get();
        void draw(uint8_t *pixels, size_t width, size_t height);

    private:
        void run();
        static void runStatic(void *p);
};

#endif