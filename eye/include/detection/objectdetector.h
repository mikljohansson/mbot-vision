#ifndef _MV_OBJECTDETECTOR_H_
#define _MV_OBJECTDETECTOR_H_

#include <Arduino.h>
#include "detection/detector.h"
#include "image/jpeg.h"
#include "framerate.h"

class ObjectDetector : public Detector {
    private:
        TaskHandle_t _task;
        JpegDecoder _decoder;
        Framerate _framerate;
        DetectedObject _detected;
        SemaphoreHandle_t _signal;

        int8_t *_lastoutputbuf;

    public:
        ObjectDetector();
        ~ObjectDetector();

        void begin();
        DetectedObject wait();
        DetectedObject get();
        void draw(uint8_t *pixels, size_t width, size_t height);

    private:
        void run();
        static void runStatic(void *p);
};

#endif