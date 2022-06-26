#ifndef _MV_FACEDETECTOR_H_
#define _MV_FACEDETECTOR_H_

#include <Arduino.h>
#include "detector.h"
#include "framerate.h"

class ObjectDetector : public Detector {
    private:
        TaskHandle_t _task;
        Framerate _framerate;
        DetectedObject _detected;
        SemaphoreHandle_t _signal;

    public:
        ObjectDetector();
        ~ObjectDetector();

        void start();
        DetectedObject wait();
        DetectedObject get();

    private:
        void run();
        static void runStatic(void *p);
};

#endif