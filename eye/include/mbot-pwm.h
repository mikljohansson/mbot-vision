#ifndef _MV_MBOT_JOYSTICK_H_
#define _MV_MBOT_JOYSTICK_H_

#include <Arduino.h>
#include "wiring.h"

class BlobDetector;

class MBotPWM {
    private:
        TaskHandle_t _task;
        BlobDetector &_detector;

    public:
        MBotPWM(BlobDetector &detector);

        void start();

    private:
        void run();
        static void runStatic(void *p);
};

#endif
