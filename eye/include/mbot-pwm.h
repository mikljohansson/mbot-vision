#ifndef _MV_MBOT_JOYSTICK_H_
#define _MV_MBOT_JOYSTICK_H_

#include <Arduino.h>
#include "wiring.h"

class Detector;

class MBotPWM {
    private:
        TaskHandle_t _task;
        Detector &_detector;

    public:
        MBotPWM(Detector &detector);

        void start();

    private:
        void run();
        static void runStatic(void *p);
};

#endif
