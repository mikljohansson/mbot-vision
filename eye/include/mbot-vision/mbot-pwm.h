#ifndef _MV_MBOT_PWM_H_
#define _MV_MBOT_PWM_H_

#include <Arduino.h>
#include "mbot-vision/wiring.h"

class Detector;

class MBotPWM {
    private:
        TaskHandle_t _task;
        Detector &_detector;

    public:
        MBotPWM(Detector &detector);

        void begin();

    private:
        void run();
        static void runStatic(void *p);
};

#endif
