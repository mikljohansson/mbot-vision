#ifndef _MV_MBOT_SERIAL_H_
#define _MV_MBOT_SERIAL_H_

#include <Arduino.h>
#include "wiring.h"

class MBotSerial {
    private:
        TaskHandle_t _task;

    public:
        void start();

    private:
        void run();
        static void runStatic(void *p);
};

#endif
