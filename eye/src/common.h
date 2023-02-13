#ifndef _MV_COMMON_H_
#define _MV_COMMON_H_

#include <Arduino.h>
#include "wiring.h"

template <typename... T>
void serialPrint(const char *message, T... args) {
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        Serial.print(buf);
    }
}

class BenchmarkTimer {
    private:
        unsigned long started;
        unsigned long stopped;

    public:
        BenchmarkTimer()
         : started(millis()), stopped(0) {}

        unsigned long took() {
            stop();
            return stopped - started;
        }

        void stop() {
            if (!stopped) {
                stopped = millis();
            }
        }
};

static void backoff() {
    delay(random(1, 26));
}

#endif
