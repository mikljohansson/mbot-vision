#ifndef _MV_COMMON_H_
#define _MV_COMMON_H_

#include <Arduino.h>

template <typename... T>
void serialPrint(const char *message, T... args) {
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        Serial.print(buf);
    }
}

#endif
