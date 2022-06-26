#ifndef _MV_DETECTOR_H_
#define _MV_DETECTOR_H_

#include <Arduino.h>

struct DetectedObject {
    float x, y;
    bool detected;

    void serialize(String &out);
};

class Detector {
    public:
        virtual DetectedObject wait() = 0;
        virtual DetectedObject get() = 0;
};

#endif
