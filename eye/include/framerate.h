#ifndef _MV_FRAMERATE_H_
#define _MV_FRAMERATE_H_

#include <Arduino.h>
#include "common.h"

class Framerate {
    private:
        const char *_format;
        double windowTime = 0.0;
        unsigned long windowFrames = 0;
        unsigned long lastUpdatedWindow;
        unsigned long lastShowedFramerate = 0;

    public:
        Framerate(const char *format)
         : _format(format) {}

        void init() {
            lastUpdatedWindow = millis();
        }

        void tick() {
            // Calculate the framerate
            while (windowFrames >= 10.0) {
                windowTime -= (windowTime / (double)windowFrames);
                windowFrames--;
            }

            unsigned long ts = millis();
            windowTime += (ts - lastUpdatedWindow);
            windowFrames++;
            lastUpdatedWindow = ts;

            // Display the framerate
            if (ts - lastShowedFramerate > 5000) {
                serialPrint(_format, (double)windowFrames / (windowTime / 1000.0));
                lastShowedFramerate = ts;
            }
        }
};

#endif