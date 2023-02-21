#include "mbot-vision/framerate.h"

#include <Arduino.h>
#include "mbot-vision/common.h"

Framerate::Framerate(const char *format)
 : _format(format) {}

void Framerate::init() {
    lastUpdatedWindow = millis();
}

void Framerate::tick() {
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
