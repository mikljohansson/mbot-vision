#ifndef _MV_COMMON_H_
#define _MV_COMMON_H_

#include <Arduino.h>
#include <Adafruit_SSD1306.h>

#define LOG_TO_SDCARD false

extern Adafruit_SSD1306 oled;

template <typename... T>
void serialPrint(const char *message, T... args) {
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        Serial.print(buf);
    }
}

template <typename... T>
void oledPrint(const char *message, T... args) {
    if (LOG_TO_SDCARD) {
        return;
    }
    
    oled.clearDisplay();
    oled.setCursor(0, 0);
    
    int len = snprintf(NULL, 0, message, args...);
    if (len) {
        char buf[len];
        sprintf(buf, message, args...);
        oled.print(buf);
    }
    
    oled.display();
}

static void oledDisplayImage(const uint8_t *image, size_t width, size_t height) {
    oled.drawGrayscaleBitmap(0, 0, image, width, height);
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

#endif
