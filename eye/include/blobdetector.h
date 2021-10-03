#ifndef _MV_BLOBDETECTOR_H_
#define _MV_BLOBDETECTOR_H_

#include <Arduino.h>
#include <JPEGDEC.h>
#include "camera.h"
#include "framerate.h"

#define SCALING     JPEG_SCALE_EIGHTH

typedef struct color {
    uint8_t r, g, b;
} color_t;

class BlobDetector {
    private:
        color_t _color;
        TaskHandle_t _task;
        Framerate _framerate;

    public:
        BlobDetector(color_t color)
         : _color(color), _framerate("Blob detector framerate: %02f\n") {}

        void start() {
            xTaskCreatePinnedToCore(runStatic, "blobDetector", 10000, this, 2, &_task, 0);
        }

    private:
        void run() {
            uint8_t *pixels = 0;
            _framerate.init();

            while (true) {
                camera_fb_t *fb = fbqueue->take();

                if (fb) {
                    if (!pixels) {
                        pixels = new uint8_t[(fb->width / SCALING) * (fb->height / SCALING) * 3];
                    }

                    if (!pixels) {
                        Serial.println("Failed to allocate buffer for uncompressed image");
                        break;
                    }

                    if (!fmt2rgb888(fb->buf, fb->len, fb->format, pixels)) {
                        Serial.println("Failed to convert framebuffer to jpeg");
                    }

                    fbqueue->release(fb);
                }
            }

            delete pixels;
        }

        static void runStatic(void *p) {
            BlobDetector *detector = (BlobDetector *)p;
            detector->run();
            vTaskDelete(NULL);
        }
};

#endif